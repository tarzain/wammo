from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

from wammo.data.notepad import generate_episode, rare_event_rate
from wammo.eval.calibrated_ladder import add_ratios, chunk_start_states, sim_channel_metrics, sim_variant_rollout
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import DeskAction, NotePadDesk, load_spec
from wammo.train.overfit_one import denormalize_frames, normalize_frames
from wammo.train.train_notepad import generate_training_dataset


@dataclass(frozen=True)
class NextFrameBaselineConfig:
    hidden_channels: int = 64
    blocks: int = 4
    key_count: int = 18
    max_delta: float = 8.0
    predict_residual: bool = True


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class NotePadNextFrameCNN(nn.Module):
    def __init__(self, config: NextFrameBaselineConfig):
        super().__init__()
        self.config = config
        action_channels = 2 + 1 + config.key_count
        hidden = config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3 + action_channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(config.blocks)])
        self.out = nn.Conv2d(hidden, 3, kernel_size=3, padding=1)
        if config.predict_residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_planes = action_to_planes(action, frame.shape[-2:], self.config.key_count)
        hidden = self.stem(torch.cat([frame, action_planes], dim=1))
        pred = self.out(self.blocks(hidden))
        if self.config.predict_residual:
            return (frame + pred).clamp(-1.0, 1.0)
        return torch.tanh(pred)


class NotePadFramePairs:
    def __init__(self, frames: np.ndarray, actions: np.ndarray, motion_oversample: bool = True):
        if frames.ndim != 5:
            raise ValueError(f"expected ETHWC frames, got {frames.shape}")
        if actions.ndim != 3:
            raise ValueError(f"expected ETA actions, got {actions.shape}")
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError("frames and actions must share episode/time dimensions")
        self.frames = frames
        self.actions = actions
        self.motion_oversample = motion_oversample
        self.pairs = self._pairs()
        self.motion_pairs = self._motion_pairs()
        self.rollout_start_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        spec = load_spec()
        self.max_delta = float(spec["cursor"]["max_delta"])
        self.key_count = len(spec["keys"])
        self.width = int(spec["canvas"]["width"])
        self.height = int(spec["canvas"]["height"])

    def sample(self, batch_size: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pairs = self.motion_pairs if self.motion_oversample and len(self.motion_pairs) else self.pairs
        pair_idx = torch.randint(len(pairs), (batch_size,), generator=generator).numpy()
        selected = pairs[pair_idx]
        input_frames = self.frames[selected[:, 0], selected[:, 1] - 1]
        target_frames = self.frames[selected[:, 0], selected[:, 1]]
        actions = self.actions[selected[:, 0], selected[:, 1]]
        return frames_to_bchw(input_frames).to(device), normalize_actions(actions, self.max_delta).to(device), frames_to_bchw(target_frames).to(device)

    def sample_rollout(
        self,
        batch_size: int,
        rollout_steps: int,
        generator: torch.Generator,
        device: torch.device,
        event_oversample_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be >= 1")
        all_starts, event_starts = self.rollout_starts(rollout_steps)
        if event_oversample_prob > 0 and len(event_starts):
            use_event = torch.rand((batch_size,), generator=generator).numpy() < event_oversample_prob
            event_idx = torch.randint(len(event_starts), (batch_size,), generator=generator).numpy()
            all_idx = torch.randint(len(all_starts), (batch_size,), generator=generator).numpy()
            selected = np.where(use_event[:, None], event_starts[event_idx], all_starts[all_idx])
        else:
            start_idx = torch.randint(len(all_starts), (batch_size,), generator=generator).numpy()
            selected = all_starts[start_idx]
        input_frames, target_frames, action_chunks = [], [], []
        for ep, start in selected:
            input_frames.append(self.frames[int(ep), int(start)])
            target_frames.append(self.frames[int(ep), int(start) + 1 : int(start) + 1 + rollout_steps])
            action_chunks.append(self.actions[int(ep), int(start) + 1 : int(start) + 1 + rollout_steps])
        return (
            frames_to_bchw(np.stack(input_frames)).to(device),
            normalize_actions(np.stack(action_chunks), self.max_delta).to(device),
            rollout_frames_to_btchw(np.stack(target_frames)).to(device),
        )

    def rollout_starts(self, rollout_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if rollout_steps in self.rollout_start_cache:
            return self.rollout_start_cache[rollout_steps]
        max_start = self.frames.shape[1] - rollout_steps - 1
        if max_start < 0:
            raise ValueError(f"rollout_steps={rollout_steps} exceeds episode length {self.frames.shape[1]}")
        starts = []
        event_starts = []
        for ep in range(self.frames.shape[0]):
            for start in range(max_start + 1):
                starts.append((ep, start))
                if self._rollout_window_has_event(ep, start, rollout_steps):
                    event_starts.append((ep, start))
        all_starts = np.asarray(starts, dtype=np.int64)
        event_array = np.asarray(event_starts, dtype=np.int64)
        if event_array.size == 0:
            event_array = event_array.reshape(0, 2)
        self.rollout_start_cache[rollout_steps] = (all_starts, event_array)
        return all_starts, event_array

    def all_pairs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_frames = self.frames[self.pairs[:, 0], self.pairs[:, 1] - 1]
        target_frames = self.frames[self.pairs[:, 0], self.pairs[:, 1]]
        actions = self.actions[self.pairs[:, 0], self.pairs[:, 1]]
        return frames_to_bchw(input_frames).to(device), normalize_actions(actions, self.max_delta).to(device), frames_to_bchw(target_frames).to(device)

    def _pairs(self) -> np.ndarray:
        pairs = [(ep, t) for ep in range(self.frames.shape[0]) for t in range(1, self.frames.shape[1])]
        return np.asarray(pairs, dtype=np.int64)

    def _motion_pairs(self) -> np.ndarray:
        pairs = []
        for ep, t in self._pairs():
            action = self.actions[ep, t]
            changed = np.abs(self.frames[ep, t].astype(np.int16) - self.frames[ep, t - 1].astype(np.int16)).mean() > 1.0
            if changed or abs(action[0]) > 0.5 or abs(action[1]) > 0.5 or action[2] > 0 or action[3] > 0:
                pairs.append((ep, t))
        return np.asarray(pairs, dtype=np.int64)

    def _rollout_window_has_event(self, ep: int, start: int, rollout_steps: int) -> bool:
        actions = self.actions[ep, start + 1 : start + 1 + rollout_steps]
        if bool(((actions[:, 2] > 0) | (actions[:, 3] > 0)).any()):
            return True
        prev = self.frames[ep, start : start + rollout_steps].astype(np.int16)
        nxt = self.frames[ep, start + 1 : start + 1 + rollout_steps].astype(np.int16)
        return bool(np.abs(nxt - prev).mean(axis=(1, 2, 3)).max() > 1.0)


def frames_to_bchw(frames: np.ndarray) -> torch.Tensor:
    return normalize_frames(frames).permute(0, 3, 1, 2).contiguous()


def rollout_frames_to_btchw(frames: np.ndarray) -> torch.Tensor:
    tensor = normalize_frames(frames)
    return tensor.permute(0, 1, 4, 2, 3).contiguous()


def bchw_to_thwc(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 2, 3, 1).contiguous()


def normalize_actions(actions: np.ndarray | torch.Tensor, max_delta: float) -> torch.Tensor:
    x = torch.as_tensor(actions, dtype=torch.float32).clone()
    x[..., 0:2] = (x[..., 0:2] / max_delta).clamp(-1.0, 1.0)
    x[..., 2] = x[..., 2].clamp(0.0, 1.0)
    return x


def action_to_planes(action: torch.Tensor, spatial: tuple[int, int], key_count: int) -> torch.Tensor:
    if action.ndim != 2 or action.shape[1] != 4:
        raise ValueError(f"expected BA actions, got {tuple(action.shape)}")
    height, width = spatial
    delta_button = action[:, 0:3].reshape(action.shape[0], 3, 1, 1).expand(-1, -1, height, width)
    key_ids = action[:, 3].long().clamp(0, key_count - 1)
    key = F.one_hot(key_ids, num_classes=key_count).float().reshape(action.shape[0], key_count, 1, 1)
    return torch.cat([delta_button, key.expand(-1, -1, height, width)], dim=1)


def changed_pixel_weight(input_frame: torch.Tensor, target_frame: torch.Tensor, changed_weight: float, threshold: float = 0.02) -> torch.Tensor:
    if changed_weight <= 0:
        return torch.ones_like(target_frame[:, :1])
    changed = (target_frame - input_frame).abs().mean(dim=1, keepdim=True) > threshold
    return torch.where(changed, torch.full_like(target_frame[:, :1], changed_weight), torch.ones_like(target_frame[:, :1]))


def baseline_step(
    model: NotePadNextFrameCNN,
    input_frame: torch.Tensor,
    action: torch.Tensor,
    target_frame: torch.Tensor,
    changed_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = model(input_frame, action)
    per_pixel = (pred - target_frame).abs().mean(dim=1, keepdim=True)
    weights = changed_pixel_weight(input_frame, target_frame, changed_weight)
    loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
    changed = weights > 1
    changed_loss = per_pixel[changed].mean() if bool(changed.any()) else torch.zeros((), device=target_frame.device)
    unchanged_loss = per_pixel[~changed].mean() if bool((~changed).any()) else torch.zeros((), device=target_frame.device)
    return loss, {
        "loss": float(loss.detach()),
        "mae": float(per_pixel.detach().mean()),
        "changed_mae": float(changed_loss.detach()),
        "unchanged_mae": float(unchanged_loss.detach()),
        "changed_pixel_rate": float(changed.float().mean()),
    }


def rollout_training_step(
    model: NotePadNextFrameCNN,
    input_frame: torch.Tensor,
    actions: torch.Tensor,
    target_frames: torch.Tensor,
    changed_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if actions.ndim != 3:
        raise ValueError(f"expected BTA actions, got {tuple(actions.shape)}")
    if target_frames.ndim != 5:
        raise ValueError(f"expected BTCHW targets, got {tuple(target_frames.shape)}")
    current = input_frame
    losses = []
    maes = []
    changed_maes = []
    unchanged_maes = []
    changed_rates = []
    for step in range(actions.shape[1]):
        target = target_frames[:, step]
        pred = model(current, actions[:, step])
        loss, metrics = baseline_step_identity(pred, current, target, changed_weight)
        losses.append(loss)
        maes.append(metrics["mae"])
        changed_maes.append(metrics["changed_mae"])
        unchanged_maes.append(metrics["unchanged_mae"])
        changed_rates.append(metrics["changed_pixel_rate"])
        current = pred
    total = torch.stack(losses).mean()
    return total, {
        "loss": float(total.detach()),
        "mae": float(np.mean(maes)),
        "changed_mae": float(np.mean(changed_maes)),
        "unchanged_mae": float(np.mean(unchanged_maes)),
        "changed_pixel_rate": float(np.mean(changed_rates)),
    }


def baseline_step_identity(
    pred: torch.Tensor,
    input_frame: torch.Tensor,
    target_frame: torch.Tensor,
    changed_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    per_pixel = (pred - target_frame).abs().mean(dim=1, keepdim=True)
    weights = changed_pixel_weight(input_frame, target_frame, changed_weight)
    loss = (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)
    changed = weights > 1
    changed_loss = per_pixel[changed].mean() if bool(changed.any()) else torch.zeros((), device=target_frame.device)
    unchanged_loss = per_pixel[~changed].mean() if bool((~changed).any()) else torch.zeros((), device=target_frame.device)
    return loss, {
        "loss": float(loss.detach()),
        "mae": float(per_pixel.detach().mean()),
        "changed_mae": float(changed_loss.detach()),
        "unchanged_mae": float(unchanged_loss.detach()),
        "changed_pixel_rate": float(changed.float().mean()),
    }


@torch.no_grad()
def evaluate_baseline(
    model: NotePadNextFrameCNN,
    dataset: NotePadFramePairs,
    device: torch.device,
    changed_weight: float,
    batch_size: int = 256,
) -> dict[str, float]:
    model.eval()
    inputs, actions, targets = dataset.all_pairs(device)
    totals = {"loss": 0.0, "mae": 0.0, "changed_mae": 0.0, "unchanged_mae": 0.0, "changed_pixel_rate": 0.0}
    count = 0
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        _, metrics = baseline_step(model, inputs[start:end], actions[start:end], targets[start:end], changed_weight)
        n = inputs[start:end].shape[0]
        count += n
        for key in totals:
            totals[key] += metrics[key] * n
    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def rollout_model(model: NotePadNextFrameCNN, frame: np.ndarray, actions: list[DeskAction], device: torch.device) -> np.ndarray:
    spec = load_spec()
    max_delta = float(spec["cursor"]["max_delta"])
    current = frames_to_bchw(frame[None]).to(device)
    rendered = []
    for action in actions:
        action_arr = np.asarray([[action.dx, action.dy, float(action.mouse_down), float(action.key)]], dtype=np.float32)
        action_t = normalize_actions(action_arr, max_delta).to(device)
        current = model(current, action_t)
        rendered.append(denormalize_frames(bchw_to_thwc(current))[0].cpu().numpy())
    return np.stack(rendered)


@torch.no_grad()
def baseline_calibrated_ladder(
    model: NotePadNextFrameCNN,
    device: torch.device,
    episodes: int,
    seed: int,
    horizons: tuple[int, ...],
) -> dict[str, object]:
    model.eval()
    key_index = load_spec()["keys"].index("h")
    max_horizon = max(horizons)
    states = []
    for episode in range(episodes):
        states.extend(chunk_start_states(seed + episode))
    simulator: dict[str, float] = {}
    model_metrics: dict[str, float] = {}
    for channel in ("cursor", "click", "key"):
        sim_positive, sim_negative, model_positive, model_negative = [], [], [], []
        for state in states:
            sim_positive.append(sim_variant_rollout(state, channel, positive=True, key_index=key_index, frames=max_horizon))
            sim_negative.append(sim_variant_rollout(state, channel, positive=False, key_index=key_index, frames=max_horizon))
            model_positive.append(model_variant_rollout(model, state, channel, True, key_index, max_horizon, device))
            model_negative.append(model_variant_rollout(model, state, channel, False, key_index, max_horizon, device))
        simulator.update(sim_channel_metrics(channel, np.stack(sim_positive), np.stack(sim_negative), horizons))
        model_metrics.update(model_channel_metrics(channel, np.stack(model_positive), np.stack(model_negative), horizons))
    return {
        "eval_episodes": episodes,
        "eval_seed": seed,
        "horizons": list(horizons),
        "simulator": simulator,
        "model_ladder": model_metrics,
        "calibrated_ratio": add_ratios(model_metrics, simulator),
    }


def model_variant_rollout(
    model: NotePadNextFrameCNN,
    state: NotePadDesk,
    channel: str,
    positive: bool,
    key_index: int,
    frames: int,
    device: torch.device,
) -> np.ndarray:
    action = sim_model_action(channel, positive, key_index, state)
    return rollout_model(model, state.render(), [action] * frames, device)


def sim_model_action(channel: str, positive: bool, key_index: int, desk: NotePadDesk) -> DeskAction:
    max_delta = float(desk.spec["cursor"]["max_delta"])
    if channel == "cursor":
        return DeskAction(max_delta if positive else -max_delta, 0.0, False, 0)
    if channel == "click":
        return DeskAction(0.0, 0.0, positive, 0)
    if channel == "key":
        return DeskAction(0.0, 0.0, False, key_index if positive else 0)
    raise ValueError(f"unknown channel {channel}")


def model_channel_metrics(channel: str, positive: np.ndarray, negative: np.ndarray, horizons: tuple[int, ...]) -> dict[str, float]:
    positive_norm = patchify(normalize_frames(positive))
    negative_norm = patchify(normalize_frames(negative))
    patch_mse = (positive_norm - negative_norm).pow(2)
    raw_abs = np.abs(positive.astype(np.int16) - negative.astype(np.int16))
    out: dict[str, float] = {}
    for horizon in horizons:
        frame = horizon - 1
        out[f"ar_ladder_{channel}_h{horizon}"] = float(patch_mse[:, frame].mean())
        out[f"ar_raw_abs_{channel}_h{horizon}"] = float(raw_abs[:, frame].mean())
        out[f"ar_material_frac_{channel}_h{horizon}"] = float((raw_abs[:, frame] > 5).mean())
    return out


def make_eval_pairs(seed: int) -> tuple[NotePadFramePairs, dict[str, float]]:
    frames, actions = generate_episode(seed)
    return NotePadFramePairs(frames[None], actions[None], motion_oversample=False), {
        "eval_seed": seed,
        "eval_rare_event_rate": rare_event_rate(actions),
    }


def write_contact_sheet(
    model: NotePadNextFrameCNN,
    dataset: NotePadFramePairs,
    path: Path,
    device: torch.device,
    examples: int = 8,
) -> None:
    inputs, actions, targets = dataset.all_pairs(device)
    with torch.no_grad():
        preds = model(inputs[:examples], actions[:examples])
    input_u8 = denormalize_frames(bchw_to_thwc(inputs[:examples])).cpu().numpy()
    pred_u8 = denormalize_frames(bchw_to_thwc(preds)).cpu().numpy()
    target_u8 = denormalize_frames(bchw_to_thwc(targets[:examples])).cpu().numpy()
    width, height = input_u8.shape[2], input_u8.shape[1]
    sheet = Image.new("RGB", (width * examples, height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(examples):
        sheet.paste(Image.fromarray(input_u8[i]), (i * width, 0))
        sheet.paste(Image.fromarray(pred_u8[i]), (i * width, height))
        sheet.paste(Image.fromarray(target_u8[i]), (i * width, height * 2))
    draw.text((2, 2), "input", fill=(255, 255, 255))
    draw.text((2, height + 2), "pred", fill=(255, 255, 255))
    draw.text((2, height * 2 + 2), "target", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/notepad-next-frame-baseline"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--ladder-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--predict-residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--changed-pixel-weight", type=float, default=64.0)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--event-oversample-prob", type=float, default=0.0)
    parser.add_argument("--motion-oversample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)

    train_frames, train_actions, train_metadata = generate_training_dataset(
        args.episodes,
        args.seed,
        progress_every=args.generate_progress_every,
    )
    train_dataset = NotePadFramePairs(train_frames, train_actions, motion_oversample=args.motion_oversample)
    eval_dataset, eval_metadata = make_eval_pairs(args.eval_seed)
    spec = load_spec()
    config = NextFrameBaselineConfig(
        hidden_channels=args.hidden_channels,
        blocks=args.blocks,
        key_count=len(spec["keys"]),
        max_delta=float(spec["cursor"]["max_delta"]),
        predict_residual=args.predict_residual,
    )
    model = NotePadNextFrameCNN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    rollout_starts, event_starts = train_dataset.rollout_starts(args.rollout_steps)
    payload = {
        "args": vars(args) | {"out": str(args.out)},
        "model": asdict(config),
        "train_dataset": train_metadata
        | {
            "pairs": int(len(train_dataset.pairs)),
            "motion_pairs": int(len(train_dataset.motion_pairs)),
            "rollout_starts": int(len(rollout_starts)),
            "event_rollout_starts": int(len(event_starts)),
        },
        "eval_dataset": eval_metadata,
    }
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    first_eval: dict[str, float] | None = None
    start = time.time()
    metrics_path = args.out / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            if args.rollout_steps > 1:
                inputs, actions, targets = train_dataset.sample_rollout(
                    args.batch_size,
                    args.rollout_steps,
                    sample_generator,
                    device,
                    event_oversample_prob=args.event_oversample_prob,
                )
            else:
                inputs, actions, targets = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            if args.rollout_steps > 1:
                loss, train_metrics = rollout_training_step(model, inputs, actions, targets, args.changed_pixel_weight)
            else:
                loss, train_metrics = baseline_step(model, inputs, actions, targets, args.changed_pixel_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate_baseline(model, eval_dataset, device, args.changed_pixel_weight)
                if first_eval is None:
                    first_eval = copy.deepcopy(eval_metrics)
                row = {
                    "step": step,
                    "elapsed_sec": round(time.time() - start, 3),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"eval_{k}": v for k, v in eval_metrics.items()},
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"step={step} loss={row['eval_loss']:.4f} mae={row['eval_mae']:.4f} "
                    f"changed={row['eval_changed_mae']:.4f} unchanged={row['eval_unchanged_mae']:.4f}"
                )
            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
                save_checkpoint(model, config, args.out / f"checkpoint_step_{step}.pt", step)
            if args.ladder_every > 0 and (step % args.ladder_every == 0 or step == args.steps):
                ladder = baseline_calibrated_ladder(model, device, args.eval_episodes, args.eval_seed, horizons=(1, 2, 4, 8, 16))
                analysis_path = args.out / "analysis" / f"calibrated_ladder_step_{step}.json"
                analysis_path.parent.mkdir(parents=True, exist_ok=True)
                analysis_path.write_text(json.dumps(ladder, indent=2) + "\n", encoding="utf-8")

    save_checkpoint(model, config, args.out / "checkpoint.pt", args.steps)
    write_contact_sheet(model, eval_dataset, args.out / "contact_sheet.png", device)
    final_eval = evaluate_baseline(model, eval_dataset, device, args.changed_pixel_weight)
    final_ladder = baseline_calibrated_ladder(model, device, args.eval_episodes, args.eval_seed, horizons=(1, 2, 4, 8, 16))
    (args.out / "summary.json").write_text(
        json.dumps({"first_eval": first_eval, "final_eval": final_eval, "final_ladder": final_ladder}, indent=2) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(model: NotePadNextFrameCNN, config: NextFrameBaselineConfig, path: Path, step: int) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": step}, path)


if __name__ == "__main__":
    main()
