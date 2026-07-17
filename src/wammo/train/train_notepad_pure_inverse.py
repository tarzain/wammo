from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from wammo.model.dit import MicroWAMConfig
from wammo.model.tokenizer import add_coordinate_channels, patchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import normalize_notepad_actions
from wammo.train.overfit_one import normalize_frames
from wammo.train.train_notepad import generate_training_dataset
from wammo.train.train_notepad_binned_delta import DELTA_BINS, bins_to_delta_norm, delta_to_bins


def augment_frames(frames: torch.Tensor, input_mode: str) -> torch.Tensor:
    channels = [frames]
    if "coord" in input_mode:
        channels = [add_coordinate_channels(frames)]
    if "diff" in input_mode:
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        channels.append(diff)
    if input_mode == "rgb":
        return frames
    if input_mode == "coord":
        return channels[0]
    if input_mode in {"diff", "coord-diff"}:
        return torch.cat(channels, dim=-1)
    raise ValueError(f"unknown input mode {input_mode}")


def patch_dim_for_mode(input_mode: str, patch_size: int) -> int:
    channel_count = {"rgb": 3, "coord": 5, "diff": 6, "coord-diff": 8}[input_mode]
    return channel_count * patch_size * patch_size


class NotePadPureInverseChunks:
    def __init__(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        input_mode: str,
        patch_size: int = 4,
        chunk_frames: int = 4,
        motion_oversample: bool = True,
    ):
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError("frames and actions must share episode/time dimensions")
        if frames.shape[1] % chunk_frames:
            raise ValueError("episode length must be divisible by chunk_frames")
        self.frames = frames
        self.actions = actions
        self.input_mode = input_mode
        self.patch_size = patch_size
        self.chunk_frames = chunk_frames
        self.chunks_per_episode = frames.shape[1] // chunk_frames
        self.motion_oversample = motion_oversample
        self.motion_pairs = self._motion_pairs()
        spec = load_spec()
        self.max_delta = float(spec["cursor"]["max_delta"])
        self.key_count = len(spec["keys"])

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.motion_oversample and len(self.motion_pairs):
            pair_idx = torch.randint(len(self.motion_pairs), (batch_size,), generator=generator).numpy()
            pairs = self.motion_pairs[pair_idx]
            episode_idx = pairs[:, 0]
            chunk_idx = pairs[:, 1]
        else:
            episode_idx = torch.randint(self.frames.shape[0], (batch_size,), generator=generator).numpy()
            chunk_idx = torch.randint(self.chunks_per_episode, (batch_size,), generator=generator).numpy()
        video_chunks, action_chunks = [], []
        for ep, chunk in zip(episode_idx, chunk_idx, strict=True):
            start = int(chunk) * self.chunk_frames
            video_chunks.append(self.frames[int(ep), start : start + self.chunk_frames])
            action_chunks.append(self.actions[int(ep), start : start + self.chunk_frames])
        video = normalize_frames(np.stack(video_chunks)).to(device)
        actions = normalize_notepad_actions(np.stack(action_chunks), self.max_delta, self.key_count).to(device)
        return patchify(augment_frames(video, self.input_mode), patch_size=self.patch_size), actions, torch.as_tensor(chunk_idx, dtype=torch.long, device=device)

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = normalize_frames(self.frames.reshape(-1, *self.frames.shape[2:])).reshape(
            -1, self.chunk_frames, *self.frames.shape[2:]
        )
        actions = normalize_notepad_actions(
            self.actions.reshape(-1, self.actions.shape[-1]), self.max_delta, self.key_count
        ).reshape(-1, self.chunk_frames, self.actions.shape[-1])
        chunk_ids = torch.arange(self.chunks_per_episode).repeat(self.frames.shape[0])
        video = augment_frames(video, self.input_mode)
        return patchify(video, patch_size=self.patch_size).to(device), actions.to(device), chunk_ids.to(device)

    def _motion_pairs(self) -> np.ndarray:
        pairs = []
        for ep in range(self.actions.shape[0]):
            for chunk in range(self.chunks_per_episode):
                start = chunk * self.chunk_frames
                deltas = self.actions[ep, start : start + self.chunk_frames, 0:2]
                visible = deltas[1:] if self.chunk_frames > 1 else deltas
                if np.abs(visible).max() > 0.5:
                    pairs.append((ep, chunk))
        return np.asarray(pairs, dtype=np.int64)


class NotePadPureInverseModel(nn.Module):
    def __init__(self, config: MicroWAMConfig):
        super().__init__()
        self.config = config
        self.video_in = nn.Linear(config.patch_dim, config.d_model)
        self.query = nn.Parameter(torch.zeros(1, config.chunk_frames, config.d_model))
        self.video_pos = nn.Parameter(torch.zeros(1, config.chunk_frames * config.patches_per_frame, config.d_model))
        self.action_pos = nn.Parameter(torch.zeros(1, config.chunk_frames, config.d_model))
        self.chunk_pos = nn.Embedding(config.max_chunks, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.dx_out = nn.Linear(config.d_model, DELTA_BINS)
        self.dy_out = nn.Linear(config.d_model, DELTA_BINS)

    def forward(self, video_patches: torch.Tensor, chunk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, p, d = video_patches.shape
        if (c, p, d) != (self.config.chunk_frames, self.config.patches_per_frame, self.config.patch_dim):
            raise ValueError(f"unexpected video patch shape {tuple(video_patches.shape)}")
        chunk_tokens = self.chunk_pos(chunk_ids).unsqueeze(1)
        video_tokens = self.video_in(video_patches).reshape(b, c * p, self.config.d_model)
        video_tokens = video_tokens + self.video_pos + chunk_tokens
        queries = self.query.expand(b, -1, -1) + self.action_pos + chunk_tokens
        hidden = self.backbone(torch.cat([video_tokens, queries], dim=1))
        action_hidden = hidden[:, c * p :]
        return self.dx_out(action_hidden), self.dy_out(action_hidden)


def visible_mask(actions: torch.Tensor, mask_first_frame: bool) -> torch.Tensor:
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    if mask_first_frame:
        mask[:, 0] = False
    return mask


def pure_inverse_step(
    model: NotePadPureInverseModel,
    video: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    mask_first_frame: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    dx_logits, dy_logits = model(video, chunk_ids)
    delta_targets = delta_to_bins(actions[..., 0:2])
    mask = visible_mask(actions, mask_first_frame)
    dx_loss = F.cross_entropy(dx_logits[mask], delta_targets[..., 0][mask])
    dy_loss = F.cross_entropy(dy_logits[mask], delta_targets[..., 1][mask])
    loss = 0.5 * (dx_loss + dy_loss)
    metrics = evaluate_delta_logits(dx_logits, dy_logits, actions, model_max_delta(), mask)
    metrics.update({"loss": float(loss.detach()), "dx_loss": float(dx_loss.detach()), "dy_loss": float(dy_loss.detach())})
    return loss, metrics


def model_max_delta() -> float:
    return float(load_spec()["cursor"]["max_delta"])


@torch.no_grad()
def evaluate_pure_inverse(
    model: NotePadPureInverseModel,
    dataset: NotePadPureInverseChunks,
    device: torch.device,
    mask_first_frame: bool = True,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        video, actions, chunk_ids = dataset.all_chunks(device)
        dx_logits, dy_logits = model(video, chunk_ids)
        mask = visible_mask(actions, mask_first_frame)
        return evaluate_delta_logits(dx_logits, dy_logits, actions, dataset.max_delta, mask)
    finally:
        model.train(was_training)


def evaluate_delta_logits(
    dx_logits: torch.Tensor,
    dy_logits: torch.Tensor,
    actions: torch.Tensor,
    max_delta: float,
    mask: torch.Tensor,
) -> dict[str, float]:
    pred = torch.stack(
        [bins_to_delta_norm(dx_logits.argmax(dim=-1)), bins_to_delta_norm(dy_logits.argmax(dim=-1))],
        dim=-1,
    ) * max_delta
    true = actions[..., 0:2] * max_delta
    motion = (true.abs().amax(dim=-1) > 0.5) & mask
    visible = mask
    if motion.any():
        pred_motion = pred[motion]
        true_motion = true[motion]
        motion_mae = float((pred_motion - true_motion).abs().mean())
        pred_abs = float(pred_motion.abs().mean())
        zero_mae = float(true_motion.abs().mean())
    else:
        motion_mae = pred_abs = zero_mae = 0.0
    visible_mae = float((pred[visible] - true[visible]).abs().mean()) if visible.any() else 0.0
    return {
        "visible_delta_mae_px": visible_mae,
        "motion_delta_mae_px": motion_mae,
        "motion_zero_delta_mae_px": zero_mae,
        "motion_pred_abs_mean_px": pred_abs,
        "motion_frames": int(motion.sum()),
        "visible_frames": int(visible.sum()),
        "motion_rate": float(motion.float().mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["rgb", "coord", "diff", "coord-diff"], default="coord")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=400_000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--no-mask-first-frame", action="store_true")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--generate-progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)

    train_frames, train_actions, train_metadata = generate_training_dataset(
        args.episodes, args.seed, progress_every=args.generate_progress_every
    )
    eval_frames, eval_actions, eval_metadata = generate_training_dataset(
        args.eval_episodes, args.eval_seed, progress_every=0
    )
    train_dataset = NotePadPureInverseChunks(train_frames, train_actions, args.input_mode, args.patch_size)
    eval_dataset = NotePadPureInverseChunks(eval_frames, eval_actions, args.input_mode, args.patch_size, motion_oversample=False)
    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        patch_dim=patch_dim_for_mode(args.input_mode, args.patch_size),
        action_dim=4,
        patches_per_frame=(96 // args.patch_size) * (96 // args.patch_size),
        max_chunks=16,
    )
    model = NotePadPureInverseModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    mask_first_frame = not args.no_mask_first_frame
    first_eval = None
    start = time.time()
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = pure_inverse_step(model, video, actions, chunk_ids, mask_first_frame)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate_pure_inverse(model, eval_dataset, device, mask_first_frame)
                if first_eval is None:
                    first_eval = eval_metrics
                row = {
                    "step": step,
                    "elapsed_sec": round(time.time() - start, 3),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"eval_{k}": v for k, v in eval_metrics.items()},
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"step={step} loss={row['eval_visible_delta_mae_px']:.3f}px "
                    f"motion={row['eval_motion_delta_mae_px']:.3f}px "
                    f"zero={row['eval_motion_zero_delta_mae_px']:.3f}px "
                    f"pred_abs={row['eval_motion_pred_abs_mean_px']:.3f}px"
                )
    final_eval = evaluate_pure_inverse(model, eval_dataset, device, mask_first_frame)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "model_kind": "notepad_pure_inverse",
            "input_mode": args.input_mode,
            "patch_size": args.patch_size,
            "mask_first_frame": mask_first_frame,
        },
        args.out / "checkpoint.pt",
    )
    summary = {
        "model_kind": "notepad_pure_inverse",
        "args": vars(args) | {"out": str(args.out), "mask_first_frame": mask_first_frame},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
        "first_eval": first_eval,
        "final_eval": final_eval,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
