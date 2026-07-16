from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

from wammo.data.notepad import generate_episode, rare_event_rate
from wammo.eval.divergence_ladder import notepad_divergence_ladder
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import euler_step_toward_data, interpolate, velocity_target
from wammo.model.tokenizer import patchify, unpatchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_one import denormalize_frames, normalize_frames


def normalize_notepad_actions(actions: np.ndarray | torch.Tensor, max_delta: float, key_count: int) -> torch.Tensor:
    del key_count
    x = torch.as_tensor(actions, dtype=torch.float32).clone()
    x[..., 0:2] = x[..., 0:2] / max_delta
    x[..., 0:2] = x[..., 0:2].clamp(-1.0, 1.0)
    return x


def denormalize_notepad_actions(actions: torch.Tensor, max_delta: float, key_count: int) -> torch.Tensor:
    x = actions.clone()
    x[..., 0:2] = x[..., 0:2].clamp(-1.0, 1.0) * max_delta
    x[..., 2] = x[..., 2].round().clamp(0, 1)
    x[..., 3] = x[..., 3].round().clamp(0, key_count - 1)
    return x


class NotePadJointModel(nn.Module):
    def __init__(self, config: MicroWAMConfig, key_count: int):
        super().__init__()
        self.config = config
        self.key_count = key_count
        self.video_in = nn.Linear(config.patch_dim, config.d_model)
        self.delta_in = nn.Linear(2, config.d_model)
        self.button_in = nn.Embedding(3, config.d_model)
        self.key_in = nn.Embedding(key_count + 1, config.d_model)
        self.video_out = nn.Linear(config.d_model, config.patch_dim)
        self.delta_out = nn.Linear(config.d_model, 2)
        self.button_out = nn.Linear(config.d_model, 2)
        self.key_out = nn.Linear(config.d_model, key_count)
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
        self.video_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        self.action_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))

    def forward(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        action_drop: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_hidden, action_hidden = self.encode_hidden(
            video_patches,
            delta_actions,
            button_ids,
            key_ids,
            sigma_video,
            sigma_action,
            chunk_ids,
            action_drop,
        )
        return self.video_out(video_hidden), self.delta_out(action_hidden), self.button_out(action_hidden), self.key_out(action_hidden)

    def encode_hidden(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        action_drop: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, p, d = video_patches.shape
        if (c, p, d) != (self.config.chunk_frames, self.config.patches_per_frame, self.config.patch_dim):
            raise ValueError(f"unexpected video patch shape {tuple(video_patches.shape)}")
        if action_drop is not None:
            button_ids = torch.where(action_drop.squeeze(-1), torch.full_like(button_ids, 2), button_ids)
            key_ids = torch.where(action_drop.squeeze(-1), torch.full_like(key_ids, self.key_count), key_ids)
            delta_actions = torch.where(action_drop, torch.zeros_like(delta_actions), delta_actions)
        chunk_tokens = self.chunk_pos(chunk_ids).unsqueeze(1)
        video_tokens = self.video_in(video_patches).reshape(b, c * p, self.config.d_model)
        action_tokens = self.delta_in(delta_actions) + self.button_in(button_ids) + self.key_in(key_ids)
        video_tokens = video_tokens + self.video_pos + chunk_tokens + self.video_sigma(sigma_video.reshape(b, 1)).unsqueeze(1)
        action_tokens = action_tokens + self.action_pos + chunk_tokens + self.action_sigma(sigma_action.reshape(b, 1)).unsqueeze(1)
        hidden = self.backbone(torch.cat([video_tokens, action_tokens], dim=1))
        video_hidden = hidden[:, : c * p].reshape(b, c, p, self.config.d_model)
        action_hidden = hidden[:, c * p :]
        return video_hidden, action_hidden


class NotePadEpisodeChunks:
    def __init__(self, frames: torch.Tensor, actions: torch.Tensor, chunk_frames: int = 4):
        if frames.shape[0] % chunk_frames:
            raise ValueError("episode length must be divisible by chunk_frames")
        self.frames = frames
        self.actions = actions
        self.chunk_frames = chunk_frames
        self.starts = torch.arange(0, frames.shape[0], chunk_frames)

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.starts), (batch_size,), generator=generator)
        video_chunks, action_chunks = [], []
        for start in self.starts[idx]:
            s = int(start.item())
            video_chunks.append(self.frames[s : s + self.chunk_frames])
            action_chunks.append(self.actions[s : s + self.chunk_frames])
        video = torch.stack(video_chunks).to(device)
        actions = torch.stack(action_chunks).to(device)
        return patchify(video), actions, idx.to(device)

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video = self.frames.reshape(-1, self.chunk_frames, *self.frames.shape[1:]).to(device)
        actions = self.actions.reshape(-1, self.chunk_frames, self.actions.shape[-1]).to(device)
        chunk_ids = torch.arange(video.shape[0], device=device)
        return patchify(video), actions, chunk_ids


def make_dataset(seed: int) -> tuple[NotePadEpisodeChunks, dict[str, float]]:
    spec = load_spec()
    frames, actions = generate_episode(seed)
    frames_t = normalize_frames(frames)
    actions_t = normalize_notepad_actions(
        actions,
        max_delta=float(spec["cursor"]["max_delta"]),
        key_count=len(spec["keys"]),
    )
    metadata = {"rare_event_rate": rare_event_rate(actions)}
    return NotePadEpisodeChunks(frames_t, actions_t), metadata


def training_step(
    model: NotePadJointModel,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
    action_weight: float = 1.0,
    action_dropout: float = 0.0,
    delta_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean.shape[0]
    delta_clean = action_clean[..., 0:2]
    button_target = action_clean[..., 2].long()
    key_target = action_clean[..., 3].long()
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    delta_noise = torch.randn(delta_clean.shape, device=action_clean.device, generator=generator)
    t_video = torch.rand((b,), device=video_clean.device, generator=generator)
    t_action = torch.rand((b,), device=video_clean.device, generator=generator)
    video_noisy = interpolate(video_clean, video_noise, t_video)
    delta_noisy = interpolate(delta_clean, delta_noise, t_action)
    action_drop = None
    if action_dropout > 0:
        action_drop = torch.rand((b, 1, 1), device=video_clean.device, generator=generator) < action_dropout
    video_target = velocity_target(video_clean, video_noise)
    delta_target = velocity_target(delta_clean, delta_noise)
    video_pred, delta_pred, button_logits, key_logits = model(
        video_noisy, delta_noisy, button_target, key_target, t_video, t_action, chunk_ids, action_drop
    )
    video_loss = F.mse_loss(video_pred, video_target)
    delta_loss = F.mse_loss(delta_pred, delta_target)
    button_loss = F.cross_entropy(button_logits.reshape(-1, 2), button_target.reshape(-1))
    key_loss = F.cross_entropy(key_logits.reshape(-1, model.key_count), key_target.reshape(-1))
    action_loss = delta_weight * delta_loss + button_loss + key_loss
    loss = video_loss + action_weight * action_loss
    return loss, {
        "loss": float(loss.detach()),
        "video_loss": float(video_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "weighted_delta_loss": float((delta_weight * delta_loss).detach()),
        "button_loss": float(button_loss.detach()),
        "key_loss": float(key_loss.detach()),
    }


@torch.no_grad()
def denoise_once(
    model: NotePadJointModel,
    video_noise: torch.Tensor,
    delta_noise: torch.Tensor,
    button_ids: torch.Tensor,
    key_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = video_noise.shape[0]
    t = torch.ones((b,), device=video_noise.device)
    video_v, delta_v, button_logits, key_logits = model(video_noise, delta_noise, button_ids, key_ids, t, t, chunk_ids)
    return euler_step_toward_data(video_noise, video_v, dt=1.0), euler_step_toward_data(delta_noise, delta_v, dt=1.0), button_logits, key_logits


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: NotePadEpisodeChunks,
    device: torch.device,
    action_weight: float,
    action_dropout: float,
    delta_weight: float = 1.0,
    seed: int = 999,
) -> dict[str, float]:
    spec = load_spec()
    key_count = len(spec["keys"])
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    video_clean, action_clean, chunk_ids = dataset.all_chunks(device)
    loss, metrics = training_step(
        model, video_clean, action_clean, chunk_ids, generator, action_weight, action_dropout=0.0, delta_weight=delta_weight
    )
    del loss
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    delta_noise = torch.randn(action_clean[..., 0:2].shape, device=device, generator=generator)
    button_true = action_clean[..., 2].long()
    key_true = action_clean[..., 3].long()
    video_denoised, delta_denoised, button_logits, key_logits = denoise_once(
        model, video_noise, delta_noise, button_true, key_true, chunk_ids
    )
    raw_true = denormalize_notepad_actions(action_clean, float(spec["cursor"]["max_delta"]), key_count)
    raw_pred = raw_true.clone()
    raw_pred[..., 0:2] = delta_denoised.clamp(-1, 1) * float(spec["cursor"]["max_delta"])
    raw_pred[..., 2] = button_logits.argmax(dim=-1)
    raw_pred[..., 3] = key_logits.argmax(dim=-1)
    click_true = raw_true[..., 2] >= 0.5
    click_pred = raw_pred[..., 2] >= 0.5
    key_true = raw_true[..., 3].long()
    key_pred = raw_pred[..., 3].long().clamp(0, key_count - 1)
    key_event = key_true != 0
    key_acc = (key_pred == key_true).float().mean()
    key_event_acc = (key_pred[key_event] == key_true[key_event]).float().mean() if key_event.any() else torch.tensor(1.0, device=device)
    metrics.update(
        {
            "video_mae": float((video_denoised - video_clean).abs().mean()),
            "delta_mae_px": float((raw_pred[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
            "click_accuracy": float((click_pred == click_true).float().mean()),
            "key_accuracy": float(key_acc),
            "key_event_accuracy": float(key_event_acc),
            "action_mae": float((delta_denoised - action_clean[..., 0:2]).abs().mean()),
        }
    )
    model.train()
    return metrics


@torch.no_grad()
def write_contact_sheet(model: nn.Module, dataset: NotePadEpisodeChunks, out: Path, device: torch.device) -> None:
    model.eval()
    video_clean, action_clean, chunk_ids = dataset.all_chunks(device)
    del action_clean
    generator = torch.Generator(device=device).manual_seed(1234)
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    delta_noise = torch.randn((video_clean.shape[0], video_clean.shape[1], 2), device=device, generator=generator)
    button_ids = torch.zeros((video_clean.shape[0], video_clean.shape[1]), dtype=torch.long, device=device)
    key_ids = torch.zeros((video_clean.shape[0], video_clean.shape[1]), dtype=torch.long, device=device)
    video_denoised, _, _, _ = denoise_once(model, video_noise, delta_noise, button_ids, key_ids, chunk_ids)
    clean_frames = unpatchify(video_clean[:4].reshape(-1, video_clean.shape[2], video_clean.shape[3]), height=96, width=96).permute(0, 2, 3, 1)
    pred_frames = unpatchify(video_denoised[:4].reshape(-1, video_denoised.shape[2], video_denoised.shape[3]), height=96, width=96).permute(0, 2, 3, 1)
    clean = denormalize_frames(clean_frames.cpu())
    pred = denormalize_frames(pred_frames.cpu())
    tile, gap, label_h = 96, 6, 14
    cols = clean.shape[0]
    sheet = Image.new("RGB", (cols * tile + (cols - 1) * gap, 2 * tile + label_h + gap), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((0, 0), "clean", fill=(240, 240, 240))
    draw.text((0, tile + label_h), "denoised", fill=(240, 240, 240))
    for i in range(cols):
        x = i * (tile + gap)
        sheet.paste(Image.fromarray(clean[i].numpy(), "RGB"), (x, label_h))
        sheet.paste(Image.fromarray(pred[i].numpy(), "RGB"), (x, tile + label_h + gap))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/notepad-overfit-one"))
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-dropout", type=float, default=0.2)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--ladder-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu to run on CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    dataset, dataset_metadata = make_dataset(args.seed)
    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        action_dim=4,
        patches_per_frame=24 * 24,
        max_chunks=16,
    )
    model = NotePadJointModel(config, key_count=len(load_spec()["keys"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args) | {"out": str(args.out)}, "model": asdict(config), "dataset": dataset_metadata}
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    first_eval: dict[str, float] | None = None
    start = time.time()
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, chunk_ids = dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = training_step(
                model, video, actions, chunk_ids, noise_generator, args.action_weight, args.action_dropout, args.delta_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate(model, dataset, device, args.action_weight, args.action_dropout, args.delta_weight)
                if args.ladder_every > 0 and (step == 1 or step % args.ladder_every == 0 or step == args.steps):
                    video_all, action_all, chunk_ids_all = dataset.all_chunks(device)
                    eval_metrics.update(
                        notepad_divergence_ladder(
                            model,
                            video_all,
                            action_all,
                            chunk_ids_all,
                            key_index=load_spec()["keys"].index("h"),
                        )
                    )
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
                    f"step={step} loss={row['eval_loss']:.4f} video={row['eval_video_loss']:.4f} "
                    f"action={row['eval_action_loss']:.4f} click={row['eval_click_accuracy']:.3f} "
                    f"key={row['eval_key_accuracy']:.3f} key_event={row['eval_key_event_accuracy']:.3f}"
                )
    torch.save({"model": model.state_dict(), "config": asdict(config)}, args.out / "checkpoint.pt")
    write_contact_sheet(model, dataset, args.out / "contact_sheet.png", device)
    final_eval = evaluate(model, dataset, device, args.action_weight, args.action_dropout, args.delta_weight)
    (args.out / "summary.json").write_text(json.dumps({"first_eval": first_eval, "final_eval": final_eval}, indent=2) + "\n")


if __name__ == "__main__":
    main()
