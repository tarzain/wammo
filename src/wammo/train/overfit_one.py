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

from wammo.cursor_world.sim import load_spec
from wammo.data.generate import generate_episode
from wammo.model.dit import MicroWAMConfig, MicroWAMDiT
from wammo.model.flow import euler_step_toward_data, interpolate, velocity_target
from wammo.model.tokenizer import patchify, unpatchify


def normalize_frames(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames).float().div(127.5).sub(1.0)


def denormalize_frames(frames: torch.Tensor) -> torch.Tensor:
    return frames.add(1.0).mul(127.5).clamp(0, 255).to(torch.uint8)


def normalize_actions(actions: np.ndarray | torch.Tensor, max_delta: float) -> torch.Tensor:
    x = torch.as_tensor(actions, dtype=torch.float32).clone()
    x[..., 0:2] = x[..., 0:2] / max_delta
    x[..., 2] = x[..., 2] * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)


def denormalize_actions(actions: torch.Tensor, max_delta: float) -> torch.Tensor:
    x = actions.clone()
    x[..., 0:2] = x[..., 0:2].clamp(-1.0, 1.0) * max_delta
    x[..., 2] = (x[..., 2].clamp(-1.0, 1.0) + 1.0) * 0.5
    return x


class OneEpisodeChunks:
    def __init__(self, frames: torch.Tensor, actions: torch.Tensor, chunk_frames: int = 4):
        if frames.ndim != 4:
            raise ValueError(f"expected THWC frames, got {tuple(frames.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"expected TA actions, got {tuple(actions.shape)}")
        if frames.shape[0] != actions.shape[0]:
            raise ValueError("frames and actions must have matching time length")
        self.frames = frames
        self.actions = actions
        self.chunk_frames = chunk_frames
        self.starts = torch.arange(0, frames.shape[0], chunk_frames)
        if frames.shape[0] % chunk_frames:
            raise ValueError("episode length must be divisible by chunk_frames")

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self.starts), (batch_size,), generator=generator)
        video_chunks = []
        action_chunks = []
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


def make_dataset(seed: int, chunk_frames: int = 4) -> OneEpisodeChunks:
    spec = load_spec()
    frames, actions = generate_episode(seed)
    frames_t = normalize_frames(frames)
    actions_t = normalize_actions(actions, max_delta=float(spec["cursor"]["max_delta"]))
    return OneEpisodeChunks(frames_t, actions_t, chunk_frames=chunk_frames)


def training_step(
    model: nn.Module,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean.shape[0]
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    action_noise = torch.randn(action_clean.shape, device=action_clean.device, generator=generator)
    t_video = torch.rand((b,), device=video_clean.device, generator=generator)
    t_action = torch.rand((b,), device=video_clean.device, generator=generator)

    video_noisy = interpolate(video_clean, video_noise, t_video)
    action_noisy = interpolate(action_clean, action_noise, t_action)
    video_target = velocity_target(video_clean, video_noise)
    action_target = velocity_target(action_clean, action_noise)
    video_pred, action_pred = model(video_noisy, action_noisy, t_video, t_action, chunk_ids)
    video_loss = F.mse_loss(video_pred, video_target)
    action_loss = F.mse_loss(action_pred, action_target)
    loss = video_loss + action_loss
    return loss, {"loss": float(loss.detach()), "video_loss": float(video_loss.detach()), "action_loss": float(action_loss.detach())}


@torch.no_grad()
def denoise_once(
    model: nn.Module, video_noise: torch.Tensor, action_noise: torch.Tensor, chunk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    b = video_noise.shape[0]
    t = torch.ones((b,), device=video_noise.device)
    video_v, action_v = model(video_noise, action_noise, t, t, chunk_ids)
    return euler_step_toward_data(video_noise, video_v, dt=1.0), euler_step_toward_data(action_noise, action_v, dt=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: OneEpisodeChunks, device: torch.device, seed: int = 999) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    video_clean, action_clean, chunk_ids = dataset.all_chunks(device)
    loss, metrics = training_step(model, video_clean, action_clean, chunk_ids, generator)
    del loss
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    action_noise = torch.randn(action_clean.shape, device=device, generator=generator)
    video_denoised, action_denoised = denoise_once(model, video_noise, action_noise, chunk_ids)
    action_mae = (action_denoised - action_clean).abs().mean()
    click_pred = action_denoised[..., 2] > 0
    click_true = action_clean[..., 2] > 0
    metrics.update(
        {
            "action_mae": float(action_mae),
            "click_accuracy": float((click_pred == click_true).float().mean()),
            "video_mae": float((video_denoised - video_clean).abs().mean()),
        }
    )
    model.train()
    return metrics


@torch.no_grad()
def write_contact_sheet(model: nn.Module, dataset: OneEpisodeChunks, out: Path, device: torch.device) -> None:
    model.eval()
    video_clean, action_clean, chunk_ids = dataset.all_chunks(device)
    generator = torch.Generator(device=device).manual_seed(1234)
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    action_noise = torch.randn(action_clean.shape, device=device, generator=generator)
    video_denoised, _ = denoise_once(model, video_noise, action_noise, chunk_ids)

    clean_frames = unpatchify(video_clean[:4].reshape(-1, video_clean.shape[2], video_clean.shape[3])).permute(0, 2, 3, 1)
    pred_frames = unpatchify(video_denoised[:4].reshape(-1, video_denoised.shape[2], video_denoised.shape[3])).permute(0, 2, 3, 1)
    clean = denormalize_frames(clean_frames.cpu())
    pred = denormalize_frames(pred_frames.cpu())

    tile = 64
    gap = 6
    label_h = 14
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/overfit-one"))
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu to run on CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)

    dataset = make_dataset(args.seed)
    config = MicroWAMConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    model = MicroWAMDiT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    config_payload = {"args": vars(args) | {"out": str(args.out)}, "model": asdict(config)}
    (args.out / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    metrics_path = args.out / "metrics.jsonl"
    start = time.time()
    first_eval: dict[str, float] | None = None

    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, chunk_ids = dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = training_step(model, video, actions, chunk_ids, noise_generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate(model, dataset, device)
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
                    f"step={step} loss={row['eval_loss']:.4f} "
                    f"video={row['eval_video_loss']:.4f} action={row['eval_action_loss']:.4f} "
                    f"action_mae={row['eval_action_mae']:.4f} click_acc={row['eval_click_accuracy']:.3f}"
                )

    torch.save({"model": model.state_dict(), "config": asdict(config)}, args.out / "checkpoint.pt")
    write_contact_sheet(model, dataset, args.out / "contact_sheet.png", device)
    final_metrics = evaluate(model, dataset, device)
    summary = {"first_eval": first_eval, "final_eval": final_metrics}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
