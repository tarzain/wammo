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

from wammo.eval.divergence_ladder import notepad_divergence_ladder
from wammo.eval.notepad_pixels import cursor_positions_from_actions
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import euler_step_toward_data, interpolate, velocity_target
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import (
    NotePadJointModel,
    denormalize_notepad_actions,
    normalize_notepad_actions,
    write_contact_sheet,
)
from wammo.train.overfit_one import normalize_frames
from wammo.train.train_notepad_pure_inverse import augment_frames
from wammo.train.train_notepad import generate_training_dataset, make_eval_dataset
from wammo.train.train_notepad_binned_delta import DELTA_BINS, bins_to_delta_norm, delta_to_bins


def normalize_positions(positions: np.ndarray, width: int, height: int) -> torch.Tensor:
    out = torch.as_tensor(positions, dtype=torch.float32).clone()
    out[..., 0] = out[..., 0] / max(1, width - 1)
    out[..., 1] = out[..., 1] / max(1, height - 1)
    return out


def denormalize_positions(positions: torch.Tensor, width: int, height: int) -> torch.Tensor:
    out = positions.clone()
    out[..., 0] = out[..., 0] * max(1, width - 1)
    out[..., 1] = out[..., 1] * max(1, height - 1)
    return out


class NotePadHybridModel(NotePadJointModel):
    def __init__(
        self,
        config: MicroWAMConfig,
        key_count: int,
        head_sigma_conditioned: bool = False,
        video_output_dim: int = 4 * 4 * 3,
    ):
        super().__init__(config, key_count)
        self.head_sigma_conditioned = head_sigma_conditioned
        self.video_output_dim = video_output_dim
        self.video_out = nn.Linear(config.d_model, video_output_dim)
        self.dx_out = nn.Linear(config.d_model, DELTA_BINS)
        self.dy_out = nn.Linear(config.d_model, DELTA_BINS)
        self.cursor_out = nn.Linear(config.d_model, 2)
        if head_sigma_conditioned:
            self.action_head_sigma = nn.Sequential(
                nn.Linear(1, config.d_model),
                nn.SiLU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.cursor_head_sigma = nn.Sequential(
                nn.Linear(1, config.d_model),
                nn.SiLU(),
                nn.Linear(config.d_model, config.d_model),
            )
        else:
            self.action_head_sigma = None
            self.cursor_head_sigma = None

    def forward_all(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        action_drop: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        frame_hidden = video_hidden.mean(dim=2)
        if self.head_sigma_conditioned:
            action_hidden = action_hidden + self.action_head_sigma(sigma_action.reshape(-1, 1)).unsqueeze(1)
            frame_hidden = frame_hidden + self.cursor_head_sigma(sigma_video.reshape(-1, 1)).unsqueeze(1)
        return (
            self.video_out(video_hidden),
            self.delta_out(action_hidden),
            self.dx_out(action_hidden),
            self.dy_out(action_hidden),
            self.button_out(action_hidden),
            self.key_out(action_hidden),
            self.cursor_out(frame_hidden).sigmoid(),
        )

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


def hybrid_video_patches(frames: torch.Tensor, input_mode: str, patch_size: int = 4) -> torch.Tensor:
    return patchify(augment_frames(frames, input_mode), patch_size=patch_size)


def sample_sigma_pair(
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    corner_weight: float = 0.0,
    corner_low: float = 0.0,
    corner_high: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    t_video = torch.rand((batch_size,), device=device, generator=generator)
    t_action = torch.rand((batch_size,), device=device, generator=generator)
    corner_count = 0
    inverse_corner_count = 0
    generation_corner_count = 0
    if corner_weight > 0:
        use_corner = torch.rand((batch_size,), device=device, generator=generator) < corner_weight
        inverse_corner = torch.rand((batch_size,), device=device, generator=generator) < 0.5
        inverse_mask = use_corner & inverse_corner
        generation_mask = use_corner & ~inverse_corner
        t_video = torch.where(inverse_mask, torch.full_like(t_video, corner_low), t_video)
        t_action = torch.where(inverse_mask, torch.full_like(t_action, corner_high), t_action)
        t_video = torch.where(generation_mask, torch.full_like(t_video, corner_high), t_video)
        t_action = torch.where(generation_mask, torch.full_like(t_action, corner_low), t_action)
        corner_count = int(use_corner.sum())
        inverse_corner_count = int(inverse_mask.sum())
        generation_corner_count = int(generation_mask.sum())
    return t_video, t_action, {
        "sigma_video_mean": float(t_video.detach().mean()),
        "sigma_action_mean": float(t_action.detach().mean()),
        "sigma_corner_rate": float(corner_count / max(1, batch_size)),
        "sigma_inverse_corner_rate": float(inverse_corner_count / max(1, batch_size)),
        "sigma_generation_corner_rate": float(generation_corner_count / max(1, batch_size)),
    }


class NotePadHybridChunks:
    def __init__(self, frames: np.ndarray, actions: np.ndarray, chunk_frames: int = 4, motion_oversample: bool = True):
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError("frames and actions must share episode/time dimensions")
        if frames.shape[1] % chunk_frames:
            raise ValueError("episode length must be divisible by chunk_frames")
        self.frames = frames
        self.actions = actions
        self.positions = cursor_positions_from_actions(actions)
        self.chunk_frames = chunk_frames
        self.chunks_per_episode = frames.shape[1] // chunk_frames
        self.motion_oversample = motion_oversample
        self.motion_pairs = self._motion_pairs()
        spec = load_spec()
        self.max_delta = float(spec["cursor"]["max_delta"])
        self.key_count = len(spec["keys"])
        self.width = int(spec["canvas"]["width"])
        self.height = int(spec["canvas"]["height"])

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.motion_oversample and len(self.motion_pairs):
            pair_idx = torch.randint(len(self.motion_pairs), (batch_size,), generator=generator).numpy()
            pairs = self.motion_pairs[pair_idx]
            episode_idx = pairs[:, 0]
            chunk_idx = pairs[:, 1]
        else:
            episode_idx = torch.randint(self.frames.shape[0], (batch_size,), generator=generator).numpy()
            chunk_idx = torch.randint(self.chunks_per_episode, (batch_size,), generator=generator).numpy()
        video_chunks, action_chunks, position_chunks = [], [], []
        for ep, chunk in zip(episode_idx, chunk_idx, strict=True):
            start = int(chunk) * self.chunk_frames
            video_chunks.append(self.frames[int(ep), start : start + self.chunk_frames])
            action_chunks.append(self.actions[int(ep), start : start + self.chunk_frames])
            position_chunks.append(self.positions[int(ep), start : start + self.chunk_frames])
        video = normalize_frames(np.stack(video_chunks)).to(device)
        actions = normalize_notepad_actions(np.stack(action_chunks), self.max_delta, self.key_count).to(device)
        positions = normalize_positions(np.stack(position_chunks), self.width, self.height).to(device)
        return patchify(video), actions, positions, torch.as_tensor(chunk_idx, dtype=torch.long, device=device)

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video = normalize_frames(self.frames.reshape(-1, *self.frames.shape[2:])).reshape(
            -1, self.chunk_frames, *self.frames.shape[2:]
        )
        actions = normalize_notepad_actions(
            self.actions.reshape(-1, self.actions.shape[-1]), self.max_delta, self.key_count
        ).reshape(-1, self.chunk_frames, self.actions.shape[-1])
        positions = normalize_positions(self.positions.reshape(-1, self.chunk_frames, 2), self.width, self.height)
        chunk_ids = torch.arange(self.chunks_per_episode).repeat(self.frames.shape[0])
        return patchify(video).to(device), actions.to(device), positions.to(device), chunk_ids.to(device)

    def _motion_pairs(self) -> np.ndarray:
        pairs = []
        for ep in range(self.actions.shape[0]):
            for chunk in range(self.chunks_per_episode):
                start = chunk * self.chunk_frames
                deltas = self.actions[ep, start : start + self.chunk_frames, 0:2]
                if np.abs(deltas).max() > 0.5:
                    pairs.append((ep, chunk))
        return np.asarray(pairs, dtype=np.int64)


class ContactSheetDatasetAdapter:
    def __init__(self, dataset: NotePadHybridChunks):
        self.dataset = dataset

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video, actions, _, chunk_ids = self.dataset.all_chunks(device)
        return video, actions, chunk_ids


def make_hybrid_eval_dataset(seed: int) -> tuple[NotePadHybridChunks, dict[str, float]]:
    from wammo.data.notepad import generate_episode, rare_event_rate

    frames, actions = generate_episode(seed)
    return NotePadHybridChunks(frames[None], actions[None], motion_oversample=False), {
        "eval_seed": seed,
        "eval_rare_event_rate": rare_event_rate(actions),
    }


def hybrid_training_step(
    model: NotePadHybridModel,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    position_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
    action_weight: float = 1.0,
    action_dropout: float = 0.0,
    delta_weight: float = 4.0,
    delta_ce_weight: float = 1.0,
    cursor_weight: float = 1.0,
    sigma_corner_weight: float = 0.0,
    sigma_corner_low: float = 0.0,
    sigma_corner_high: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean.shape[0]
    delta_clean = action_clean[..., 0:2]
    delta_targets = delta_to_bins(delta_clean)
    button_target = action_clean[..., 2].long()
    key_target = action_clean[..., 3].long()
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    delta_noise = torch.randn(delta_clean.shape, device=action_clean.device, generator=generator)
    t_video, t_action, sigma_metrics = sample_sigma_pair(
        b,
        video_clean.device,
        generator,
        sigma_corner_weight,
        sigma_corner_low,
        sigma_corner_high,
    )
    video_noisy = interpolate(video_clean, video_noise, t_video)
    delta_noisy = interpolate(delta_clean, delta_noise, t_action)
    action_drop = None
    if action_dropout > 0:
        action_drop = torch.rand((b, 1, 1), device=video_clean.device, generator=generator) < action_dropout
    video_target = velocity_target(video_clean, video_noise)
    delta_target = velocity_target(delta_clean, delta_noise)
    video_pred, delta_pred, dx_logits, dy_logits, button_logits, key_logits, cursor_pred = model.forward_all(
        video_noisy, delta_noisy, button_target, key_target, t_video, t_action, chunk_ids, action_drop
    )
    video_loss = F.mse_loss(video_pred, video_target)
    delta_loss = F.mse_loss(delta_pred, delta_target)
    dx_loss = F.cross_entropy(dx_logits.reshape(-1, DELTA_BINS), delta_targets[..., 0].reshape(-1))
    dy_loss = F.cross_entropy(dy_logits.reshape(-1, DELTA_BINS), delta_targets[..., 1].reshape(-1))
    delta_ce_loss = 0.5 * (dx_loss + dy_loss)
    button_loss = F.cross_entropy(button_logits.reshape(-1, 2), button_target.reshape(-1))
    key_loss = F.cross_entropy(key_logits.reshape(-1, model.key_count), key_target.reshape(-1))
    cursor_loss = F.mse_loss(cursor_pred, position_clean)
    action_loss = delta_weight * delta_loss + delta_ce_weight * delta_ce_loss + button_loss + key_loss
    loss = video_loss + action_weight * action_loss + cursor_weight * cursor_loss
    return loss, {
        "loss": float(loss.detach()),
        "video_loss": float(video_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "weighted_delta_loss": float((delta_weight * delta_loss).detach()),
        "delta_ce_loss": float(delta_ce_loss.detach()),
        "button_loss": float(button_loss.detach()),
        "key_loss": float(key_loss.detach()),
        "cursor_loss": float(cursor_loss.detach()),
        **sigma_metrics,
    }


@torch.no_grad()
def denoise_once_hybrid(
    model: NotePadHybridModel,
    video_noise: torch.Tensor,
    delta_noise: torch.Tensor,
    button_ids: torch.Tensor,
    key_ids: torch.Tensor,
    chunk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = video_noise.shape[0]
    t = torch.ones((b,), device=video_noise.device)
    out = model.forward_all(video_noise, delta_noise, button_ids, key_ids, t, t, chunk_ids)
    video_v, delta_v, *_ = out
    return euler_step_toward_data(video_noise, video_v, dt=1.0), euler_step_toward_data(delta_noise, delta_v, dt=1.0), *out[2:]


@torch.no_grad()
def evaluate_hybrid(
    model: NotePadHybridModel,
    dataset: NotePadHybridChunks,
    device: torch.device,
    action_weight: float,
    action_dropout: float,
    delta_weight: float,
    delta_ce_weight: float,
    cursor_weight: float,
    sigma_corner_weight: float = 0.0,
    sigma_corner_low: float = 0.0,
    sigma_corner_high: float = 1.0,
    seed: int = 999,
) -> dict[str, float]:
    spec = load_spec()
    key_count = len(spec["keys"])
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    video_clean, action_clean, position_clean, chunk_ids = dataset.all_chunks(device)
    loss, metrics = hybrid_training_step(
        model,
        video_clean,
        action_clean,
        position_clean,
        chunk_ids,
        generator,
        action_weight,
        action_dropout=0.0,
        delta_weight=delta_weight,
        delta_ce_weight=delta_ce_weight,
        cursor_weight=cursor_weight,
        sigma_corner_weight=sigma_corner_weight,
        sigma_corner_low=sigma_corner_low,
        sigma_corner_high=sigma_corner_high,
    )
    del loss
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    delta_noise = torch.randn(action_clean[..., 0:2].shape, device=device, generator=generator)
    button_true = action_clean[..., 2].long()
    key_true = action_clean[..., 3].long()
    (
        video_denoised,
        delta_denoised,
        dx_logits,
        dy_logits,
        button_logits,
        key_logits,
        cursor_pred,
    ) = denoise_once_hybrid(model, video_noise, delta_noise, button_true, key_true, chunk_ids)
    raw_true = denormalize_notepad_actions(action_clean, float(spec["cursor"]["max_delta"]), key_count)
    raw_flow = raw_true.clone()
    raw_flow[..., 0:2] = delta_denoised.clamp(-1, 1) * float(spec["cursor"]["max_delta"])
    raw_ce = raw_true.clone()
    raw_ce[..., 0] = bins_to_delta_norm(dx_logits.argmax(dim=-1)) * float(spec["cursor"]["max_delta"])
    raw_ce[..., 1] = bins_to_delta_norm(dy_logits.argmax(dim=-1)) * float(spec["cursor"]["max_delta"])
    raw_ce[..., 2] = button_logits.argmax(dim=-1)
    raw_ce[..., 3] = key_logits.argmax(dim=-1)
    click_true = raw_true[..., 2] >= 0.5
    click_pred = raw_ce[..., 2] >= 0.5
    key_true_raw = raw_true[..., 3].long()
    key_pred = raw_ce[..., 3].long().clamp(0, key_count - 1)
    key_event = key_true_raw != 0
    cursor_pred_px = denormalize_positions(cursor_pred, int(spec["canvas"]["width"]), int(spec["canvas"]["height"]))
    cursor_true_px = denormalize_positions(position_clean, int(spec["canvas"]["width"]), int(spec["canvas"]["height"]))
    metrics.update(
        {
            "video_mae": float((video_denoised - video_clean).abs().mean()),
            "delta_mae_px": float((raw_flow[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
            "delta_ce_mae_px": float((raw_ce[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
            "click_accuracy": float((click_pred == click_true).float().mean()),
            "key_accuracy": float((key_pred == key_true_raw).float().mean()),
            "key_event_accuracy": float((key_pred[key_event] == key_true_raw[key_event]).float().mean()) if key_event.any() else 1.0,
            "cursor_pos_mae_px": float((cursor_pred_px - cursor_true_px).abs().mean()),
            "action_mae": float((delta_denoised - action_clean[..., 0:2]).abs().mean()),
        }
    )
    model.train()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/notepad-1k-hybrid"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ladder-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-dropout", type=float, default=0.0)
    parser.add_argument("--delta-weight", type=float, default=4.0)
    parser.add_argument("--delta-ce-weight", type=float, default=1.0)
    parser.add_argument("--cursor-weight", type=float, default=1.0)
    parser.add_argument("--head-sigma-conditioned", action="store_true")
    parser.add_argument("--sigma-corner-weight", type=float, default=0.0)
    parser.add_argument("--sigma-corner-low", type=float, default=0.0)
    parser.add_argument("--sigma-corner-high", type=float, default=1.0)
    parser.add_argument("--generate-progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu to run on CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)

    train_frames, train_actions, train_metadata = generate_training_dataset(
        args.episodes, args.seed, progress_every=args.generate_progress_every
    )
    train_dataset = NotePadHybridChunks(train_frames, train_actions, motion_oversample=True)
    eval_frames, eval_actions, eval_metadata = generate_training_dataset(args.eval_episodes, args.eval_seed, progress_every=0)
    eval_dataset = NotePadHybridChunks(eval_frames, eval_actions, motion_oversample=False)
    if args.eval_episodes == 1:
        _, legacy_eval_metadata = make_eval_dataset(args.eval_seed)
        eval_metadata.update(legacy_eval_metadata)

    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        action_dim=4,
        patches_per_frame=24 * 24,
        max_chunks=16,
    )
    model = NotePadHybridModel(
        config,
        key_count=len(load_spec()["keys"]),
        head_sigma_conditioned=args.head_sigma_conditioned,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": "notepad_hybrid",
        "args": vars(args) | {"out": str(args.out), "motion_oversample": True},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
    }
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    first_eval: dict[str, float] | None = None
    start = time.time()
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, positions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = hybrid_training_step(
                model,
                video,
                actions,
                positions,
                chunk_ids,
                noise_generator,
                args.action_weight,
                args.action_dropout,
                args.delta_weight,
                args.delta_ce_weight,
                args.cursor_weight,
                args.sigma_corner_weight,
                args.sigma_corner_low,
                args.sigma_corner_high,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate_hybrid(
                    model,
                    eval_dataset,
                    device,
                    args.action_weight,
                    args.action_dropout,
                    args.delta_weight,
                    args.delta_ce_weight,
                    args.cursor_weight,
                    args.sigma_corner_weight,
                    args.sigma_corner_low,
                    args.sigma_corner_high,
                )
                if args.ladder_every > 0 and (step == 1 or step % args.ladder_every == 0 or step == args.steps):
                    video_all, action_all, _, chunk_ids_all = eval_dataset.all_chunks(device)
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
                    f"flow_delta={row['eval_delta_mae_px']:.3f}px ce_delta={row['eval_delta_ce_mae_px']:.3f}px "
                    f"cursor={row['eval_cursor_pos_mae_px']:.3f}px click={row['eval_click_accuracy']:.3f} "
                    f"key={row['eval_key_accuracy']:.3f}"
                )
                if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": asdict(config),
                            "model_kind": "notepad_hybrid",
                            "head_sigma_conditioned": args.head_sigma_conditioned,
                            "step": step,
                        },
                        args.out / f"checkpoint_step_{step}.pt",
                    )

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "model_kind": "notepad_hybrid",
            "head_sigma_conditioned": args.head_sigma_conditioned,
        },
        args.out / "checkpoint.pt",
    )
    write_contact_sheet(model, ContactSheetDatasetAdapter(eval_dataset), args.out / "contact_sheet.png", device)
    final_eval = evaluate_hybrid(
        model,
        eval_dataset,
        device,
        args.action_weight,
        args.action_dropout,
        args.delta_weight,
        args.delta_ce_weight,
        args.cursor_weight,
        args.sigma_corner_weight,
        args.sigma_corner_low,
        args.sigma_corner_high,
    )
    video_all, action_all, _, chunk_ids_all = eval_dataset.all_chunks(device)
    final_eval.update(
        notepad_divergence_ladder(model, video_all, action_all, chunk_ids_all, key_index=load_spec()["keys"].index("h"))
    )
    (args.out / "summary.json").write_text(json.dumps({"first_eval": first_eval, "final_eval": final_eval}, indent=2) + "\n")


if __name__ == "__main__":
    main()
