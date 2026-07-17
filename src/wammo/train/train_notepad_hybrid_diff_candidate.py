from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from wammo.eval.divergence_ladder import action_variants, channel_divergence, changed_patch_mask
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import euler_step_toward_data, interpolate, velocity_target
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import denormalize_notepad_actions, normalize_notepad_actions
from wammo.train.overfit_one import normalize_frames
from wammo.train.train_notepad import generate_training_dataset, make_eval_dataset
from wammo.train.train_notepad_binned_delta import DELTA_BINS, bins_to_delta_norm, delta_to_bins
from wammo.train.train_notepad_hybrid import (
    NotePadHybridChunks,
    NotePadHybridModel,
    denormalize_positions,
    hybrid_video_patches,
    normalize_positions,
    sample_sigma_pair,
)
from wammo.train.train_notepad_pure_inverse import patch_dim_for_mode, visible_mask


class NotePadHybridFrameChunks(NotePadHybridChunks):
    def __init__(self, frames: np.ndarray, actions: np.ndarray, input_mode: str, chunk_frames: int = 4, motion_oversample: bool = True):
        super().__init__(frames, actions, chunk_frames=chunk_frames, motion_oversample=motion_oversample)
        self.input_mode = input_mode

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
        return video, actions, positions, torch.as_tensor(chunk_idx, dtype=torch.long, device=device)

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video = normalize_frames(self.frames.reshape(-1, *self.frames.shape[2:])).reshape(
            -1, self.chunk_frames, *self.frames.shape[2:]
        )
        actions = normalize_notepad_actions(
            self.actions.reshape(-1, self.actions.shape[-1]), self.max_delta, self.key_count
        ).reshape(-1, self.chunk_frames, self.actions.shape[-1])
        positions = normalize_positions(self.positions.reshape(-1, self.chunk_frames, 2), self.width, self.height)
        chunk_ids = torch.arange(self.chunks_per_episode).repeat(self.frames.shape[0])
        return video.to(device), actions.to(device), positions.to(device), chunk_ids.to(device)


def diff_hybrid_training_step(
    model: NotePadHybridModel,
    video_clean_rgb: torch.Tensor,
    action_clean: torch.Tensor,
    position_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
    input_mode: str,
    action_weight: float,
    action_dropout: float,
    delta_weight: float,
    delta_ce_weight: float,
    cursor_weight: float,
    inverse_aux_weight: float,
    sigma_corner_weight: float,
    sigma_corner_low: float,
    sigma_corner_high: float,
    mask_first_frame: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean_rgb.shape[0]
    delta_clean = action_clean[..., 0:2]
    delta_targets = delta_to_bins(delta_clean)
    button_target = action_clean[..., 2].long()
    key_target = action_clean[..., 3].long()
    video_noise_rgb = torch.randn(video_clean_rgb.shape, device=video_clean_rgb.device, generator=generator)
    delta_noise = torch.randn(delta_clean.shape, device=action_clean.device, generator=generator)
    t_video, t_action, sigma_metrics = sample_sigma_pair(
        b, video_clean_rgb.device, generator, sigma_corner_weight, sigma_corner_low, sigma_corner_high
    )
    video_noisy_rgb = interpolate(video_clean_rgb, video_noise_rgb, t_video)
    delta_noisy = interpolate(delta_clean, delta_noise, t_action)
    action_drop = None
    if action_dropout > 0:
        action_drop = torch.rand((b, 1, 1), device=video_clean_rgb.device, generator=generator) < action_dropout
    video_input = hybrid_video_patches(video_noisy_rgb, input_mode)
    video_target = velocity_target(patchify(video_clean_rgb), patchify(video_noise_rgb))
    delta_target = velocity_target(delta_clean, delta_noise)
    video_pred, delta_pred, dx_logits, dy_logits, button_logits, key_logits, cursor_pred = model.forward_all(
        video_input, delta_noisy, button_target, key_target, t_video, t_action, chunk_ids, action_drop
    )
    video_loss = F.mse_loss(video_pred, video_target)
    delta_loss = F.mse_loss(delta_pred, delta_target)
    dx_loss = F.cross_entropy(dx_logits.reshape(-1, DELTA_BINS), delta_targets[..., 0].reshape(-1))
    dy_loss = F.cross_entropy(dy_logits.reshape(-1, DELTA_BINS), delta_targets[..., 1].reshape(-1))
    delta_ce_loss = 0.5 * (dx_loss + dy_loss)
    button_loss = F.cross_entropy(button_logits.reshape(-1, 2), button_target.reshape(-1))
    key_loss = F.cross_entropy(key_logits.reshape(-1, model.key_count), key_target.reshape(-1))
    cursor_loss = F.mse_loss(cursor_pred, position_clean)

    inverse_input = hybrid_video_patches(video_clean_rgb, input_mode)
    _, _, inv_dx, inv_dy, _, _, _ = model.forward_all(
        inverse_input,
        delta_noise,
        button_target,
        key_target,
        torch.zeros((b,), device=video_clean_rgb.device),
        torch.ones((b,), device=video_clean_rgb.device),
        chunk_ids,
    )
    inverse_mask = visible_mask(action_clean, mask_first_frame)
    inverse_dx_loss = F.cross_entropy(inv_dx[inverse_mask], delta_targets[..., 0][inverse_mask])
    inverse_dy_loss = F.cross_entropy(inv_dy[inverse_mask], delta_targets[..., 1][inverse_mask])
    inverse_aux_loss = 0.5 * (inverse_dx_loss + inverse_dy_loss)

    action_loss = delta_weight * delta_loss + delta_ce_weight * delta_ce_loss + button_loss + key_loss
    loss = video_loss + action_weight * action_loss + cursor_weight * cursor_loss + inverse_aux_weight * inverse_aux_loss
    inverse_metrics = delta_logits_metrics(inv_dx, inv_dy, action_clean, model_max_delta(), inverse_mask)
    return loss, {
        "loss": float(loss.detach()),
        "video_loss": float(video_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "delta_ce_loss": float(delta_ce_loss.detach()),
        "button_loss": float(button_loss.detach()),
        "key_loss": float(key_loss.detach()),
        "cursor_loss": float(cursor_loss.detach()),
        "inverse_aux_loss": float(inverse_aux_loss.detach()),
        **{f"inverse_{k}": v for k, v in inverse_metrics.items()},
        **sigma_metrics,
    }


def model_max_delta() -> float:
    return float(load_spec()["cursor"]["max_delta"])


def delta_logits_metrics(
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
    if motion.any():
        return {
            "motion_delta_mae_px": float((pred[motion] - true[motion]).abs().mean()),
            "motion_pred_abs_mean_px": float(pred[motion].abs().mean()),
            "motion_zero_delta_mae_px": float(true[motion].abs().mean()),
            "motion_frames": int(motion.sum()),
        }
    return {
        "motion_delta_mae_px": 0.0,
        "motion_pred_abs_mean_px": 0.0,
        "motion_zero_delta_mae_px": 0.0,
        "motion_frames": 0,
    }


@torch.no_grad()
def evaluate_diff_hybrid(
    model: NotePadHybridModel,
    dataset: NotePadHybridFrameChunks,
    device: torch.device,
    input_mode: str,
    action_weight: float,
    action_dropout: float,
    delta_weight: float,
    delta_ce_weight: float,
    cursor_weight: float,
    inverse_aux_weight: float,
    sigma_corner_weight: float,
    sigma_corner_low: float,
    sigma_corner_high: float,
    seed: int = 999,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
        video_clean, action_clean, position_clean, chunk_ids = dataset.all_chunks(device)
        loss, metrics = diff_hybrid_training_step(
            model,
            video_clean,
            action_clean,
            position_clean,
            chunk_ids,
            generator,
            input_mode,
            action_weight,
            action_dropout=0.0,
            delta_weight=delta_weight,
            delta_ce_weight=delta_ce_weight,
            cursor_weight=cursor_weight,
            inverse_aux_weight=inverse_aux_weight,
            sigma_corner_weight=sigma_corner_weight,
            sigma_corner_low=sigma_corner_low,
            sigma_corner_high=sigma_corner_high,
        )
        del loss
        spec = load_spec()
        key_count = len(spec["keys"])
        video_noise_rgb = torch.randn(video_clean.shape, device=device, generator=generator)
        delta_noise = torch.randn(action_clean[..., 0:2].shape, device=device, generator=generator)
        button_true = action_clean[..., 2].long()
        key_true = action_clean[..., 3].long()
        video_input = hybrid_video_patches(video_noise_rgb, input_mode)
        t = torch.ones((video_clean.shape[0],), device=device)
        out = model.forward_all(video_input, delta_noise, button_true, key_true, t, t, chunk_ids)
        video_v, delta_v, dx_logits, dy_logits, button_logits, key_logits, cursor_pred = out
        video_denoised = euler_step_toward_data(patchify(video_noise_rgb), video_v, dt=1.0)
        delta_denoised = euler_step_toward_data(delta_noise, delta_v, dt=1.0)
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
        inverse_input = hybrid_video_patches(video_clean, input_mode)
        _, _, inv_dx, inv_dy, _, _, _ = model.forward_all(
            inverse_input,
            delta_noise,
            button_true,
            key_true,
            torch.zeros((video_clean.shape[0],), device=device),
            torch.ones((video_clean.shape[0],), device=device),
            chunk_ids,
        )
        inverse_metrics = delta_logits_metrics(inv_dx, inv_dy, action_clean, float(spec["cursor"]["max_delta"]), visible_mask(action_clean, True))
        metrics.update(
            {
                "video_mae": float((video_denoised - patchify(video_clean)).abs().mean()),
                "delta_mae_px": float((raw_flow[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
                "delta_ce_mae_px": float((raw_ce[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
                "click_accuracy": float((click_pred == click_true).float().mean()),
                "key_accuracy": float((key_pred == key_true_raw).float().mean()),
                "key_event_accuracy": float((key_pred[key_event] == key_true_raw[key_event]).float().mean()) if key_event.any() else 1.0,
                "cursor_pos_mae_px": float((cursor_pred_px - cursor_true_px).abs().mean()),
                "action_mae": float((delta_denoised - action_clean[..., 0:2]).abs().mean()),
                **{f"eval_inverse_{k}": v for k, v in inverse_metrics.items()},
            }
        )
        return metrics
    finally:
        model.train(was_training)


@torch.no_grad()
def diff_hybrid_ladder(
    model: NotePadHybridModel,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    input_mode: str,
    key_index: int,
    seed: int = 2024,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device=video_clean.device).manual_seed(seed)
        video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
        changed_mask = changed_patch_mask(patchify(video_clean))
        results: dict[str, float] = {}
        for channel in ("cursor", "click", "key"):
            positive, negative = action_variants(action_clean, channel, key_index)
            pos_video = denoise_diff_video_with_actions(model, video_noise, positive, chunk_ids, input_mode)
            neg_video = denoise_diff_video_with_actions(model, video_noise, negative, chunk_ids, input_mode)
            results.update(channel_divergence(channel, pos_video, neg_video, changed_mask, (1, 2, 3, 4)))
        return results
    finally:
        model.train(was_training)


def denoise_diff_video_with_actions(
    model: NotePadHybridModel,
    video_noise_rgb: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    input_mode: str,
) -> torch.Tensor:
    b = video_noise_rgb.shape[0]
    sigma_video = torch.ones((b,), device=video_noise_rgb.device)
    sigma_action = torch.zeros((b,), device=video_noise_rgb.device)
    video_velocity, *_ = model.forward_all(
        hybrid_video_patches(video_noise_rgb, input_mode),
        actions[..., 0:2],
        actions[..., 2].long(),
        actions[..., 3].long(),
        sigma_video,
        sigma_action,
        chunk_ids,
    )
    return euler_step_toward_data(patchify(video_noise_rgb), video_velocity, dt=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["coord", "coord-diff"], default="coord-diff")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--ladder-every", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-dropout", type=float, default=0.0)
    parser.add_argument("--delta-weight", type=float, default=4.0)
    parser.add_argument("--delta-ce-weight", type=float, default=1.0)
    parser.add_argument("--cursor-weight", type=float, default=1.0)
    parser.add_argument("--inverse-aux-weight", type=float, default=1.0)
    parser.add_argument("--sigma-corner-weight", type=float, default=0.5)
    parser.add_argument("--sigma-corner-low", type=float, default=0.0)
    parser.add_argument("--sigma-corner-high", type=float, default=1.0)
    parser.add_argument("--generate-progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    train_frames, train_actions, train_metadata = generate_training_dataset(
        args.episodes, args.seed, progress_every=args.generate_progress_every
    )
    train_dataset = NotePadHybridFrameChunks(train_frames, train_actions, input_mode=args.input_mode, motion_oversample=True)
    eval_frames, eval_actions, eval_metadata = generate_training_dataset(64, args.eval_seed, progress_every=0)
    eval_dataset = NotePadHybridFrameChunks(eval_frames, eval_actions, input_mode=args.input_mode, motion_oversample=False)
    _, legacy_eval_metadata = make_eval_dataset(args.eval_seed)
    eval_metadata.update(legacy_eval_metadata)
    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        patch_dim=patch_dim_for_mode(args.input_mode, 4),
        action_dim=4,
        patches_per_frame=24 * 24,
        max_chunks=16,
    )
    model = NotePadHybridModel(
        config,
        key_count=len(load_spec()["keys"]),
        head_sigma_conditioned=True,
        video_output_dim=4 * 4 * 3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": "notepad_hybrid_diff_candidate",
        "args": vars(args) | {"out": str(args.out), "motion_oversample": True, "head_sigma_conditioned": True},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
    }
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    first_eval = None
    start = time.time()
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, positions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = diff_hybrid_training_step(
                model,
                video,
                actions,
                positions,
                chunk_ids,
                noise_generator,
                args.input_mode,
                args.action_weight,
                args.action_dropout,
                args.delta_weight,
                args.delta_ce_weight,
                args.cursor_weight,
                args.inverse_aux_weight,
                args.sigma_corner_weight,
                args.sigma_corner_low,
                args.sigma_corner_high,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate_diff_hybrid(
                    model,
                    eval_dataset,
                    device,
                    args.input_mode,
                    args.action_weight,
                    args.action_dropout,
                    args.delta_weight,
                    args.delta_ce_weight,
                    args.cursor_weight,
                    args.inverse_aux_weight,
                    args.sigma_corner_weight,
                    args.sigma_corner_low,
                    args.sigma_corner_high,
                )
                if args.ladder_every > 0 and (step == 1 or step % args.ladder_every == 0 or step == args.steps):
                    video_all, action_all, _, chunk_ids_all = eval_dataset.all_chunks(device)
                    eval_metrics.update(
                        diff_hybrid_ladder(
                            model,
                            video_all,
                            action_all,
                            chunk_ids_all,
                            args.input_mode,
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
                    f"inverse={row['eval_eval_inverse_motion_delta_mae_px']:.3f}px "
                    f"cursor={row['eval_cursor_pos_mae_px']:.3f}px click={row['eval_click_accuracy']:.3f} "
                    f"key={row['eval_key_accuracy']:.3f}"
                )
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "model_kind": "notepad_hybrid_diff_candidate",
            "input_mode": args.input_mode,
            "head_sigma_conditioned": True,
            "video_output_dim": 4 * 4 * 3,
        },
        args.out / "checkpoint.pt",
    )
    final_eval = evaluate_diff_hybrid(
        model,
        eval_dataset,
        device,
        args.input_mode,
        args.action_weight,
        args.action_dropout,
        args.delta_weight,
        args.delta_ce_weight,
        args.cursor_weight,
        args.inverse_aux_weight,
        args.sigma_corner_weight,
        args.sigma_corner_low,
        args.sigma_corner_high,
    )
    video_all, action_all, _, chunk_ids_all = eval_dataset.all_chunks(device)
    final_eval.update(
        diff_hybrid_ladder(
            model,
            video_all,
            action_all,
            chunk_ids_all,
            args.input_mode,
            key_index=load_spec()["keys"].index("h"),
        )
    )
    (args.out / "summary.json").write_text(json.dumps({"first_eval": first_eval, "final_eval": final_eval}, indent=2) + "\n")


if __name__ == "__main__":
    main()
