from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import interpolate
from wammo.notepad_desk import load_spec
from wammo.train.train_notepad import generate_training_dataset
from wammo.train.train_notepad_binned_delta import bins_to_delta_norm
from wammo.train.train_notepad_hybrid import (
    NotePadHybridChunks,
    NotePadHybridModel,
    denormalize_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--batch-chunks", type=int, default=64)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_model(
    run: Path, checkpoint_path: Path | None, device: torch.device
) -> tuple[NotePadHybridModel, dict[str, object], dict[str, object]]:
    config_payload = json.loads((run / "config.json").read_text())
    resolved_checkpoint = checkpoint_path or (run / "checkpoint.pt")
    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    config = MicroWAMConfig(**checkpoint["config"])
    run_args = config_payload.get("args", {})
    head_sigma_conditioned = bool(checkpoint.get("head_sigma_conditioned", run_args.get("head_sigma_conditioned", False)))
    model = NotePadHybridModel(
        config,
        key_count=len(load_spec()["keys"]),
        head_sigma_conditioned=head_sigma_conditioned,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, config_payload, {
        "config": asdict(config),
        "head_sigma_conditioned": head_sigma_conditioned,
        "checkpoint": str(resolved_checkpoint),
        "step": checkpoint.get("step"),
    }


def make_dataset(config_payload: dict[str, object], args: argparse.Namespace) -> NotePadHybridChunks:
    run_args = config_payload.get("args", {})
    if not isinstance(run_args, dict):
        raise ValueError("config args must be a dict")
    seed = int(args.eval_seed if args.eval_seed is not None else run_args["eval_seed"])
    frames, actions, _ = generate_training_dataset(args.eval_episodes, seed, progress_every=0)
    return NotePadHybridChunks(frames, actions, motion_oversample=False)


@torch.no_grad()
def evaluate_at_sigmas(
    model: NotePadHybridModel,
    dataset: NotePadHybridChunks,
    sigmas: list[float],
    device: torch.device,
    batch_chunks: int,
    noise_seed: int,
) -> dict[str, list[dict[str, float]]]:
    model.eval()
    context_chunks = int(getattr(model, "_wammo_context_chunks", 0))
    if context_chunks > 0:
        (
            video_all,
            action_all,
            position_all,
            chunk_id_all,
            context_video_all,
            context_action_all,
            context_id_all,
        ) = dataset.all_chunks_with_context(torch.device("cpu"), context_chunks)
    else:
        video_all, action_all, position_all, chunk_id_all = dataset.all_chunks(torch.device("cpu"))
        context_video_all = context_action_all = context_id_all = None
    output: dict[str, list[dict[str, float]]] = {
        "cursor_by_video_sigma_clean_action": [],
        "delta_by_action_sigma_clean_video": [],
        "delta_by_video_sigma_noisy_action": [],
        "delta_by_equal_sigma": [],
    }
    for sigma in sigmas:
        accum = {
            "cursor_video": {"abs": 0.0, "euclidean": 0.0, "frames": 0.0},
            "delta_action": _empty_delta(),
            "delta_video": _empty_delta(),
            "delta_equal": _empty_delta(),
        }
        for start in range(0, video_all.shape[0], batch_chunks):
            end = start + batch_chunks
            video = video_all[start:end].to(device)
            actions = action_all[start:end].to(device)
            positions = position_all[start:end].to(device)
            chunk_ids = chunk_id_all[start:end].to(device)
            context_video = context_action = context_ids = None
            if context_video_all is not None and context_action_all is not None and context_id_all is not None:
                context_video = context_video_all[start:end].to(device)
                context_action = context_action_all[start:end].to(device)
                context_ids = context_id_all[start:end].to(device)
            video_noise = _noise(video.shape, device, noise_seed + start + 17)
            delta_noise = _noise(actions[..., 0:2].shape, device, noise_seed + start + 31)
            button = actions[..., 2].long()
            key = actions[..., 3].long()
            sigma_t = torch.full((video.shape[0],), sigma, device=device)

            _accumulate_cursor(
                accum["cursor_video"],
                model,
                interpolate(video, video_noise, sigma_t),
                actions[..., 0:2],
                button,
                key,
                positions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=0.0,
                dataset=dataset,
                context_video=context_video,
                context_actions=context_action,
                context_chunk_ids=context_ids,
            )
            _accumulate_delta(
                accum["delta_action"],
                model,
                video,
                interpolate(actions[..., 0:2], delta_noise, sigma_t),
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=0.0,
                sigma_action=sigma,
                max_delta=dataset.max_delta,
                context_video=context_video,
                context_actions=context_action,
                context_chunk_ids=context_ids,
            )
            _accumulate_delta(
                accum["delta_video"],
                model,
                interpolate(video, video_noise, sigma_t),
                delta_noise,
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=1.0,
                max_delta=dataset.max_delta,
                context_video=context_video,
                context_actions=context_action,
                context_chunk_ids=context_ids,
            )
            _accumulate_delta(
                accum["delta_equal"],
                model,
                interpolate(video, video_noise, sigma_t),
                interpolate(actions[..., 0:2], delta_noise, sigma_t),
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=sigma,
                max_delta=dataset.max_delta,
                context_video=context_video,
                context_actions=context_action,
                context_chunk_ids=context_ids,
            )
        output["cursor_by_video_sigma_clean_action"].append(_finish_cursor(sigma, accum["cursor_video"]))
        output["delta_by_action_sigma_clean_video"].append(_finish_delta(sigma, accum["delta_action"]))
        output["delta_by_video_sigma_noisy_action"].append(_finish_delta(sigma, accum["delta_video"]))
        output["delta_by_equal_sigma"].append(_finish_delta(sigma, accum["delta_equal"]))
    return output


def _noise(shape: torch.Size, device: torch.device, seed: int) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))


def _empty_delta() -> dict[str, float]:
    return {
        "flow_abs": 0.0,
        "ce_abs": 0.0,
        "flow_pred_abs": 0.0,
        "ce_pred_abs": 0.0,
        "motion_count": 0.0,
        "frame_count": 0.0,
    }


def _accumulate_cursor(
    acc: dict[str, float],
    model: NotePadHybridModel,
    video_input: torch.Tensor,
    delta_input: torch.Tensor,
    button: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    chunk_ids: torch.Tensor,
    sigma_video: float,
    sigma_action: float,
    dataset: NotePadHybridChunks,
    context_video: torch.Tensor | None = None,
    context_actions: torch.Tensor | None = None,
    context_chunk_ids: torch.Tensor | None = None,
) -> None:
    b = video_input.shape[0]
    *_, cursor_pred = model.forward_all(
        video_input,
        delta_input,
        button,
        key,
        torch.full((b,), sigma_video, device=video_input.device),
        torch.full((b,), sigma_action, device=video_input.device),
        chunk_ids,
        context_video_patches=context_video,
        context_actions=context_actions,
        context_chunk_ids=context_chunk_ids,
    )
    pred_px = denormalize_positions(cursor_pred, dataset.width, dataset.height)
    true_px = denormalize_positions(positions, dataset.width, dataset.height)
    diff = (pred_px - true_px).abs()
    acc["abs"] += float(diff.sum())
    acc["euclidean"] += float(torch.linalg.vector_norm(pred_px - true_px, dim=-1).sum())
    acc["frames"] += float(cursor_pred.numel() // 2)


def _accumulate_delta(
    acc: dict[str, float],
    model: NotePadHybridModel,
    video_input: torch.Tensor,
    delta_input: torch.Tensor,
    delta_noise: torch.Tensor,
    button: torch.Tensor,
    key: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    sigma_video: float,
    sigma_action: float,
    max_delta: float,
    context_video: torch.Tensor | None = None,
    context_actions: torch.Tensor | None = None,
    context_chunk_ids: torch.Tensor | None = None,
) -> None:
    b = video_input.shape[0]
    _, delta_velocity, dx_logits, dy_logits, *_ = model.forward_all(
        video_input,
        delta_input,
        button,
        key,
        torch.full((b,), sigma_video, device=video_input.device),
        torch.full((b,), sigma_action, device=video_input.device),
        chunk_ids,
        context_video_patches=context_video,
        context_actions=context_actions,
        context_chunk_ids=context_chunk_ids,
    )
    flow_x0 = (delta_noise - delta_velocity).clamp(-1, 1) * max_delta
    ce_x0 = torch.stack(
        [bins_to_delta_norm(dx_logits.argmax(dim=-1)), bins_to_delta_norm(dy_logits.argmax(dim=-1))],
        dim=-1,
    ) * max_delta
    true_delta = actions[..., 0:2] * max_delta
    motion = true_delta.abs().amax(dim=-1) > 0.5
    acc["frame_count"] += float(motion.numel())
    if not motion.any():
        return
    true_motion = true_delta[motion]
    flow_motion = flow_x0[motion]
    ce_motion = ce_x0[motion]
    acc["motion_count"] += float(motion.sum())
    acc["flow_abs"] += float((flow_motion - true_motion).abs().sum())
    acc["ce_abs"] += float((ce_motion - true_motion).abs().sum())
    acc["flow_pred_abs"] += float(flow_motion.abs().sum())
    acc["ce_pred_abs"] += float(ce_motion.abs().sum())


def _finish_cursor(sigma: float, acc: dict[str, float]) -> dict[str, float]:
    return {
        "sigma": sigma,
        "cursor_mae_px": acc["abs"] / (acc["frames"] * 2),
        "cursor_euclidean_px": acc["euclidean"] / acc["frames"],
    }


def _finish_delta(sigma: float, acc: dict[str, float]) -> dict[str, float]:
    denom = max(acc["motion_count"] * 2, 1.0)
    return {
        "sigma": sigma,
        "motion_frames": int(acc["motion_count"]),
        "motion_rate": acc["motion_count"] / max(acc["frame_count"], 1.0),
        "flow_motion_delta_mae_px": acc["flow_abs"] / denom,
        "ce_motion_delta_mae_px": acc["ce_abs"] / denom,
        "flow_motion_pred_abs_mean_px": acc["flow_pred_abs"] / denom,
        "ce_motion_pred_abs_mean_px": acc["ce_pred_abs"] / denom,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, config_payload, model_meta = load_model(args.run, args.checkpoint, device)
    run_args = config_payload.get("args", {})
    if isinstance(run_args, dict):
        setattr(model, "_wammo_context_chunks", int(run_args.get("context_chunks", 0)))
    dataset = make_dataset(config_payload, args)
    stratified = evaluate_at_sigmas(model, dataset, args.sigmas, device, args.batch_chunks, args.noise_seed)
    output = {
        "run": str(args.run),
        "model": model_meta,
        "eval_episodes": args.eval_episodes,
        "eval_seed": int(args.eval_seed if args.eval_seed is not None else config_payload["args"]["eval_seed"]),
        "sigmas": args.sigmas,
        "stratified": stratified,
    }
    out = args.out or (args.run / "analysis" / "sigma_stratification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
