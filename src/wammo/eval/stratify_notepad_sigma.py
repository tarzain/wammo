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
from wammo.train.train_notepad_representation_screen import (
    RepresentationScreenDataset,
    RepresentationScreenModel,
    cursor_patch_targets,
    decode_cursor_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--batch-chunks", type=int, default=64)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_screen_model(run: Path, device: torch.device) -> tuple[RepresentationScreenModel, dict[str, object], dict[str, object]]:
    summary = json.loads((run / "summary.json").read_text())
    checkpoint = torch.load(run / "checkpoint.pt", map_location=device)
    config = MicroWAMConfig(**checkpoint["config"])
    model = RepresentationScreenModel(
        config,
        key_count=len(load_spec()["keys"]),
        input_mode=checkpoint["input_mode"],
        patch_size=int(checkpoint.get("patch_size", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, summary, {"config": asdict(config), "input_mode": checkpoint["input_mode"], "patch_size": model.patch_size}


def make_dataset(summary: dict[str, object], args: argparse.Namespace) -> RepresentationScreenDataset:
    run_args = summary["args"]
    if not isinstance(run_args, dict):
        raise ValueError("summary args must be a dict")
    episodes = int(args.eval_episodes if args.eval_episodes is not None else run_args["eval_episodes"])
    seed = int(args.eval_seed if args.eval_seed is not None else run_args["eval_seed"])
    cursor_size = run_args.get("cursor_size")
    frames, actions, _ = generate_training_dataset(episodes, seed, progress_every=0, cursor_size=cursor_size)
    return RepresentationScreenDataset(frames, actions, motion_oversample=False)


def fixed_noise_like(shape: torch.Size, device: torch.device, seed: int) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))


@torch.no_grad()
def evaluate_at_sigmas(
    model: RepresentationScreenModel,
    dataset: RepresentationScreenDataset,
    sigmas: list[float],
    device: torch.device,
    batch_chunks: int,
    noise_seed: int,
) -> dict[str, list[dict[str, float]]]:
    video_all, action_all, position_all, chunk_id_all = dataset.all_chunks(torch.device("cpu"))
    max_delta = dataset.max_delta
    output: dict[str, list[dict[str, float]]] = {
        "cursor_by_video_sigma_clean_action": [],
        "cursor_by_equal_sigma": [],
        "delta_by_action_sigma_clean_video": [],
        "delta_by_video_sigma_noisy_action": [],
        "delta_by_equal_sigma": [],
    }
    for sigma in sigmas:
        accum = _empty_accumulators()
        for start in range(0, video_all.shape[0], batch_chunks):
            end = start + batch_chunks
            video = video_all[start:end].to(device)
            actions = action_all[start:end].to(device)
            positions = position_all[start:end].to(device)
            chunk_ids = chunk_id_all[start:end].to(device)
            video_noise = fixed_noise_like(video.shape, device, noise_seed + start + 17)
            delta_noise = fixed_noise_like(actions[..., 0:2].shape, device, noise_seed + start + 31)
            button = actions[..., 2].long()
            key = actions[..., 3].long()

            _accumulate_cursor(
                accum["cursor_video"],
                model,
                interpolate(video, video_noise, torch.full((video.shape[0],), sigma, device=device)),
                actions[..., 0:2],
                button,
                key,
                positions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=0.0,
                dataset=dataset,
            )
            delta_noisy_equal = interpolate(actions[..., 0:2], delta_noise, torch.full((video.shape[0],), sigma, device=device))
            _accumulate_cursor(
                accum["cursor_equal"],
                model,
                interpolate(video, video_noise, torch.full((video.shape[0],), sigma, device=device)),
                delta_noisy_equal,
                button,
                key,
                positions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=sigma,
                dataset=dataset,
            )
            _accumulate_delta(
                accum["delta_action"],
                model,
                video,
                interpolate(actions[..., 0:2], delta_noise, torch.full((video.shape[0],), sigma, device=device)),
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=0.0,
                sigma_action=sigma,
                max_delta=max_delta,
            )
            _accumulate_delta(
                accum["delta_video"],
                model,
                interpolate(video, video_noise, torch.full((video.shape[0],), sigma, device=device)),
                delta_noise,
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=1.0,
                max_delta=max_delta,
            )
            _accumulate_delta(
                accum["delta_equal"],
                model,
                interpolate(video, video_noise, torch.full((video.shape[0],), sigma, device=device)),
                delta_noisy_equal,
                delta_noise,
                button,
                key,
                actions,
                chunk_ids,
                sigma_video=sigma,
                sigma_action=sigma,
                max_delta=max_delta,
            )
        output["cursor_by_video_sigma_clean_action"].append(_finish_cursor(sigma, accum["cursor_video"]))
        output["cursor_by_equal_sigma"].append(_finish_cursor(sigma, accum["cursor_equal"]))
        output["delta_by_action_sigma_clean_video"].append(_finish_delta(sigma, accum["delta_action"]))
        output["delta_by_video_sigma_noisy_action"].append(_finish_delta(sigma, accum["delta_video"]))
        output["delta_by_equal_sigma"].append(_finish_delta(sigma, accum["delta_equal"]))
    return output


def _empty_accumulators() -> dict[str, dict[str, float]]:
    return {
        "cursor_video": {"abs": 0.0, "euclidean": 0.0, "frames": 0.0, "patch_correct": 0.0, "patch_total": 0.0},
        "cursor_equal": {"abs": 0.0, "euclidean": 0.0, "frames": 0.0, "patch_correct": 0.0, "patch_total": 0.0},
        "delta_action": _empty_delta(),
        "delta_video": _empty_delta(),
        "delta_equal": _empty_delta(),
    }


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
    model: RepresentationScreenModel,
    video_input: torch.Tensor,
    delta_input: torch.Tensor,
    button: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    chunk_ids: torch.Tensor,
    sigma_video: float,
    sigma_action: float,
    dataset: RepresentationScreenDataset,
) -> None:
    b = video_input.shape[0]
    _, _, _, _, _, _, _, patch_logits, offsets = model.forward_all(
        video_input,
        delta_input,
        button,
        key,
        torch.full((b,), sigma_video, device=video_input.device),
        torch.full((b,), sigma_action, device=video_input.device),
        chunk_ids,
    )
    decoded = decode_cursor_heatmap(patch_logits, offsets, model.patch_size, dataset.width, dataset.height)
    scale = torch.tensor([dataset.width - 1, dataset.height - 1], device=video_input.device, dtype=decoded.dtype)
    diff = (decoded - positions).abs() * scale
    acc["abs"] += float(diff.sum())
    acc["euclidean"] += float(torch.linalg.vector_norm((decoded - positions) * scale, dim=-1).sum())
    acc["frames"] += float(decoded.numel() // 2)
    target_patch, _ = cursor_patch_targets(positions, model.patch_size, dataset.width, dataset.height)
    acc["patch_correct"] += float((patch_logits.argmax(dim=-1) == target_patch).sum())
    acc["patch_total"] += float(target_patch.numel())


def _accumulate_delta(
    acc: dict[str, float],
    model: RepresentationScreenModel,
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
) -> None:
    b = video_input.shape[0]
    _, delta_velocity, dx_logits, dy_logits, _, _, _, _, _ = model.forward_all(
        video_input,
        delta_input,
        button,
        key,
        torch.full((b,), sigma_video, device=video_input.device),
        torch.full((b,), sigma_action, device=video_input.device),
        chunk_ids,
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
        "decoded_mae_px": acc["abs"] / (acc["frames"] * 2),
        "decoded_euclidean_px": acc["euclidean"] / acc["frames"],
        "patch_accuracy": acc["patch_correct"] / acc["patch_total"],
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
    model, summary, model_meta = load_screen_model(args.run, device)
    dataset = make_dataset(summary, args)
    stratified = evaluate_at_sigmas(
        model,
        dataset,
        args.sigmas,
        device,
        args.batch_chunks,
        args.noise_seed,
    )
    output = {
        "run": str(args.run),
        "model": model_meta,
        "eval_episodes": int(args.eval_episodes if args.eval_episodes is not None else summary["args"]["eval_episodes"]),
        "eval_seed": int(args.eval_seed if args.eval_seed is not None else summary["args"]["eval_seed"]),
        "cursor_size": summary["args"].get("cursor_size"),
        "sigmas": args.sigmas,
        "stratified": stratified,
    }
    out = args.out or (args.run / "analysis" / "sigma_stratification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
