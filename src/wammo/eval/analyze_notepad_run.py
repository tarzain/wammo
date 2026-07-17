from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from wammo.data.notepad import generate_episode, rare_event_rate
from wammo.eval.divergence_ladder import notepad_divergence_ladder_samples
from wammo.model.dit import MicroWAMConfig
from wammo.model.tokenizer import patchify
from wammo.eval.notepad_pixels import cursor_centroids
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import NotePadJointModel, normalize_notepad_actions
from wammo.train.train_notepad_binned_delta import NotePadBinnedDeltaModel
from wammo.train.train_notepad_hybrid import NotePadHybridModel
from wammo.train.overfit_one import normalize_frames


def delta_baselines(actions: np.ndarray) -> dict[str, float | list[float]]:
    deltas = actions[..., 0:2].reshape(-1, 2)
    mean_delta = deltas.mean(axis=0)
    zero_mae = np.abs(deltas).mean()
    mean_mae = np.abs(deltas - mean_delta).mean()
    return {
        "zero_delta_mae_px": float(zero_mae),
        "mean_delta": [float(mean_delta[0]), float(mean_delta[1])],
        "mean_delta_mae_px": float(mean_mae),
    }


def regenerate_episodes(episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_frames = []
    all_actions = []
    for i in range(episodes):
        frames, actions = generate_episode(seed + i)
        all_frames.append(frames)
        all_actions.append(actions)
    return np.stack(all_frames), np.stack(all_actions)


def regenerate_actions(episodes: int, seed: int) -> np.ndarray:
    _, actions = regenerate_episodes(episodes, seed)
    return actions


def make_battery(seeds: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    spec = load_spec()
    frames, actions = [], []
    for seed in seeds:
        ep_frames, ep_actions = generate_episode(seed)
        frames.append(ep_frames)
        actions.append(ep_actions)
    frames_np = np.stack(frames)
    actions_np = np.stack(actions)
    frames_t = normalize_frames(frames_np.reshape(-1, *frames_np.shape[2:]))
    actions_t = normalize_notepad_actions(
        actions_np.reshape(-1, actions_np.shape[-1]),
        float(spec["cursor"]["max_delta"]),
        len(spec["keys"]),
    )
    frames_t = frames_t.reshape(len(seeds) * 16, 4, *frames_t.shape[1:])
    actions_t = actions_t.reshape(len(seeds) * 16, 4, actions_t.shape[-1])
    chunk_ids = torch.arange(16).repeat(len(seeds))
    metadata = {
        "battery_episodes": len(seeds),
        "battery_chunks": int(frames_t.shape[0]),
        "battery_rare_event_rate": rare_event_rate(actions_np),
    }
    return patchify(frames_t), actions_t, chunk_ids, metadata


@torch.no_grad()
def predict_delta_x0(
    model: NotePadJointModel | NotePadBinnedDeltaModel,
    video: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    device: torch.device,
    seed: int = 3030,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    video = video.to(device)
    actions = actions.to(device)
    chunk_ids = chunk_ids.to(device)
    delta_noise = torch.randn(actions[..., 0:2].shape, device=device, generator=generator)
    sigma_video = torch.zeros((video.shape[0],), device=device)
    sigma_action = torch.ones((video.shape[0],), device=device)
    _, delta_output, _, _ = model(
        video,
        delta_noise,
        actions[..., 2].long(),
        actions[..., 3].long(),
        sigma_video,
        sigma_action,
        chunk_ids,
    )
    if getattr(model, "delta_prediction_kind", "velocity") == "x0":
        return delta_output.clamp(-1.0, 1.0)
    return (delta_noise - delta_output).clamp(-1.0, 1.0)


def delta_prediction_diagnostic(true_actions: torch.Tensor, pred_delta: torch.Tensor, max_delta: float) -> dict[str, object]:
    true_delta = true_actions[..., 0:2].detach().cpu().numpy() * max_delta
    pred_delta_px = pred_delta.detach().cpu().numpy() * max_delta
    motion = np.abs(true_delta).max(axis=-1) > 0.5
    bins = np.arange(-8.5, 9.5, 1.0)
    true_motion = true_delta[motion]
    pred_motion = pred_delta_px[motion]
    if true_motion.size == 0:
        raise ValueError("no motion frames found for delta diagnostic")
    mean_motion = true_motion.mean(axis=0)
    return {
        "motion_frames": int(motion.sum()),
        "total_frames": int(motion.size),
        "motion_rate": float(motion.mean()),
        "motion_zero_delta_mae_px": float(np.abs(true_motion).mean()),
        "motion_mean_delta": [float(mean_motion[0]), float(mean_motion[1])],
        "motion_mean_delta_mae_px": float(np.abs(true_motion - mean_motion).mean()),
        "model_motion_delta_mae_px": float(np.abs(pred_motion - true_motion).mean()),
        "model_motion_pred_abs_mean_px": float(np.abs(pred_motion).mean()),
        "model_motion_pred_near_zero_rate": float((np.abs(pred_motion).max(axis=-1) < 0.5).mean()),
        "true_motion_dx_hist": np.histogram(true_motion[:, 0], bins=bins)[0].astype(int).tolist(),
        "pred_motion_dx_hist": np.histogram(pred_motion[:, 0], bins=bins)[0].astype(int).tolist(),
        "true_motion_dy_hist": np.histogram(true_motion[:, 1], bins=bins)[0].astype(int).tolist(),
        "pred_motion_dy_hist": np.histogram(pred_motion[:, 1], bins=bins)[0].astype(int).tolist(),
        "hist_bins": bins.tolist(),
    }


def delta_by_chunk_position(true_actions: torch.Tensor, pred_delta: torch.Tensor, max_delta: float) -> dict[str, dict[str, float]]:
    true_delta = true_actions[..., 0:2].detach().cpu().numpy() * max_delta
    pred_delta_px = pred_delta.detach().cpu().numpy() * max_delta
    out = {}
    for pos in range(true_delta.shape[1]):
        true_pos = true_delta[:, pos]
        pred_pos = pred_delta_px[:, pos]
        motion = np.abs(true_pos).max(axis=-1) > 0.5
        if motion.any():
            true_motion = true_pos[motion]
            pred_motion = pred_pos[motion]
            mean_motion = true_motion.mean(axis=0)
            motion_zero = float(np.abs(true_motion).mean())
            motion_mean = float(np.abs(true_motion - mean_motion).mean())
            model_motion = float(np.abs(pred_motion - true_motion).mean())
            pred_abs = float(np.abs(pred_motion).mean())
            near_zero = float((np.abs(pred_motion).max(axis=-1) < 0.5).mean())
        else:
            motion_zero = motion_mean = model_motion = pred_abs = near_zero = 0.0
        out[f"pos_{pos + 1}"] = {
            "frames": int(true_pos.shape[0]),
            "motion_frames": int(motion.sum()),
            "motion_rate": float(motion.mean()),
            "zero_delta_mae_px": motion_zero,
            "mean_delta_mae_px": motion_mean,
            "model_delta_mae_px": model_motion,
            "model_pred_abs_mean_px": pred_abs,
            "model_pred_near_zero_rate": near_zero,
        }
    return out


def delta_sign_audit(frames: np.ndarray, actions: np.ndarray) -> dict[str, object]:
    positions = cursor_centroids(frames)
    observed = positions[:, 1:] - positions[:, :-1]
    target = actions[:, 1:, 0:2]
    valid = np.isfinite(observed).all(axis=-1)
    observed_valid = observed[valid]
    target_valid = target[valid]
    if observed_valid.size == 0:
        raise ValueError("no valid cursor centroid transitions found")
    error = observed_valid - target_valid
    out: dict[str, object] = {
        "valid_transitions": int(valid.sum()),
        "mean_observed_dxdy": observed_valid.mean(axis=0).astype(float).tolist(),
        "mean_target_dxdy": target_valid.mean(axis=0).astype(float).tolist(),
        "mean_error_dxdy": error.mean(axis=0).astype(float).tolist(),
        "mae_error_px": float(np.abs(error).mean()),
        "corr_observed_target_dx": float(np.corrcoef(observed_valid[:, 0], target_valid[:, 0])[0, 1]),
        "corr_observed_target_dy": float(np.corrcoef(observed_valid[:, 1], target_valid[:, 1])[0, 1]),
        "target_positive_dx_rate": float((target_valid[:, 0] > 0.5).mean()),
        "target_negative_dx_rate": float((target_valid[:, 0] < -0.5).mean()),
        "target_positive_dy_rate": float((target_valid[:, 1] > 0.5).mean()),
        "target_negative_dy_rate": float((target_valid[:, 1] < -0.5).mean()),
    }
    return out


def summarize_samples(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    summary = {}
    for key, values in samples.items():
        values = values.detach().float().cpu()
        std = values.std(unbiased=False)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(std),
            "sem": float(std / max(1, values.numel()) ** 0.5),
            "n": int(values.numel()),
        }
    return summary


def load_model(run_dir: Path, device: torch.device) -> NotePadJointModel | NotePadBinnedDeltaModel:
    config_payload = json.loads((run_dir / "config.json").read_text())
    model_config = MicroWAMConfig(**config_payload["model"])
    if config_payload.get("model_kind") == "notepad_binned_delta":
        model = NotePadBinnedDeltaModel(model_config, key_count=len(load_spec()["keys"])).to(device)
    elif config_payload.get("model_kind") == "notepad_hybrid":
        model = NotePadHybridModel(
            model_config,
            key_count=len(load_spec()["keys"]),
            head_sigma_conditioned=bool(config_payload.get("args", {}).get("head_sigma_conditioned", False)),
        ).to(device)
    else:
        model = NotePadJointModel(model_config, key_count=len(load_spec()["keys"])).to(device)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def metrics_correlations(run_dir: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    ladder_rows = [row for row in rows if "eval_ladder_cursor_h4" in row]
    out: dict[str, float] = {}
    if len(ladder_rows) < 2:
        return out
    video = np.array([row["eval_video_loss"] for row in ladder_rows], dtype=np.float64)
    for channel in ("cursor", "click", "key"):
        metric = f"eval_ladder_{channel}_h4"
        if metric not in ladder_rows[0]:
            continue
        values = np.array([row[metric] for row in ladder_rows], dtype=np.float64)
        if np.std(values) > 0 and np.std(video) > 0:
            out[f"corr_video_loss_ladder_{channel}_h4"] = float(np.corrcoef(video, values)[0, 1])
    return out


def write_ladder_csv(summary: dict[str, dict[str, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "sem", "n"])
        writer.writeheader()
        for metric, values in sorted(summary.items()):
            writer.writerow({"metric": metric, **values})


def write_histogram_csv(diagnostic: dict[str, object], out: Path) -> None:
    bins = diagnostic["hist_bins"]
    rows = zip(
        bins[:-1],
        bins[1:],
        diagnostic["true_motion_dx_hist"],
        diagnostic["pred_motion_dx_hist"],
        diagnostic["true_motion_dy_hist"],
        diagnostic["pred_motion_dy_hist"],
        strict=True,
    )
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "bin_right", "true_dx", "pred_dx", "true_dy", "pred_dy"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/notepad-1k"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--battery-size", type=int, default=64)
    parser.add_argument("--battery-seed", type=int, default=200_000)
    parser.add_argument("--probe-train-episodes", type=int, default=256)
    parser.add_argument("--probe-eval-episodes", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--no-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    config = json.loads((args.run / "config.json").read_text())
    train_args = config["args"]
    analysis_dir = args.run / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    train_actions = regenerate_actions(train_args["episodes"], train_args["seed"])
    eval_frames, eval_actions = regenerate_episodes(1, train_args["eval_seed"])
    baseline_summary = {
        "train": {
            **delta_baselines(train_actions),
            "rare_event_rate": rare_event_rate(train_actions),
        },
        "eval": {
            **delta_baselines(eval_actions),
            "rare_event_rate": rare_event_rate(eval_actions),
        },
    }

    battery_seeds = [args.battery_seed + i for i in range(args.battery_size)]
    video, actions, chunk_ids, battery_metadata = make_battery(battery_seeds)
    model = load_model(args.run, device)
    pred_delta = predict_delta_x0(model, video, actions, chunk_ids, device)
    delta_diagnostic = delta_prediction_diagnostic(actions, pred_delta, max_delta=float(load_spec()["cursor"]["max_delta"]))
    position_diagnostic = delta_by_chunk_position(actions, pred_delta, max_delta=float(load_spec()["cursor"]["max_delta"]))
    sign_audit = delta_sign_audit(eval_frames, eval_actions)
    write_histogram_csv(delta_diagnostic, analysis_dir / "delta_prediction_histogram.csv")
    samples = notepad_divergence_ladder_samples(
        model,
        video.to(device),
        actions.to(device),
        chunk_ids.to(device),
        key_index=load_spec()["keys"].index("h"),
    )
    ladder_summary = summarize_samples(samples)
    write_ladder_csv(ladder_summary, analysis_dir / "ladder_battery.csv")

    output = {
        "run": str(args.run),
        "model": asdict(model.config),
        "delta_baselines": baseline_summary,
        "battery": battery_metadata,
        "delta_prediction": delta_diagnostic,
        "delta_by_chunk_position": position_diagnostic,
        "delta_sign_audit": sign_audit,
        "ladder_battery": ladder_summary,
        "metrics_correlations": metrics_correlations(args.run),
    }
    if not args.no_probe:
        from wammo.eval.probe_notepad import run_probe

        output["linear_probe"] = run_probe(
            args.run,
            device,
            train_episodes=args.probe_train_episodes,
            eval_episodes=args.probe_eval_episodes,
            steps=args.probe_steps,
        )
    (analysis_dir / "analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
