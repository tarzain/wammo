from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from wammo.eval.autoregressive_ladder import (
    autoregressive_context_ladder,
    load_model,
    make_dataset,
)
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import DeskAction, NotePadDesk, NotepadScriptedPolicy, load_spec
from wammo.train.overfit_one import normalize_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--noise-seed", type=int, default=2024)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def simulator_calibration(
    episodes: int,
    seed: int,
    horizons: tuple[int, ...],
    key_index: int,
) -> dict[str, float]:
    max_horizon = max(horizons)
    states = []
    for episode in range(episodes):
        states.extend(chunk_start_states(seed + episode))
    results: dict[str, float] = {}
    for channel in ("cursor", "click", "key"):
        positive_frames, negative_frames = [], []
        for state in states:
            positive_frames.append(sim_variant_rollout(state, channel, positive=True, key_index=key_index, frames=max_horizon))
            negative_frames.append(sim_variant_rollout(state, channel, positive=False, key_index=key_index, frames=max_horizon))
        positive = np.stack(positive_frames)
        negative = np.stack(negative_frames)
        results.update(sim_channel_metrics(channel, positive, negative, horizons))
    return results


def chunk_start_states(seed: int, chunk_frames: int = 4) -> list[NotePadDesk]:
    for attempt in range(64):
        desk = NotePadDesk(seed=seed + attempt * 10_000)
        policy = NotepadScriptedPolicy(desk, seed=seed + 20_000 + attempt)
        states: list[NotePadDesk] = []
        rare = 0
        for step in range(int(desk.spec["episode_steps"])):
            if step % chunk_frames == 0:
                states.append(copy.deepcopy(desk))
            action = policy.next_action(step)
            if action.mouse_down or action.key != 0:
                rare += 1
            desk.step(action)
        if rare / int(desk.spec["episode_steps"]) >= float(desk.spec["policies"]["rare_event_min_rate"]):
            return states
    return states


def sim_variant_rollout(state: NotePadDesk, channel: str, positive: bool, key_index: int, frames: int) -> np.ndarray:
    desk = copy.deepcopy(state)
    rendered = []
    for _ in range(frames):
        desk.step(sim_action(channel, positive, key_index, desk))
        rendered.append(desk.render())
    return np.stack(rendered)


def sim_action(channel: str, positive: bool, key_index: int, desk: NotePadDesk) -> DeskAction:
    max_delta = float(desk.spec["cursor"]["max_delta"])
    if channel == "cursor":
        return DeskAction(max_delta if positive else -max_delta, 0.0, False, 0)
    if channel == "click":
        return DeskAction(0.0, 0.0, positive, 0)
    if channel == "key":
        return DeskAction(0.0, 0.0, False, key_index if positive else 0)
    raise ValueError(f"unknown channel {channel}")


def sim_channel_metrics(channel: str, positive: np.ndarray, negative: np.ndarray, horizons: tuple[int, ...]) -> dict[str, float]:
    positive_norm = patchify(normalize_frames(positive))
    negative_norm = patchify(normalize_frames(negative))
    patch_mse = (positive_norm - negative_norm).pow(2)
    raw_abs = np.abs(positive.astype(np.int16) - negative.astype(np.int16))
    out: dict[str, float] = {}
    for horizon in horizons:
        frame = horizon - 1
        out[f"sim_ladder_{channel}_h{horizon}"] = float(patch_mse[:, frame].mean())
        out[f"sim_raw_abs_{channel}_h{horizon}"] = float(raw_abs[:, frame].mean())
        out[f"sim_material_frac_{channel}_h{horizon}"] = float((raw_abs[:, frame] > 5).mean())
    return out


def add_ratios(model_metrics: dict[str, float], sim_metrics: dict[str, float]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for key, value in model_metrics.items():
        if not key.startswith("ar_ladder_"):
            continue
        sim_key = "sim_ladder_" + key.removeprefix("ar_ladder_")
        if sim_key not in sim_metrics:
            continue
        denom = sim_metrics[sim_key]
        ratios[f"calibrated_{key.removeprefix('ar_ladder_')}"] = float(value / denom) if denom > 0 else 0.0
    return ratios


def main() -> None:
    args = parse_args()
    spec = load_spec()
    key_index = spec["keys"].index("h")
    horizons = tuple(args.horizons)
    sim_metrics = simulator_calibration(args.eval_episodes, args.eval_seed, horizons, key_index)
    output = {
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "horizons": args.horizons,
        "simulator": sim_metrics,
    }
    if args.run is not None and args.checkpoint is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable; pass --device cpu")
        device = torch.device(args.device)
        model, config_payload, model_meta = load_model(args.run, args.checkpoint, device)
        dataset = make_dataset(config_payload, args.eval_episodes, args.eval_seed)
        model_metrics = autoregressive_context_ladder(model, dataset, device, key_index, horizons, seed=args.noise_seed)
        output["run"] = str(args.run)
        output["model"] = model_meta | {"config": asdict(model.config)}
        output["model_ladder"] = model_metrics
        output["calibrated_ratio"] = add_ratios(model_metrics, sim_metrics)
    out = args.out
    if out is None:
        if args.run is not None:
            step = output.get("model", {}).get("step", "sim")
            out = args.run / "analysis" / f"calibrated_ladder_step_{step}.json"
        else:
            out = Path("runs") / "analysis" / "sim_ladder_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
