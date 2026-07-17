from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from wammo.notepad_desk import DeskAction, NotePadDesk, load_spec
from wammo.train.train_notepad_next_frame_baseline import (
    NextFrameBaselineConfig,
    NotePadNextFrameCNN,
    rollout_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_baseline(checkpoint_path: Path, device: torch.device) -> tuple[NotePadNextFrameCNN, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = NextFrameBaselineConfig(**checkpoint["config"])
    model = NotePadNextFrameCNN(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {"config": asdict(config), "checkpoint": str(checkpoint_path), "step": checkpoint.get("step")}


def fixed_state(seed: int) -> NotePadDesk:
    desk = NotePadDesk(seed=seed)
    return desk


def action_rows(key_index: int, max_delta: float, frames: int = 4) -> dict[str, list[DeskAction]]:
    return {
        "idle": [DeskAction(0.0, 0.0, False, 0) for _ in range(frames)],
        "cursor-right": [DeskAction(max_delta, 0.0, False, 0) for _ in range(frames)],
        "mouse-down": [DeskAction(0.0, 0.0, True, 0) for _ in range(frames)],
        "type-h": [DeskAction(0.0, 0.0, False, key_index)] + [DeskAction(0.0, 0.0, False, 0) for _ in range(frames - 1)],
    }


def scripted_sequence(key_index: int, max_delta: float) -> list[DeskAction]:
    # From the default cursor start near the middle, walk toward the toolbar new-note button,
    # click/release, type h, then move away. The model never sees simulator state updates here;
    # this is deliberately an autoregressive learned-world rollout.
    return [
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(-max_delta, -max_delta, False, 0),
        DeskAction(0.0, 0.0, True, 0),
        DeskAction(0.0, 0.0, False, 0),
        DeskAction(0.0, 0.0, False, key_index),
        DeskAction(max_delta, max_delta, False, 0),
        DeskAction(max_delta, max_delta, False, 0),
        DeskAction(max_delta, 0.0, False, 0),
        DeskAction(max_delta, 0.0, False, 0),
        DeskAction(0.0, max_delta, False, 0),
        DeskAction(0.0, max_delta, False, 0),
        DeskAction(0.0, 0.0, False, 0),
    ]


def write_grid(rows: dict[str, np.ndarray], out: Path) -> None:
    labels = list(rows)
    first = next(iter(rows.values()))
    row_count = len(rows)
    cols = first.shape[0]
    height, width = first.shape[1:3]
    label_w = 96
    sheet = Image.new("RGB", (label_w + cols * width, row_count * height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, label in enumerate(labels):
        y = row_idx * height
        draw.text((4, y + 4), label, fill=(0, 0, 0))
        for col_idx, frame in enumerate(rows[label]):
            sheet.paste(Image.fromarray(frame), (label_w + col_idx * width, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def write_strip(frames: np.ndarray, out: Path) -> None:
    height, width = frames.shape[1:3]
    sheet = Image.new("RGB", (frames.shape[0] * width, height), "white")
    for idx, frame in enumerate(frames):
        sheet.paste(Image.fromarray(frame), (idx * width, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def pixel_diff_metrics(rows: dict[str, np.ndarray]) -> dict[str, float]:
    idle = rows["idle"].astype(np.int16)
    metrics: dict[str, float] = {}
    for label, frames in rows.items():
        if label == "idle":
            continue
        diff = np.abs(frames.astype(np.int16) - idle)
        metrics[f"{label}_mean_abs"] = float(diff.mean())
        metrics[f"{label}_material_frac"] = float((diff > 5).mean())
    return metrics


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, meta = load_baseline(args.checkpoint, device)
    spec = load_spec()
    key_index = spec["keys"].index("h")
    max_delta = float(spec["cursor"]["max_delta"])
    desk = fixed_state(args.seed)
    initial = desk.render()

    rows = {
        label: rollout_model(model, initial, actions, device)
        for label, actions in action_rows(key_index, max_delta).items()
    }
    sequence = rollout_model(model, initial, scripted_sequence(key_index, max_delta), device)
    args.out.mkdir(parents=True, exist_ok=True)
    write_grid(rows, args.out / "fixed_action_rows.png")
    write_strip(sequence, args.out / "scripted_sequence.png")
    report = {
        "model": meta,
        "seed": args.seed,
        "fixed_action_rows": str(args.out / "fixed_action_rows.png"),
        "scripted_sequence": str(args.out / "scripted_sequence.png"),
        "pixel_diff": pixel_diff_metrics(rows),
    }
    (args.out / "rollout_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
