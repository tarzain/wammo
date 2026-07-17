from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad import generate_training_dataset
from wammo.train.train_notepad_representation_screen import (
    RepresentationScreenDataset,
    RepresentationScreenModel,
    probe_screen_model,
)
from wammo.notepad_desk import load_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "spatial"], default="spatial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    summary = json.loads((args.run / "summary.json").read_text())
    checkpoint = torch.load(args.run / "checkpoint.pt", map_location=device)
    config = MicroWAMConfig(**checkpoint["config"])
    input_mode = checkpoint["input_mode"]
    patch_size = int(checkpoint.get("patch_size", 4))
    model = RepresentationScreenModel(config, key_count=len(load_spec()["keys"]), input_mode=input_mode, patch_size=patch_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    run_args = summary["args"]
    cursor_size = run_args.get("cursor_size")
    train_frames, train_actions, _ = generate_training_dataset(
        run_args["episodes"],
        run_args["seed"],
        progress_every=0,
        cursor_size=cursor_size,
    )
    eval_frames, eval_actions, _ = generate_training_dataset(
        run_args["eval_episodes"],
        run_args["eval_seed"],
        progress_every=0,
        cursor_size=cursor_size,
    )
    train_dataset = RepresentationScreenDataset(train_frames, train_actions)
    eval_dataset = RepresentationScreenDataset(eval_frames, eval_actions, motion_oversample=False)
    probe = probe_screen_model(
        model,
        train_dataset,
        eval_dataset,
        device,
        args.probe_steps,
        1e-2,
        mlp_hidden=args.mlp_hidden,
        pooling=args.pooling,
    )
    output = {
        "run": str(args.run),
        "input_mode": input_mode,
        "patch_size": patch_size,
        "cursor_size": cursor_size,
        "model": asdict(config),
        "probe": probe,
    }
    out_dir = args.run / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mlp_probe.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
