from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from wammo.data.notepad import generate_episode, rare_event_rate
from wammo.eval.divergence_ladder import notepad_divergence_ladder
from wammo.model.dit import MicroWAMConfig
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import (
    NotePadEpisodeChunks,
    NotePadJointModel,
    evaluate,
    normalize_notepad_actions,
    training_step,
    write_contact_sheet,
)
from wammo.train.overfit_one import normalize_frames


class NotePadMultiEpisodeChunks:
    def __init__(self, frames: np.ndarray, actions: np.ndarray, chunk_frames: int = 4, motion_oversample: bool = False):
        if frames.ndim != 5:
            raise ValueError(f"expected ETHWC frames, got {frames.shape}")
        if actions.ndim != 3:
            raise ValueError(f"expected ETA actions, got {actions.shape}")
        if frames.shape[:2] != actions.shape[:2]:
            raise ValueError("frames and actions must share episode/time dimensions")
        if frames.shape[1] % chunk_frames:
            raise ValueError("episode length must be divisible by chunk_frames")
        self.frames = frames
        self.actions = actions
        self.chunk_frames = chunk_frames
        self.chunks_per_episode = frames.shape[1] // chunk_frames
        self.motion_oversample = motion_oversample
        self.motion_pairs = self._motion_pairs()
        spec = load_spec()
        self.max_delta = float(spec["cursor"]["max_delta"])
        self.key_count = len(spec["keys"])

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.motion_oversample and len(self.motion_pairs):
            pair_idx = torch.randint(len(self.motion_pairs), (batch_size,), generator=generator).numpy()
            pairs = self.motion_pairs[pair_idx]
            episode_idx = pairs[:, 0]
            chunk_idx = pairs[:, 1]
        else:
            episode_idx = torch.randint(self.frames.shape[0], (batch_size,), generator=generator).numpy()
            chunk_idx = torch.randint(self.chunks_per_episode, (batch_size,), generator=generator).numpy()
        video_chunks = []
        action_chunks = []
        for ep, chunk in zip(episode_idx, chunk_idx, strict=True):
            start = int(chunk) * self.chunk_frames
            video_chunks.append(self.frames[int(ep), start : start + self.chunk_frames])
            action_chunks.append(self.actions[int(ep), start : start + self.chunk_frames])
        video = normalize_frames(np.stack(video_chunks)).to(device)
        actions = normalize_notepad_actions(np.stack(action_chunks), self.max_delta, self.key_count).to(device)
        return patchify(video), actions, torch.as_tensor(chunk_idx, dtype=torch.long, device=device)

    def _motion_pairs(self) -> np.ndarray:
        pairs = []
        for ep in range(self.actions.shape[0]):
            for chunk in range(self.chunks_per_episode):
                start = chunk * self.chunk_frames
                deltas = self.actions[ep, start : start + self.chunk_frames, 0:2]
                if np.abs(deltas).max() > 0.5:
                    pairs.append((ep, chunk))
        return np.asarray(pairs, dtype=np.int64)


def generate_training_dataset(episodes: int, seed: int, progress_every: int = 100) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spec = load_spec()
    steps = int(spec["episode_steps"])
    width = int(spec["canvas"]["width"])
    height = int(spec["canvas"]["height"])
    frames = np.empty((episodes, steps, height, width, 3), dtype=np.uint8)
    actions = np.empty((episodes, steps, 4), dtype=np.float32)
    start = time.time()
    for i in range(episodes):
        ep_frames, ep_actions = generate_episode(seed + i)
        frames[i] = ep_frames
        actions[i] = ep_actions
        if progress_every > 0 and (i + 1 == 1 or (i + 1) % progress_every == 0 or i + 1 == episodes):
            elapsed = time.time() - start
            print(f"generated episodes={i + 1}/{episodes} rare_event_rate={rare_event_rate(actions[: i + 1]):.3f} elapsed={elapsed:.1f}s")
    metadata = {
        "episodes": episodes,
        "rare_event_rate": rare_event_rate(actions),
        "frames_shape": list(frames.shape),
        "actions_shape": list(actions.shape),
    }
    return frames, actions, metadata


def make_eval_dataset(seed: int) -> tuple[NotePadEpisodeChunks, dict[str, float]]:
    spec = load_spec()
    frames, actions = generate_episode(seed)
    dataset = NotePadEpisodeChunks(
        normalize_frames(frames),
        normalize_notepad_actions(actions, float(spec["cursor"]["max_delta"]), len(spec["keys"])),
    )
    return dataset, {"eval_seed": seed, "eval_rare_event_rate": rare_event_rate(actions)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/notepad-1k"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ladder-every", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-dropout", type=float, default=0.0)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--motion-oversample", action="store_true")
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
    train_dataset = NotePadMultiEpisodeChunks(train_frames, train_actions, motion_oversample=args.motion_oversample)
    eval_dataset, eval_metadata = make_eval_dataset(args.eval_seed)

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
    payload = {
        "args": vars(args) | {"out": str(args.out)},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
    }
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    first_eval: dict[str, float] | None = None
    start = time.time()
    metrics_path = args.out / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = training_step(
                model, video, actions, chunk_ids, noise_generator, args.action_weight, args.action_dropout, args.delta_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate(model, eval_dataset, device, args.action_weight, args.action_dropout, args.delta_weight)
                if args.ladder_every > 0 and (step == 1 or step % args.ladder_every == 0 or step == args.steps):
                    video_all, action_all, chunk_ids_all = eval_dataset.all_chunks(device)
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
    write_contact_sheet(model, eval_dataset, args.out / "contact_sheet.png", device)
    final_eval = evaluate(model, eval_dataset, device, args.action_weight, args.action_dropout, args.delta_weight)
    video_all, action_all, chunk_ids_all = eval_dataset.all_chunks(device)
    final_eval.update(
        notepad_divergence_ladder(model, video_all, action_all, chunk_ids_all, key_index=load_spec()["keys"].index("h"))
    )
    (args.out / "summary.json").write_text(json.dumps({"first_eval": first_eval, "final_eval": final_eval}, indent=2) + "\n")


if __name__ == "__main__":
    main()
