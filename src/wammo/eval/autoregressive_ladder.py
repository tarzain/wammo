from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from wammo.eval.divergence_ladder import action_variants
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import euler_step_toward_data
from wammo.notepad_desk import load_spec
from wammo.train.train_notepad import generate_training_dataset
from wammo.train.train_notepad_hybrid import NotePadHybridChunks, NotePadHybridModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--noise-seed", type=int, default=2024)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_model(run: Path, checkpoint_path: Path, device: torch.device) -> tuple[NotePadHybridModel, dict[str, object], dict[str, object]]:
    config_payload = json.loads((run / "config.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MicroWAMConfig(**checkpoint["config"])
    run_args = config_payload.get("args", {})
    head_sigma_conditioned = bool(checkpoint.get("head_sigma_conditioned", run_args.get("head_sigma_conditioned", False)))
    model = NotePadHybridModel(config, key_count=len(load_spec()["keys"]), head_sigma_conditioned=head_sigma_conditioned).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, config_payload, {
        "config": asdict(config),
        "head_sigma_conditioned": head_sigma_conditioned,
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step"),
    }


def make_dataset(config_payload: dict[str, object], eval_episodes: int, eval_seed: int | None) -> NotePadHybridChunks:
    run_args = config_payload.get("args", {})
    if not isinstance(run_args, dict):
        raise ValueError("config args must be a dict")
    seed = int(eval_seed if eval_seed is not None else run_args["eval_seed"])
    frames, actions, _ = generate_training_dataset(eval_episodes, seed, progress_every=0)
    return NotePadHybridChunks(frames, actions, motion_oversample=False)


@torch.no_grad()
def autoregressive_context_ladder(
    model: NotePadHybridModel,
    dataset: NotePadHybridChunks,
    device: torch.device,
    key_index: int,
    horizons: tuple[int, ...],
    seed: int = 2024,
) -> dict[str, float]:
    model.eval()
    if max(horizons) > dataset.frames.shape[1]:
        raise ValueError(f"horizon {max(horizons)} exceeds episode length {dataset.frames.shape[1]}")
    video_clean, action_clean, _, chunk_ids = dataset.all_chunks(device)
    episodes = dataset.frames.shape[0]
    chunks = dataset.chunks_per_episode
    chunk_frames = dataset.chunk_frames
    video_clean = video_clean.reshape(episodes, chunks, chunk_frames, dataset.width // 4 * dataset.height // 4, 48)
    action_clean = action_clean.reshape(episodes, chunks, chunk_frames, 4)
    clean_episode = video_clean.reshape(episodes, chunks * chunk_frames, video_clean.shape[-2], video_clean.shape[-1])
    changed_mask = episode_changed_patch_mask(clean_episode)
    results: dict[str, float] = {}
    for channel in ("cursor", "click", "key"):
        positive_chunks, negative_chunks = [], []
        pos_context_video, pos_context_actions, pos_context_ids = blank_context(dataset, episodes, device)
        neg_context_video, neg_context_actions, neg_context_ids = blank_context(dataset, episodes, device)
        generator = torch.Generator(device=device).manual_seed(seed)
        for chunk in range(chunks):
            actions = action_clean[:, chunk]
            positive, negative = action_variants(actions, channel, key_index)
            video_noise = torch.randn(video_clean[:, chunk].shape, device=device, generator=generator)
            chunk_id = torch.full((episodes,), chunk, dtype=torch.long, device=device)
            pos_video = denoise_context_chunk(
                model, video_noise, positive, chunk_id, pos_context_video, pos_context_actions, pos_context_ids
            )
            neg_video = denoise_context_chunk(
                model, video_noise, negative, chunk_id, neg_context_video, neg_context_actions, neg_context_ids
            )
            positive_chunks.append(pos_video)
            negative_chunks.append(neg_video)
            pos_context_video = pos_video.unsqueeze(1)
            pos_context_actions = positive.unsqueeze(1)
            pos_context_ids = chunk_id.reshape(episodes, 1)
            neg_context_video = neg_video.unsqueeze(1)
            neg_context_actions = negative.unsqueeze(1)
            neg_context_ids = chunk_id.reshape(episodes, 1)
        positive_episode = torch.stack(positive_chunks, dim=1).reshape_as(clean_episode)
        negative_episode = torch.stack(negative_chunks, dim=1).reshape_as(clean_episode)
        results.update(episode_channel_divergence(channel, positive_episode, negative_episode, changed_mask, horizons))
    return results


def blank_context(
    dataset: NotePadHybridChunks, episodes: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_video = torch.zeros((episodes, 1, dataset.chunk_frames, dataset.width // 4 * dataset.height // 4, 48), device=device)
    context_actions = torch.zeros((episodes, 1, dataset.chunk_frames, 4), device=device)
    context_ids = torch.zeros((episodes, 1), dtype=torch.long, device=device)
    return context_video, context_actions, context_ids


def denoise_context_chunk(
    model: NotePadHybridModel,
    video_noise: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    context_video: torch.Tensor,
    context_actions: torch.Tensor,
    context_chunk_ids: torch.Tensor,
) -> torch.Tensor:
    b = video_noise.shape[0]
    sigma_video = torch.ones((b,), device=video_noise.device)
    sigma_action = torch.zeros((b,), device=video_noise.device)
    video_velocity, *_ = model.forward_all(
        video_noise,
        actions[..., 0:2],
        actions[..., 2].long(),
        actions[..., 3].long(),
        sigma_video,
        sigma_action,
        chunk_ids,
        context_video_patches=context_video,
        context_actions=context_actions,
        context_chunk_ids=context_chunk_ids,
    )
    return euler_step_toward_data(video_noise, video_velocity, dt=1.0)


def episode_changed_patch_mask(video_clean: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    previous = torch.cat([video_clean[:, :1], video_clean[:, :-1]], dim=1)
    patch_delta = (video_clean - previous).abs().mean(dim=-1)
    return patch_delta > threshold


def episode_channel_divergence(
    channel: str,
    positive_video: torch.Tensor,
    negative_video: torch.Tensor,
    changed_mask: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    diff = (positive_video - negative_video).pow(2)
    out: dict[str, float] = {}
    for horizon in horizons:
        if horizon < 1 or horizon > diff.shape[1]:
            raise ValueError(f"horizon {horizon} is outside episode length {diff.shape[1]}")
        frame_idx = horizon - 1
        frame_diff = diff[:, frame_idx]
        out[f"ar_ladder_{channel}_h{horizon}"] = float(frame_diff.mean())
        mask = changed_mask[:, frame_idx]
        out[f"ar_ladder_{channel}_changed_h{horizon}"] = float(frame_diff[mask].mean()) if bool(mask.any()) else 0.0
    return out


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, config_payload, model_meta = load_model(args.run, args.checkpoint, device)
    dataset = make_dataset(config_payload, args.eval_episodes, args.eval_seed)
    metrics = autoregressive_context_ladder(
        model,
        dataset,
        device,
        key_index=load_spec()["keys"].index("h"),
        horizons=tuple(args.horizons),
        seed=args.noise_seed,
    )
    output = {
        "run": str(args.run),
        "model": model_meta,
        "eval_episodes": args.eval_episodes,
        "eval_seed": int(args.eval_seed if args.eval_seed is not None else config_payload["args"]["eval_seed"]),
        "horizons": args.horizons,
        "metrics": metrics,
    }
    out = args.out or (args.run / "analysis" / f"autoregressive_ladder_step_{model_meta.get('step', 'unknown')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
