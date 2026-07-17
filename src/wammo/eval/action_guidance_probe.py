from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from wammo.eval.autoregressive_ladder import (
    blank_context,
    episode_channel_divergence,
    episode_changed_patch_mask,
    load_model,
    make_dataset,
)
from wammo.eval.divergence_ladder import action_variants
from wammo.model.flow import euler_step_toward_data
from wammo.notepad_desk import load_spec
from wammo.train.train_notepad_hybrid import NotePadHybridModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--noise-seed", type=int, default=2024)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 3.0, 8.0])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def guided_autoregressive_ladder(
    model: NotePadHybridModel,
    dataset,
    device: torch.device,
    key_index: int,
    horizons: tuple[int, ...],
    guidance_weight: float,
    seed: int = 2024,
) -> dict[str, float]:
    video_clean, action_clean, _, _ = dataset.all_chunks(device)
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
            pos_video = denoise_guided_chunk(
                model,
                video_noise,
                positive,
                chunk_id,
                pos_context_video,
                pos_context_actions,
                pos_context_ids,
                guidance_weight,
            )
            neg_video = denoise_guided_chunk(
                model,
                video_noise,
                negative,
                chunk_id,
                neg_context_video,
                neg_context_actions,
                neg_context_ids,
                guidance_weight,
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
    return {f"w{guidance_weight:g}_{key}": value for key, value in results.items()}


def denoise_guided_chunk(
    model: NotePadHybridModel,
    video_noise: torch.Tensor,
    actions: torch.Tensor,
    chunk_ids: torch.Tensor,
    context_video: torch.Tensor,
    context_actions: torch.Tensor,
    context_chunk_ids: torch.Tensor,
    guidance_weight: float,
) -> torch.Tensor:
    b = video_noise.shape[0]
    sigma_video = torch.ones((b,), device=video_noise.device)
    sigma_action = torch.zeros((b,), device=video_noise.device)
    null_actions = torch.zeros_like(actions)
    cond_velocity, *_ = model.forward_all(
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
    null_velocity, *_ = model.forward_all(
        video_noise,
        null_actions[..., 0:2],
        null_actions[..., 2].long(),
        null_actions[..., 3].long(),
        sigma_video,
        sigma_action,
        chunk_ids,
        context_video_patches=context_video,
        context_actions=context_actions,
        context_chunk_ids=context_chunk_ids,
    )
    guided_velocity = null_velocity + guidance_weight * (cond_velocity - null_velocity)
    return euler_step_toward_data(video_noise, guided_velocity, dt=1.0)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    model, config_payload, model_meta = load_model(args.run, args.checkpoint, device)
    dataset = make_dataset(config_payload, args.eval_episodes, args.eval_seed)
    key_index = load_spec()["keys"].index("h")
    metrics = {}
    for weight in args.weights:
        metrics.update(
            guided_autoregressive_ladder(
                model,
                dataset,
                device,
                key_index,
                tuple(args.horizons),
                guidance_weight=weight,
                seed=args.noise_seed,
            )
        )
    output = {
        "run": str(args.run),
        "model": model_meta | {"config": asdict(model.config)},
        "eval_episodes": args.eval_episodes,
        "eval_seed": int(args.eval_seed if args.eval_seed is not None else config_payload["args"]["eval_seed"]),
        "horizons": args.horizons,
        "weights": args.weights,
        "metrics": metrics,
    }
    out = args.out or (args.run / "analysis" / f"action_guidance_step_{model_meta.get('step', 'final')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
