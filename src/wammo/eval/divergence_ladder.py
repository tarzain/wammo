from __future__ import annotations

import torch

from wammo.model.flow import euler_step_toward_data


HORIZONS = (1, 2, 3, 4)


@torch.no_grad()
def notepad_divergence_ladder(
    model,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    key_index: int,
    seed: int = 2024,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, float]:
    """Measure chunk-local video authority from clamped action variants.

    This is a same-noise, same-context instrument for the current NotePad trainer.
    It is intentionally cheap enough to log during training: every channel uses
    the same video noise and only changes clean action tokens. Until the trainer
    has autoregressive rollout, horizons are limited to frames inside one chunk.
    """
    model.eval()
    generator = torch.Generator(device=video_clean.device).manual_seed(seed)
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    changed_mask = changed_patch_mask(video_clean)
    results: dict[str, float] = {}
    for channel in ("cursor", "click", "key"):
        positive, negative = action_variants(action_clean, channel, key_index)
        pos_video = denoise_video_with_actions(model, video_noise, positive, chunk_ids)
        neg_video = denoise_video_with_actions(model, video_noise, negative, chunk_ids)
        results.update(channel_divergence(channel, pos_video, neg_video, changed_mask, horizons))
    return results


@torch.no_grad()
def notepad_divergence_ladder_samples(
    model,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    key_index: int,
    seed: int = 2024,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, torch.Tensor]:
    model.eval()
    generator = torch.Generator(device=video_clean.device).manual_seed(seed)
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    changed_mask = changed_patch_mask(video_clean)
    results: dict[str, torch.Tensor] = {}
    for channel in ("cursor", "click", "key"):
        positive, negative = action_variants(action_clean, channel, key_index)
        pos_video = denoise_video_with_actions(model, video_noise, positive, chunk_ids)
        neg_video = denoise_video_with_actions(model, video_noise, negative, chunk_ids)
        results.update(channel_divergence_samples(channel, pos_video, neg_video, changed_mask, horizons))
    return results


def action_variants(action_clean: torch.Tensor, channel: str, key_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.zeros_like(action_clean)
    negative = torch.zeros_like(action_clean)
    if channel == "cursor":
        positive[..., 0] = 1.0
        negative[..., 0] = -1.0
    elif channel == "click":
        positive[..., 2] = 1
    elif channel == "key":
        positive[..., 3] = key_index
    else:
        raise ValueError(f"unknown ladder channel {channel}")
    return positive, negative


def denoise_video_with_actions(model, video_noise: torch.Tensor, actions: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
    b = video_noise.shape[0]
    sigma_video = torch.ones((b,), device=video_noise.device)
    sigma_action = torch.zeros((b,), device=video_noise.device)
    delta_actions = actions[..., 0:2]
    button_ids = actions[..., 2].long()
    key_ids = actions[..., 3].long()
    video_velocity, _, _, _ = model(video_noise, delta_actions, button_ids, key_ids, sigma_video, sigma_action, chunk_ids)
    return euler_step_toward_data(video_noise, video_velocity, dt=1.0)


def channel_divergence(
    channel: str,
    positive_video: torch.Tensor,
    negative_video: torch.Tensor,
    changed_mask: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    diff = (positive_video - negative_video).pow(2)
    flat_diff = diff.reshape(-1, *diff.shape[2:])
    flat_mask = changed_mask.reshape(-1, changed_mask.shape[-1])
    out: dict[str, float] = {}
    for horizon in horizons:
        if horizon < 1 or horizon > positive_video.shape[1]:
            raise ValueError(f"horizon {horizon} is outside chunk length {positive_video.shape[1]}")
        frame_idx = horizon - 1
        frame_diff = flat_diff[frame_idx]
        out[f"ladder_{channel}_h{horizon}"] = float(frame_diff.mean())
        mask = flat_mask[frame_idx]
        if bool(mask.any()):
            out[f"ladder_{channel}_changed_h{horizon}"] = float(frame_diff[mask].mean())
        else:
            out[f"ladder_{channel}_changed_h{horizon}"] = 0.0
    return out


def channel_divergence_samples(
    channel: str,
    positive_video: torch.Tensor,
    negative_video: torch.Tensor,
    changed_mask: torch.Tensor,
    horizons: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    diff = (positive_video - negative_video).pow(2)
    out: dict[str, torch.Tensor] = {}
    for horizon in horizons:
        if horizon < 1 or horizon > diff.shape[1]:
            raise ValueError(f"horizon {horizon} is outside chunk length {diff.shape[1]}")
        frame_idx = horizon - 1
        frame_diff = diff[:, frame_idx]
        out[f"ladder_{channel}_h{horizon}"] = frame_diff.mean(dim=(1, 2))
        mask = changed_mask[:, frame_idx]
        changed_values = []
        for sample_diff, sample_mask in zip(frame_diff, mask, strict=True):
            if bool(sample_mask.any()):
                changed_values.append(sample_diff[sample_mask].mean())
            else:
                changed_values.append(torch.zeros((), device=sample_diff.device))
        out[f"ladder_{channel}_changed_h{horizon}"] = torch.stack(changed_values)
    return out


def changed_patch_mask(video_clean: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    flat = video_clean.reshape(-1, *video_clean.shape[2:])
    previous = torch.cat([flat[:1], flat[:-1]], dim=0)
    patch_delta = (flat - previous).abs().mean(dim=-1)
    return (patch_delta > threshold).reshape(video_clean.shape[0], video_clean.shape[1], video_clean.shape[2])
