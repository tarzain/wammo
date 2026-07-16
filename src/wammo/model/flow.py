from __future__ import annotations

import torch


def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Flow convention: x_t = (1 - t) * x0 + t * noise."""
    while t.ndim < x0.ndim:
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * noise


def velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Flow convention: v = noise - x0."""
    return noise - x0


def euler_step_toward_data(x_t: torch.Tensor, predicted_v: torch.Tensor, dt: float) -> torch.Tensor:
    """Move from larger t toward smaller t; subtract velocity when denoising."""
    return x_t - dt * predicted_v

