from __future__ import annotations

import torch


def patchify(frames: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """Convert BCHW or BTHWC frames to patch vectors."""
    if frames.ndim == 5:
        b, t, h, w, c = frames.shape
        x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        patches = _patchify_bchw(x, patch_size)
        return patches.reshape(b, t, patches.shape[1], patches.shape[2])
    if frames.ndim == 4:
        return _patchify_bchw(frames, patch_size)
    raise ValueError(f"expected BCHW or BTHWC tensor, got shape {tuple(frames.shape)}")


def unpatchify(patches: torch.Tensor, height: int = 64, width: int = 64, patch_size: int = 4) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError(f"expected BNP tensor, got shape {tuple(patches.shape)}")
    b, n, dim = patches.shape
    channels = dim // (patch_size * patch_size)
    grid_h = height // patch_size
    grid_w = width // patch_size
    if n != grid_h * grid_w:
        raise ValueError(f"expected {grid_h * grid_w} patches, got {n}")
    x = patches.reshape(b, grid_h, grid_w, channels, patch_size, patch_size)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, height, width)


def _patchify_bchw(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = frames.shape
    if h % patch_size or w % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    x = frames.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch_size) * (w // patch_size), c * patch_size * patch_size)

