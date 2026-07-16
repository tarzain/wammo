from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MicroWAMConfig:
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    patch_dim: int = 4 * 4 * 3
    action_dim: int = 3
    chunk_frames: int = 4
    patches_per_frame: int = 16 * 16
    max_chunks: int = 16


class MicroWAMDiT(nn.Module):
    """Joint video/action denoising transformer skeleton.

    This preserves the intended interface: noisy video patches and noisy action
    vectors enter one transformer and produce velocity predictions for both.
    Full 3D RoPE, chunk KV caching, and ONNX-friendly cache plumbing are the next
    implementation milestone.
    """

    def __init__(self, config: MicroWAMConfig = MicroWAMConfig()):
        super().__init__()
        self.config = config
        self.video_in = nn.Linear(config.patch_dim, config.d_model)
        self.action_in = nn.Linear(config.action_dim, config.d_model)
        self.video_out = nn.Linear(config.d_model, config.patch_dim)
        self.action_out = nn.Linear(config.d_model, config.action_dim)
        self.video_pos = nn.Parameter(torch.zeros(1, config.chunk_frames * config.patches_per_frame, config.d_model))
        self.action_pos = nn.Parameter(torch.zeros(1, config.chunk_frames, config.d_model))
        self.chunk_pos = nn.Embedding(config.max_chunks, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.video_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        self.action_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))

    def forward(
        self,
        video_patches: torch.Tensor,
        actions: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, p, d = video_patches.shape
        if c != self.config.chunk_frames or p != self.config.patches_per_frame or d != self.config.patch_dim:
            raise ValueError(f"unexpected video patch shape {tuple(video_patches.shape)}")
        if actions.shape != (b, c, self.config.action_dim):
            raise ValueError(f"unexpected action shape {tuple(actions.shape)}")

        if chunk_ids is None:
            chunk_ids = torch.zeros((b,), dtype=torch.long, device=video_patches.device)
        if chunk_ids.shape != (b,):
            raise ValueError(f"unexpected chunk_ids shape {tuple(chunk_ids.shape)}")

        chunk_tokens = self.chunk_pos(chunk_ids).unsqueeze(1)
        video_tokens = self.video_in(video_patches).reshape(b, c * p, self.config.d_model)
        action_tokens = self.action_in(actions)
        video_tokens = video_tokens + self.video_pos + chunk_tokens
        action_tokens = action_tokens + self.action_pos + chunk_tokens
        video_tokens = video_tokens + self.video_sigma(sigma_video.reshape(b, 1)).unsqueeze(1)
        action_tokens = action_tokens + self.action_sigma(sigma_action.reshape(b, 1)).unsqueeze(1)

        tokens = torch.cat([video_tokens, action_tokens], dim=1)
        hidden = self.backbone(tokens)
        video_hidden = hidden[:, : c * p].reshape(b, c, p, self.config.d_model)
        action_hidden = hidden[:, c * p :]
        return self.video_out(video_hidden), self.action_out(action_hidden)
