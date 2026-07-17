import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_pure_inverse import (
    NotePadPureInverseChunks,
    NotePadPureInverseModel,
    augment_frames,
    patch_dim_for_mode,
    pure_inverse_step,
)


def test_augment_frames_coord_diff_channels():
    frames = torch.zeros((1, 4, 96, 96, 3))
    frames[:, 1, 10, 20, 0] = 1
    augmented = augment_frames(frames, "coord-diff")
    assert augmented.shape == (1, 4, 96, 96, 8)
    assert augmented[:, 0, ..., 5:].abs().sum() == 0
    assert augmented[:, 1, ..., 5:].abs().sum() > 0


def test_pure_inverse_step_is_finite():
    frames = np.zeros((1, 64, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((1, 64, 4), dtype=np.float32)
    actions[:, :, 0] = 4
    dataset = NotePadPureInverseChunks(frames, actions, input_mode="coord-diff", motion_oversample=False)
    video, action, chunk_ids = dataset.sample(1, torch.Generator().manual_seed(0), torch.device("cpu"))
    model = NotePadPureInverseModel(
        MicroWAMConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            patch_dim=patch_dim_for_mode("coord-diff", 4),
            patches_per_frame=576,
        )
    )
    loss, metrics = pure_inverse_step(model, video, action, chunk_ids)
    assert torch.isfinite(loss)
    assert metrics["motion_frames"] == 3
