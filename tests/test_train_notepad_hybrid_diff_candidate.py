import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_hybrid import NotePadHybridModel
from wammo.train.train_notepad_hybrid_diff_candidate import NotePadHybridFrameChunks, diff_hybrid_training_step
from wammo.train.train_notepad_pure_inverse import patch_dim_for_mode


def test_diff_candidate_training_step_is_finite():
    frames = np.zeros((1, 64, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((1, 64, 4), dtype=np.float32)
    actions[:, :, 0] = 4
    dataset = NotePadHybridFrameChunks(frames, actions, "coord-diff", motion_oversample=False)
    video, action, positions, chunk_ids = dataset.sample(1, torch.Generator().manual_seed(0), torch.device("cpu"))
    model = NotePadHybridModel(
        MicroWAMConfig(
            d_model=32,
            n_layers=1,
            n_heads=4,
            patch_dim=patch_dim_for_mode("coord-diff", 4),
            patches_per_frame=576,
        ),
        key_count=18,
        head_sigma_conditioned=True,
        video_output_dim=48,
    )
    loss, metrics = diff_hybrid_training_step(
        model,
        video,
        action,
        positions,
        chunk_ids,
        torch.Generator().manual_seed(0),
        "coord-diff",
        1.0,
        0.0,
        4.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.0,
        1.0,
    )
    assert torch.isfinite(loss)
    assert metrics["inverse_motion_frames"] == 3
