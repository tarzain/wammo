import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_hybrid import (
    NotePadHybridChunks,
    NotePadHybridModel,
    hybrid_training_step,
)


def test_hybrid_sampler_shapes():
    frames = np.zeros((2, 64, 96, 96, 3), dtype=np.uint8)
    frames[:, :, 10, 20] = [245, 245, 245]
    actions = np.zeros((2, 64, 4), dtype=np.float32)
    actions[1, 8, 0] = 4
    dataset = NotePadHybridChunks(frames, actions, motion_oversample=True)
    video, action, positions, chunk_ids = dataset.sample(2, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert video.shape == (2, 4, 576, 48)
    assert action.shape == (2, 4, 4)
    assert positions.shape == (2, 4, 2)
    assert chunk_ids.eq(2).all()


def test_hybrid_training_step_is_finite():
    model = NotePadHybridModel(
        MicroWAMConfig(d_model=32, n_layers=1, n_heads=4, patches_per_frame=576),
        key_count=18,
    )
    video = torch.randn(1, 4, 576, 48)
    actions = torch.zeros(1, 4, 4)
    positions = torch.full((1, 4, 2), 0.5)
    chunk_ids = torch.zeros((1,), dtype=torch.long)
    loss, metrics = hybrid_training_step(model, video, actions, positions, chunk_ids, torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert metrics["cursor_loss"] > 0
