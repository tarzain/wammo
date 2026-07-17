import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_representation_screen import (
    RepresentationScreenDataset,
    RepresentationScreenModel,
    screen_training_step,
)


def test_representation_screen_modes_step():
    frames = np.zeros((1, 64, 96, 96, 3), dtype=np.uint8)
    frames[:, :, 10, 20] = [245, 245, 245]
    actions = np.zeros((1, 64, 4), dtype=np.float32)
    dataset = RepresentationScreenDataset(frames, actions, motion_oversample=False)
    video, action, positions, chunk_ids = dataset.sample(1, torch.Generator().manual_seed(0), torch.device("cpu"))
    for mode in ("linear", "coord", "conv"):
        model = RepresentationScreenModel(
            MicroWAMConfig(d_model=32, n_layers=1, n_heads=4, patches_per_frame=576),
            key_count=18,
            input_mode=mode,
        )
        loss, metrics = screen_training_step(model, video, action, positions, chunk_ids, torch.Generator().manual_seed(0), 4, 1, 1)
        assert torch.isfinite(loss)
        assert metrics["cursor_loss"] >= 0
