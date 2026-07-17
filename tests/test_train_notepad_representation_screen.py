import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_representation_screen import (
    RepresentationScreenDataset,
    RepresentationScreenModel,
    cursor_patch_targets,
    decode_cursor_heatmap,
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
        assert metrics["cursor_heatmap_loss"] >= 0
        assert metrics["cursor_offset_loss"] >= 0
        assert metrics["cursor_decoded_mae_px"] >= 0


def test_cursor_patch_targets_decode_round_trip():
    positions = torch.tensor([[[10.0 / 95.0, 18.0 / 95.0], [95.0 / 95.0, 0.0 / 95.0]]])
    patch_index, offsets = cursor_patch_targets(positions, patch_size=4)
    assert patch_index.tolist() == [[4 * 24 + 2, 23]]
    logits = torch.full((1, 2, 24 * 24), -100.0)
    logits.scatter_(2, patch_index.unsqueeze(-1), 100.0)
    patch_offsets = torch.zeros((1, 2, 24 * 24, 2))
    gather_index = patch_index.unsqueeze(-1).unsqueeze(-1).expand(1, 2, 1, 2)
    patch_offsets.scatter_(2, gather_index, offsets.unsqueeze(2))
    decoded = decode_cursor_heatmap(logits, patch_offsets, patch_size=4)
    assert torch.allclose(decoded, positions, atol=1e-6)
