import torch

from wammo.train.overfit_one import (
    OneEpisodeChunks,
    denormalize_actions,
    normalize_actions,
    normalize_frames,
    training_step,
)
from wammo.model.dit import MicroWAMConfig, MicroWAMDiT


def test_action_normalization_round_trip():
    raw = torch.tensor([[-8.0, 4.0, 0.0], [8.0, -4.0, 1.0]])
    normalized = normalize_actions(raw, max_delta=8.0)
    restored = denormalize_actions(normalized, max_delta=8.0)
    torch.testing.assert_close(restored, raw)


def test_chunk_batch_shapes():
    frames = normalize_frames(torch.zeros((64, 64, 64, 3), dtype=torch.uint8).numpy())
    actions = normalize_actions(torch.zeros((64, 3)), max_delta=8.0)
    dataset = OneEpisodeChunks(frames, actions)
    video, action, chunk_ids = dataset.sample(2, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert video.shape == (2, 4, 256, 48)
    assert action.shape == (2, 4, 3)
    assert chunk_ids.shape == (2,)


def test_training_step_is_finite():
    model = MicroWAMDiT(MicroWAMConfig(d_model=32, n_layers=1, n_heads=4))
    video = torch.randn(1, 4, 256, 48)
    actions = torch.randn(1, 4, 3)
    chunk_ids = torch.zeros((1,), dtype=torch.long)
    loss, metrics = training_step(model, video, actions, chunk_ids, torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert metrics["loss"] > 0
