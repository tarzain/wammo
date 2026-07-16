import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_binned_delta import (
    NotePadBinnedDeltaModel,
    bins_to_delta_norm,
    binned_training_step,
    delta_to_bins,
)


def test_delta_bins_round_trip_integer_pixels():
    delta = torch.tensor([[[-1.0, -0.5], [0.0, 0.5], [1.0, 0.125]]])
    bins = delta_to_bins(delta)
    restored = bins_to_delta_norm(bins)
    torch.testing.assert_close(restored, delta)


def test_binned_training_step_is_finite():
    model = NotePadBinnedDeltaModel(
        MicroWAMConfig(d_model=32, n_layers=1, n_heads=4, patches_per_frame=576),
        key_count=18,
    )
    video = torch.randn(1, 4, 576, 48)
    actions = torch.zeros(1, 4, 4)
    chunk_ids = torch.zeros((1,), dtype=torch.long)
    loss, metrics = binned_training_step(model, video, actions, chunk_ids, torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert metrics["delta_loss"] > 0
