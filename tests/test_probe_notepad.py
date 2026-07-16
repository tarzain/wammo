import torch

from wammo.eval.probe_notepad import fit_linear_probe, visible_delta_features


def test_fit_linear_probe_learns_simple_map():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.stack([x[:, 0] * 2, x[:, 1] - 1], dim=-1)
    _, metrics = fit_linear_probe(x, y, x, y, steps=80, lr=5e-2, device=torch.device("cpu"))
    assert metrics["mae_mean"] < 0.25


def test_visible_delta_features_pairs_adjacent_frames():
    features = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    deltas = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    x, y = visible_delta_features(features, deltas)
    assert x.shape == (3, 4)
    assert y.shape == (3, 2)
    torch.testing.assert_close(x[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(y[-1], torch.tensor([4.0, 5.0]))
