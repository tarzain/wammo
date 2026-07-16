import numpy as np
import torch

from wammo.eval.analyze_notepad_run import (
    cursor_centroids,
    delta_baselines,
    delta_by_chunk_position,
    delta_prediction_diagnostic,
    summarize_samples,
)


def test_delta_baselines_zero_and_mean():
    actions = np.zeros((1, 2, 4), dtype=np.float32)
    actions[0, 0, 0:2] = [2, -2]
    actions[0, 1, 0:2] = [4, 0]
    baselines = delta_baselines(actions)
    assert baselines["zero_delta_mae_px"] == 2.0
    assert baselines["mean_delta"] == [3.0, -1.0]
    assert baselines["mean_delta_mae_px"] == 1.0


def test_summarize_samples():
    summary = summarize_samples({"x": torch.tensor([1.0, 3.0])})
    assert summary["x"]["mean"] == 2.0
    assert summary["x"]["n"] == 2


def test_delta_prediction_diagnostic_motion_split():
    true_actions = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
    pred_delta = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    diagnostic = delta_prediction_diagnostic(true_actions, pred_delta, max_delta=8.0)
    assert diagnostic["motion_frames"] == 1
    assert diagnostic["model_motion_delta_mae_px"] == 4.0
    assert diagnostic["model_motion_pred_near_zero_rate"] == 1.0


def test_delta_by_chunk_position():
    true_actions = torch.zeros(2, 4, 4)
    true_actions[:, 3, 0] = 1.0
    pred_delta = torch.zeros(2, 4, 2)
    diagnostic = delta_by_chunk_position(true_actions, pred_delta, max_delta=8.0)
    assert diagnostic["pos_1"]["motion_frames"] == 0
    assert diagnostic["pos_4"]["motion_frames"] == 2
    assert diagnostic["pos_4"]["model_delta_mae_px"] == 4.0


def test_cursor_centroids_extracts_cursor_pixels():
    frames = np.zeros((1, 1, 96, 96, 3), dtype=np.uint8)
    frames[0, 0, 10, 20] = [245, 245, 245]
    positions = cursor_centroids(frames)
    np.testing.assert_allclose(positions[0, 0], [20, 10])
