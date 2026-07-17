import numpy as np
import torch

from wammo.train.train_notepad_next_frame_baseline import (
    NextFrameBaselineConfig,
    NotePadFramePairs,
    NotePadNextFrameCNN,
    action_to_planes,
    baseline_step,
    rollout_training_step,
)


def test_next_frame_pair_sampler_uses_previous_frame_and_current_action():
    frames = np.zeros((1, 4, 96, 96, 3), dtype=np.uint8)
    frames[0, 1, :, :] = 32
    frames[0, 2, :, :] = 64
    actions = np.zeros((1, 4, 4), dtype=np.float32)
    actions[0, 1, 0] = 8
    actions[0, 2, 3] = 5
    dataset = NotePadFramePairs(frames, actions, motion_oversample=False)

    inputs, sampled_actions, targets = dataset.all_pairs(torch.device("cpu"))

    assert inputs.shape == (3, 3, 96, 96)
    assert targets.shape == (3, 3, 96, 96)
    assert torch.allclose(inputs[0], torch.full_like(inputs[0], -1.0))
    assert torch.allclose(targets[0], torch.full_like(targets[0], 32 / 127.5 - 1.0))
    assert sampled_actions[0, 0].item() == 1.0
    assert sampled_actions[1, 3].item() == 5.0


def test_action_to_planes_encodes_key_and_delta():
    actions = torch.tensor([[0.5, -0.25, 1.0, 3.0]])
    planes = action_to_planes(actions, (2, 3), key_count=6)

    assert planes.shape == (1, 9, 2, 3)
    assert torch.all(planes[:, 0] == 0.5)
    assert torch.all(planes[:, 1] == -0.25)
    assert torch.all(planes[:, 2] == 1.0)
    assert torch.all(planes[:, 3 + 3] == 1.0)
    assert planes[:, 3:].sum().item() == 6.0


def test_baseline_step_is_finite_and_reports_changed_pixels():
    model = NotePadNextFrameCNN(NextFrameBaselineConfig(hidden_channels=16, blocks=1, key_count=18))
    input_frame = torch.zeros(2, 3, 96, 96)
    target = input_frame.clone()
    target[:, :, 10:14, 10:14] = 1
    actions = torch.zeros(2, 4)

    loss, metrics = baseline_step(model, input_frame, actions, target, changed_weight=10)

    assert torch.isfinite(loss)
    assert metrics["changed_pixel_rate"] > 0
    assert metrics["changed_mae"] > 0


def test_residual_baseline_starts_as_copy_model():
    model = NotePadNextFrameCNN(NextFrameBaselineConfig(hidden_channels=16, blocks=1, key_count=18, predict_residual=True))
    frames = torch.rand(2, 3, 96, 96).mul(2).sub(1)
    actions = torch.zeros(2, 4)

    pred = model(frames, actions)

    assert torch.allclose(pred, frames)


def test_rollout_sampler_aligns_actions_and_targets():
    frames = np.zeros((1, 5, 96, 96, 3), dtype=np.uint8)
    for t in range(5):
        frames[0, t, :, :] = t * 10
    actions = np.zeros((1, 5, 4), dtype=np.float32)
    for t in range(5):
        actions[0, t, 0] = t
    dataset = NotePadFramePairs(frames, actions, motion_oversample=False)

    inputs, action_chunks, targets = dataset.sample_rollout(1, 2, torch.Generator().manual_seed(0), torch.device("cpu"))

    assert inputs.shape == (1, 3, 96, 96)
    assert action_chunks.shape == (1, 2, 4)
    assert targets.shape == (1, 2, 3, 96, 96)
    start_value = round(float((inputs[0, 0, 0, 0] + 1) * 127.5 / 10))
    assert action_chunks[0, 0, 0].item() == (start_value + 1) / 8
    expected_target = ((start_value + 1) * 10) / 127.5 - 1
    assert torch.allclose(targets[0, 0], torch.full_like(targets[0, 0], expected_target))


def test_rollout_training_step_is_finite():
    model = NotePadNextFrameCNN(NextFrameBaselineConfig(hidden_channels=16, blocks=1, key_count=18, predict_residual=True))
    input_frame = torch.zeros(2, 3, 96, 96)
    targets = torch.zeros(2, 3, 3, 96, 96)
    targets[:, :, :, 10:14, 10:14] = 1
    actions = torch.zeros(2, 3, 4)

    loss, metrics = rollout_training_step(model, input_frame, actions, targets, changed_weight=10)

    assert torch.isfinite(loss)
    assert metrics["changed_pixel_rate"] > 0
