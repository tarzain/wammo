import numpy as np
import torch

from wammo.model.dit import MicroWAMConfig
from wammo.train.train_notepad_hybrid import (
    NotePadHybridChunks,
    NotePadHybridModel,
    hybrid_training_step,
    training_changed_patch_mask,
    weighted_video_loss,
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


def test_hybrid_sampler_context_shapes():
    frames = np.zeros((2, 64, 96, 96, 3), dtype=np.uint8)
    frames[:, :, 10, 20] = [245, 245, 245]
    actions = np.zeros((2, 64, 4), dtype=np.float32)
    actions[1, 8, 0] = 4
    dataset = NotePadHybridChunks(frames, actions, motion_oversample=True)
    video, action, positions, chunk_ids, context_video, context_actions, context_ids = dataset.sample_with_context(
        2, torch.Generator().manual_seed(0), torch.device("cpu"), context_chunks=1
    )
    assert video.shape == (2, 4, 576, 48)
    assert action.shape == (2, 4, 4)
    assert positions.shape == (2, 4, 2)
    assert chunk_ids.eq(2).all()
    assert context_video.shape == (2, 1, 4, 576, 48)
    assert context_actions.shape == (2, 1, 4, 4)
    assert context_ids.eq(1).all()


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


def test_changed_patch_video_loss_upweights_changed_patches():
    video = torch.zeros(1, 4, 2, 3)
    target = torch.zeros_like(video)
    pred = torch.zeros_like(video)
    video[:, 1, 0] = 1
    target[:, 1, 0] = 1
    pred[:, 1, 0] = 3
    mask = training_changed_patch_mask(video, context_video=None, threshold=0.02)
    assert mask[0, 1, 0]
    weighted, metrics = weighted_video_loss(pred, target, video, None, changed_patch_weight=10)
    unweighted, _ = weighted_video_loss(pred, target, video, None, changed_patch_weight=0)
    assert weighted > unweighted
    assert metrics["changed_patch_rate"] > 0
