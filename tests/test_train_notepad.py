import numpy as np
import torch

from wammo.train.train_notepad import NotePadMultiEpisodeChunks, generate_training_dataset, make_eval_dataset


def test_multi_episode_sampler_shapes():
    frames = np.zeros((3, 64, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((3, 64, 4), dtype=np.float32)
    dataset = NotePadMultiEpisodeChunks(frames, actions)
    video, action, chunk_ids = dataset.sample(2, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert video.shape == (2, 4, 576, 48)
    assert action.shape == (2, 4, 4)
    assert chunk_ids.shape == (2,)


def test_motion_oversample_sampler_uses_motion_chunks():
    frames = np.zeros((2, 64, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((2, 64, 4), dtype=np.float32)
    actions[1, 8, 0] = 4
    dataset = NotePadMultiEpisodeChunks(frames, actions, motion_oversample=True)
    assert dataset.motion_pairs.tolist() == [[1, 2]]
    _, sampled_actions, chunk_ids = dataset.sample(3, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert chunk_ids.eq(2).all()
    assert sampled_actions[:, 0, 0].eq(0.5).all()


def test_eval_dataset_shapes():
    dataset, metadata = make_eval_dataset(123)
    video, action, chunk_ids = dataset.all_chunks(torch.device("cpu"))
    assert video.shape == (16, 4, 576, 48)
    assert action.shape == (16, 4, 4)
    assert chunk_ids.shape == (16,)
    assert metadata["eval_rare_event_rate"] >= 0.15


def test_generate_training_dataset_cursor_size_override():
    frames, actions, metadata = generate_training_dataset(1, 123, progress_every=0, cursor_size=9)
    assert frames.shape == (1, 64, 96, 96, 3)
    assert actions.shape == (1, 64, 4)
    assert metadata["cursor_size"] == 9
