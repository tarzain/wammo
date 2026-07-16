import torch

from wammo.train.overfit_notepad_one import denormalize_notepad_actions, normalize_notepad_actions


def test_notepad_action_normalization_round_trip():
    raw = torch.tensor([[-8.0, 4.0, 0.0, 0.0], [8.0, -4.0, 1.0, 17.0]])
    normalized = normalize_notepad_actions(raw, max_delta=8.0, key_count=18)
    restored = denormalize_notepad_actions(normalized, max_delta=8.0, key_count=18)
    torch.testing.assert_close(restored, raw)

