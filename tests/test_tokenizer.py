import torch

from wammo.model.tokenizer import add_coordinate_channels, patchify, patchify_with_coords, unpatchify


def test_patchify_round_trip_exact():
    frame = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
    patches = patchify(frame, patch_size=4)
    restored = unpatchify(patches, height=64, width=64, patch_size=4)
    torch.testing.assert_close(restored, frame)


def test_patchify_with_coords_adds_two_channels():
    frames = torch.zeros((1, 4, 96, 96, 3))
    patches = patchify_with_coords(frames)
    assert patches.shape == (1, 4, 24 * 24, 5 * 4 * 4)
    with_coords = add_coordinate_channels(frames)
    assert with_coords.shape[-1] == 5
    assert float(with_coords[..., 3].min()) == -1.0
    assert float(with_coords[..., 3].max()) == 1.0
