import torch

from wammo.model.tokenizer import patchify, unpatchify


def test_patchify_round_trip_exact():
    frame = torch.arange(2 * 3 * 64 * 64, dtype=torch.float32).reshape(2, 3, 64, 64)
    patches = patchify(frame, patch_size=4)
    restored = unpatchify(patches, height=64, width=64, patch_size=4)
    torch.testing.assert_close(restored, frame)

