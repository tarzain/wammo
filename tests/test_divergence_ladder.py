import torch

from wammo.eval.divergence_ladder import action_variants, changed_patch_mask, notepad_divergence_ladder
from wammo.model.dit import MicroWAMConfig
from wammo.train.overfit_notepad_one import NotePadJointModel


def test_action_variants_are_channel_specific():
    actions = torch.zeros(2, 4, 4)
    pos, neg = action_variants(actions, "cursor", key_index=4)
    assert pos[..., 0].eq(1).all()
    assert neg[..., 0].eq(-1).all()
    pos, neg = action_variants(actions, "click", key_index=4)
    assert pos[..., 2].eq(1).all()
    assert neg.eq(0).all()
    pos, _ = action_variants(actions, "key", key_index=4)
    assert pos[..., 3].eq(4).all()


def test_changed_patch_mask_marks_changed_frames():
    video = torch.zeros(1, 4, 2, 3)
    video[:, 2, 1] = 1
    mask = changed_patch_mask(video, threshold=0.02)
    assert mask[0, 0].sum() == 0
    assert mask[0, 2, 1]


def test_notepad_ladder_outputs_expected_keys():
    model = NotePadJointModel(
        MicroWAMConfig(d_model=32, n_layers=1, n_heads=4, action_dim=4, patches_per_frame=4, max_chunks=1),
        key_count=18,
    )
    video = torch.randn(1, 4, 4, 48)
    actions = torch.zeros(1, 4, 4)
    chunk_ids = torch.zeros(1, dtype=torch.long)
    metrics = notepad_divergence_ladder(model, video, actions, chunk_ids, key_index=4, horizons=(1, 4))
    assert "ladder_cursor_h1" in metrics
    assert "ladder_click_changed_h4" in metrics
    assert "ladder_key_h4" in metrics


def test_notepad_ladder_rejects_out_of_chunk_horizons():
    model = NotePadJointModel(
        MicroWAMConfig(d_model=32, n_layers=1, n_heads=4, action_dim=4, patches_per_frame=4, max_chunks=1),
        key_count=18,
    )
    video = torch.randn(1, 4, 4, 48)
    actions = torch.zeros(1, 4, 4)
    chunk_ids = torch.zeros(1, dtype=torch.long)
    try:
        notepad_divergence_ladder(model, video, actions, chunk_ids, key_index=4, horizons=(8,))
    except ValueError as error:
        assert "outside chunk length" in str(error)
    else:
        raise AssertionError("expected out-of-chunk horizon to fail")
