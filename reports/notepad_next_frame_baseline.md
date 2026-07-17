# NotePad Desk Next-Frame Baseline

Date: 2026-07-17

## Purpose

The WAM checkpoint had good inverse/action metrics but weak interactive authority. This baseline asks a simpler question:

```text
frame[t-1] + action[t] -> frame[t]
```

No diffusion, no joint action denoising, no sigma schedules. The model is a small action-conditioned CNN. The goal is to establish a behavioral floor for NotePad Desk before further WAM surgery.

## Code

- Trainer: `src/wammo/train/train_notepad_next_frame_baseline.py`
- Rollout visualizer: `src/wammo/eval/next_frame_baseline_rollouts.py`
- Tests: `tests/test_train_notepad_next_frame_baseline.py`

Implemented variants:

- Direct full-frame prediction.
- Residual prediction initialized as exact copy.
- Short autoregressive rollout training with `--rollout-steps`.
- Consequential-window rollout oversampling with `--event-oversample-prob`.

## Runs

### Direct full-frame, one-step

Run: `runs/notepad-next-frame-baseline-10k`

Stopped at step 2500 after visual inspection.

Result:

- Scalar loss improved quickly.
- Autoregressive rollouts were unstable and over-responsive.
- Fixed rows showed cursor/right and click creating large blobs.
- Conclusion: action authority exists, but full-frame one-step prediction is not a usable world-model baseline.

Artifacts:

- `runs/notepad-next-frame-baseline-10k/checkpoint_step_2500.pt`
- `runs/notepad-next-frame-baseline-10k/rollouts/step_2500/fixed_action_rows.png`
- `runs/notepad-next-frame-baseline-10k/rollouts/step_2500/scripted_sequence.png`

### Residual, one-step

Run: `runs/notepad-next-frame-residual-1k`

Stopped after step 1000 visual inspection.

Result:

- Starts as an exact copy model, as intended.
- Still over-amplifies actions over autoregressive rollout.
- Better than direct full-frame initialization, but still not a usable floor.

Artifacts:

- `runs/notepad-next-frame-residual-1k/checkpoint_step_1000.pt`
- `runs/notepad-next-frame-residual-1k/rollouts/step_1000/fixed_action_rows.png`
- `runs/notepad-next-frame-residual-1k/rollouts/step_1000/scripted_sequence.png`

### Residual, rollout-4 training

Run: `runs/notepad-next-frame-rollout4-1k`

Command:

```bash
/venv/main/bin/python -u -m wammo.train.train_notepad_next_frame_baseline \
  --episodes 1000 \
  --steps 5000 \
  --batch-size 64 \
  --device cuda \
  --out runs/notepad-next-frame-rollout4-1k \
  --log-every 250 \
  --checkpoint-every 1000 \
  --ladder-every 1000 \
  --eval-episodes 16 \
  --hidden-channels 64 \
  --blocks 4 \
  --predict-residual \
  --rollout-steps 4 \
  --changed-pixel-weight 64 \
  --generate-progress-every 100
```

Final scalar metrics:

- Eval loss: `0.1406`
- Eval MAE: `0.0123`
- Changed-pixel MAE: `0.2327`
- Unchanged-pixel MAE: `0.0073`
- Changed-pixel rate: `0.0221`

Calibrated authority ratios at step 5000:

| Channel | h1 | h4 | h8 | h16 |
| --- | ---: | ---: | ---: | ---: |
| cursor | 0.457 | 0.579 | 0.786 | 1.095 |
| click | 1.947 | 2.379 | 2.723 | 3.406 |
| key | 0.109 | 0.065 | 0.092 | 0.470 |

Progression:

| Step | cursor h4 | click h4 | key h4 | cursor h16 | click h16 | key h16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.147 | 0.585 | 0.062 | 0.321 | 1.507 | 0.061 |
| 2000 | 0.630 | 1.401 | 0.138 | 1.050 | 2.874 | 0.175 |
| 3000 | 0.538 | 1.693 | 0.114 | 1.079 | 2.553 | 0.279 |
| 4000 | 0.565 | 2.435 | 0.126 | 1.605 | 3.530 | 0.383 |
| 5000 | 0.579 | 2.379 | 0.065 | 1.095 | 3.406 | 0.470 |

Visual result:

- Stable compared with one-step variants.
- Cursor movement is present but small.
- Click has strong authority but smears into note/toolbar artifacts.
- Key effects are still weak at short horizons.
- Not yet hands-on playable.

Artifacts:

- `runs/notepad-next-frame-rollout4-1k/checkpoint_step_5000.pt`
- `runs/notepad-next-frame-rollout4-1k/analysis/calibrated_ladder_step_5000.json`
- `runs/notepad-next-frame-rollout4-1k/rollouts/step_5000/fixed_action_rows.png`
- `runs/notepad-next-frame-rollout4-1k/rollouts/step_5000/scripted_sequence.png`

## Interpretation

The baseline is informative but not solved.

The direct one-step CNN proves that simple action conditioning can strongly affect pixels. Its failure mode is too much authority and poor autoregressive stability.

The rollout-trained residual CNN proves that short rollout training fixes the worst drift. Its failure mode is conservative dynamics and weak sparse-event fidelity, especially typing.

Compared with WAM:

- WAM failure: too little action authority in video generation.
- One-step CNN failure: too much ungrounded authority.
- Rollout CNN failure: stable but still low-fidelity and weak on keys.

This makes the next baseline lever clear: keep rollout training, but strengthen sparse event supervision. Candidate next runs:

1. Rollout-4 residual with higher changed-pixel weight, e.g. `128` or `256`.
2. Oversample sequences containing click/key events rather than only random starts.
3. Add a small object/event auxiliary head for note creation and typed glyph pixels.

The useful floor is now defined: a baseline must beat rollout4 stability while increasing short-horizon cursor/key authority without returning to one-step blob behavior.

### Residual, rollout-4, event oversampling + stronger changed-pixel weight

Run: `runs/notepad-next-frame-rollout4-event128-1k`

Command:

```bash
/venv/main/bin/python -u -m wammo.train.train_notepad_next_frame_baseline \
  --episodes 1000 \
  --steps 5000 \
  --batch-size 64 \
  --device cuda \
  --out runs/notepad-next-frame-rollout4-event128-1k \
  --log-every 250 \
  --checkpoint-every 1000 \
  --ladder-every 1000 \
  --eval-episodes 16 \
  --hidden-channels 64 \
  --blocks 4 \
  --predict-residual \
  --rollout-steps 4 \
  --event-oversample-prob 0.75 \
  --changed-pixel-weight 128 \
  --generate-progress-every 100
```

Dataset window counts:

- Rollout starts: `60000`
- Consequential rollout starts: `49011`

Final scalar metrics:

- Eval loss: `0.1449`
- Eval MAE: `0.0206`
- Changed-pixel MAE: `0.1891`
- Unchanged-pixel MAE: `0.0168`
- Changed-pixel rate: `0.0221`

Final calibrated authority ratios:

| Channel | h1 | h4 | h8 | h16 |
| --- | ---: | ---: | ---: | ---: |
| cursor | 0.505 | 0.659 | 0.904 | 1.287 |
| click | 4.015 | 3.790 | 3.713 | 4.041 |
| key | 0.692 | 0.222 | 0.174 | 0.266 |

Comparison against rollout-4 baseline at step 5000:

| Metric | rollout4 | event128 |
| --- | ---: | ---: |
| Eval MAE | 0.0123 | 0.0206 |
| Changed-pixel MAE | 0.2327 | 0.1891 |
| Unchanged-pixel MAE | 0.0073 | 0.0168 |
| cursor h4 | 0.579 | 0.659 |
| click h4 | 2.379 | 3.790 |
| key h4 | 0.065 | 0.222 |
| key h16 | 0.470 | 0.266 |

Interpretation:

- This arm improves changed-pixel MAE and short-horizon key authority.
- It worsens unchanged-pixel fidelity and overdrives click more strongly.
- Visuals show a stronger, cleaner note-like block after mouse-down, but typing still does not produce a real glyph.
- It is still not hands-on playable.

Artifacts:

- `runs/notepad-next-frame-rollout4-event128-1k/checkpoint_step_5000.pt`
- `runs/notepad-next-frame-rollout4-event128-1k/analysis/calibrated_ladder_step_5000.json`
- `runs/notepad-next-frame-rollout4-event128-1k/rollouts/step_5000/fixed_action_rows.png`
- `runs/notepad-next-frame-rollout4-event128-1k/rollouts/step_5000/scripted_sequence.png`

Updated conclusion:

Event oversampling plus higher changed-pixel weight is directionally useful for sparse consequences, but it is not sufficient. The baseline now needs structure, not just more weighting: either separate residual heads for cursor/note/glyph layers, or an auxiliary parser/object-state loss that directly supervises note creation and typed glyph persistence.
