# Final Authority Calibration

Run: `runs/notepad-long-corner-sigma-context`

Final checkpoint:

- `runs/notepad-long-corner-sigma-context/checkpoint.pt`
- `runs/notepad-long-corner-sigma-context/checkpoint_step_100000.pt`

The interactive/non-interactive rollouts showed the model barely responds to clamped actions. This report calibrates the ladder against the real simulator and tests action guidance on the final checkpoint.

## Final Scalar Metrics

The final training metrics still look good in isolation:

| metric | value |
| --- | ---: |
| eval loss | 2.299 |
| video MAE | 0.181 |
| flow delta MAE px | 1.806 |
| CE delta MAE px | 1.285 |
| cursor pos MAE px | 7.475 |
| click accuracy | 1.000 |
| key accuracy | 1.000 |
| key-event accuracy | 1.000 |

But these numbers do not imply controllable video generation. They mostly say the model renders the desktop distribution and predicts action labels well.

## Simulator-Calibrated Ladder

Calibration output:

- `runs/notepad-long-corner-sigma-context/analysis/calibrated_ladder_final.json`

The simulator calibration branches real `NotePadDesk` states under the same opposite-action protocol and measures the true pixel/patch divergence. The calibrated ratio is:

```text
model action divergence / simulator action divergence
```

So `1.0` means simulator-level responsiveness and values near `0` mean action conditioning barely changes generated video.

Final checkpoint ratios:

| channel | h1 | h2 | h4 | h8 | h16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cursor | 0.0265 | 0.0221 | 0.0321 | 0.1754 | 0.3432 |
| click | 0.0102 | 0.0101 | 0.0097 | 0.1259 | 0.0075 |
| key | 0.0506 | 0.0251 | 0.0122 | 0.0246 | 0.0033 |

Read: the checkpoint is not a convincingly controllable world model at normal sampling. The chunk-local h4 authority that looked nonzero in raw ladder units is only about `1-3%` of the simulator response for the key action channels. The h8/h16 cursor ratios are larger, but behaviorally the near-window response is what the user feels in an interactive loop, and that response is weak.

## Teacher-Forcing Gap

The final training summary's clean-context chunk-local cursor h4 ladder is:

- clean previous context h4: `7.066e-4`

The final autoregressive generated-context cursor h4 ladder is:

- generated previous context h4: `3.114e-4`

So self-generated context roughly halves near-window cursor authority. This is real exposure bias, but it is not the root cause. Even the clean-context h4 value is only about `7%` of simulator cursor h4:

```text
7.066e-4 / 9.712e-3 = 0.073
```

The bigger problem is that the video objective underpaid action-consequential pixels during training.

## Action Guidance Probe

Guidance output:

- `runs/notepad-long-corner-sigma-context/analysis/action_guidance_final.json`

Probe definition:

```text
guided_video_velocity = null_velocity + w * (action_conditioned_velocity - null_velocity)
```

This is MIRA-style CFG over action-conditioned vs null-action video denoising. It is a checkpoint-only probe; no retraining.

AR h4 ladder by guidance weight:

| channel | simulator h4 | w=1 h4 | w=3 h4 | w=8 h4 |
| --- | ---: | ---: | ---: | ---: |
| cursor | 9.712e-3 | 3.114e-4 | 2.802e-3 | 1.993e-2 |
| click | 2.515e-2 | 2.447e-4 | 2.202e-3 | 1.566e-2 |
| key | 6.991e-3 | 8.562e-5 | 7.706e-4 | 5.480e-3 |

Guided h4 ratio to simulator:

| channel | w=1 | w=3 | w=8 |
| --- | ---: | ---: | ---: |
| cursor | 0.032 | 0.289 | 2.052 |
| click | 0.010 | 0.088 | 0.623 |
| key | 0.012 | 0.110 | 0.784 |

Read: guidance strongly amplifies action authority. The capability is not absent; it is badly under-expressed by the default denoising loop. A guided interactive demo is worth trying immediately. If high guidance makes the cursor visibly responsive but distorted, that supports the retrain diagnosis: use changed-patch/action-consequence weighting to make action-conditioned pixels expensive during training instead of relying on inference-time extrapolation.

## Consequence

The standing ladder must be reported as a calibrated ratio from now on. Raw MSE ladder values are not meaningful without the simulator denominator.

Next retrain config should keep the corner-sigma hybrid path but add action-consequence video weighting:

- changed-patch video loss weighting;
- optionally idle-counterfactual/action-difference patch weighting;
- calibrated ladder logged from step one;
- hands-on checkpoint check early in training, not after a full run.

For the current checkpoint, the immediate next experiment is a guided interactive rollout with `w = 3-8`.
