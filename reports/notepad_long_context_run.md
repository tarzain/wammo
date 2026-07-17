# NotePad Desk Long Context Run

Status: in progress as of step `85000 / 100000`.

Run directory: `runs/notepad-long-corner-sigma-context`

Checkpoint cadence: every 5000 steps, with the latest durable checkpoint at the time of this note:

- `runs/notepad-long-corner-sigma-context/checkpoint_step_85000.pt`

Final checkpoint path, once the trainer exits:

- `runs/notepad-long-corner-sigma-context/checkpoint.pt`

## Configuration

This is the corner-sigma hybrid run with previous-chunk context enabled:

```bash
python -m wammo.train.train_notepad_hybrid \
  --episodes 100000 \
  --eval-episodes 64 \
  --steps 100000 \
  --batch-size 4 \
  --device cuda \
  --out runs/notepad-long-corner-sigma-context \
  --head-sigma-conditioned \
  --sigma-corner-weight 0.5 \
  --sigma-corner-low 0.0 \
  --sigma-corner-high 1.0 \
  --delta-weight 4 \
  --delta-ce-weight 1 \
  --cursor-weight 1 \
  --context-chunks 1 \
  --log-every 1000 \
  --ladder-every 5000 \
  --checkpoint-every 5000
```

The important change from the aborted first long run is `--context-chunks 1`: current chunk tokens attend to one clean previous chunk. This is chunk-autoregressive training context, not KV-cache deployment inference.

Model size for this run is the current small trainer default:

| field | value |
| --- | ---: |
| d_model | 128 |
| layers | 4 |
| heads | 4 |
| patch size | 4x4 |
| tokens/frame | 576 |
| chunk frames | 4 |

## Training-log trajectory

The blended eval metrics continue improving through the run. Recent log values:

| step | eval loss | flow delta MAE px | CE delta MAE px | cursor pos MAE px | click acc | key acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50000 | 2.617 | 1.978 | 1.410 | 8.643 | 1.000 | 1.000 |
| 65000 | 2.513 | 1.803 | 1.442 | 8.871 | 1.000 | 1.000 |
| 75000 | 2.434 | 1.878 | 1.346 | 8.037 | 1.000 | 1.000 |
| 85000 | 2.345 | 2.010 | 1.369 | 8.017 | 1.000 | 1.000 |

Read: the discrete channels remain solved, video/action losses are stable, and CE delta prediction has stayed in the `~1.3-1.4 px` blended range late in training. Cursor localization is noisy but meaningfully better than the early `~25 px` state.

## Motion-only delta split

The blended delta metric includes idle frames, so checkpoint `65000` was rescored with the context-aware sigma stratifier at `sigma=1.0`.

Output:

- `runs/notepad-long-corner-sigma-context/analysis/sigma_stratification_step_65000_sigma1.json`

Motion-only results:

| eval condition | motion frames | motion rate | flow motion MAE px | CE motion MAE px | flow pred abs mean px | CE pred abs mean px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean video, noisy action | 2948 | 0.720 | 1.674 | 1.311 | 5.220 | 5.697 |
| noisy video, noisy action | 2948 | 0.720 | 2.107 | 1.776 | 5.214 | 5.886 |

Read: the blended metric was flattering, but not fake. The fully noisy equal-sigma corner remains harder than clean-video inverse dynamics, but CE motion MAE at `1.776 px` is now in the same range as the earlier pure-inverse coord+diff diagnostic (`1.857 px`) without integrating diff channels into the joint model.

This weakens the earlier "must add diff channels before the long run" conclusion. Training time alone solved a large part of the continuous-action shrinkage problem for this small model.

## Ladder trajectory

The context-aware chunk-local ladder is non-monotonic:

| step | cursor h4 | click h4 | key h4 |
| ---: | ---: | ---: | ---: |
| 5000 | 1.997e-3 | 3.166e-5 | 4.793e-5 |
| 10000 | 1.127e-3 | 1.910e-4 | 5.901e-5 |
| 15000 | 1.618e-3 | 1.105e-3 | 2.916e-4 |
| 20000 | 2.947e-3 | 2.535e-3 | 2.726e-4 |
| 35000 | 5.163e-4 | 3.316e-4 | 8.998e-5 |
| 50000 | 1.015e-3 | 2.244e-4 | 1.172e-4 |
| 65000 | 1.258e-3 | 3.091e-4 | 3.140e-4 |

Read: authority is wavy, not flat. The early spike at `20k`, mid-run dip, and partial recovery by `65k` are consistent with the "young model" interpretation. The right next read is the full post-run curve, not a new architecture fork mid-run.

## Autoregressive h8/h16 probe

Post-hoc autoregressive rollout scoring was added after the long run started:

- code: `src/wammo/eval/autoregressive_ladder.py`
- initial output: `runs/notepad-long-corner-sigma-context/analysis/autoregressive_ladder_step_65000_e16.json`

This evaluator rolls chunks forward by feeding each generated chunk back as the next chunk's clean context. It is not the same as KV-cache inference, but it measures real multi-chunk action authority instead of reusing chunk-local frame indices.

First 16-episode AR battery at checkpoint `65000`:

| channel | h4 | h8 | h16 |
| --- | ---: | ---: | ---: |
| cursor | 1.148e-3 | 2.059e-3 | 6.771e-4 |
| click | 3.743e-4 | 3.981e-3 | 2.119e-4 |
| key | 3.708e-4 | 1.799e-3 | 2.565e-4 |

Read: h8 is stronger than h4 in this first small battery for all three channels, while h16 drops. This is the first real far-horizon ladder measurement for the project and should be rerun across all saved checkpoints after the 100k run finishes.

## Current interpretation

The long run is materially changing the conclusion from the earlier short arms.

Earlier short-run evidence made the delta pathway look structurally broken: continuous heads were shrunk, CE heads improved inverse prediction but risked authority collapse, and diff channels looked necessary for motion-frame inverse dynamics. The 100k context-aware run says a large part of that was training maturity, not architecture.

What still looks unresolved:

- authority is non-monotonic and has not clearly consolidated;
- fully noisy video+action inverse dynamics is still worse than clean-video inverse dynamics;
- cursor localization is improved but still around `8 px`;
- h16 authority is weak in the first AR battery.

Decision: let the run finish, then run the full analysis battery on saved checkpoints before changing the architecture again. The next mandatory artifacts are:

1. full autoregressive h1/h2/h4/h8/h16 ladder over checkpoints;
2. final motion-only sigma stratification;
3. interactive rollout / hands-on demo;
4. persistence eval on typed-note drag/occlusion.
