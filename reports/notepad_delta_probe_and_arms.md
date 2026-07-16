# NotePad Delta Probe and 1k Arm Results

Date: 2026-07-16

## Frozen trunk probes

Run: `runs/notepad-1k`

The linear probes use frozen trunk activations with action tokens zeroed, so they test whether video-derived hidden states linearly expose cursor geometry.

| probe | mean MAE px | euclidean MAE px |
| --- | ---: | ---: |
| cursor position | 2.912 | 4.554 |
| current-frame delta | 3.476 | 5.369 |
| visible-transition delta | 3.250 | 5.040 |

Interpretation: cursor position is not linearly recoverable to the desired ~1 px precision. The delta failure is therefore not just a bad final scalar head; the trunk representation is already coarse. Still, objective changes can improve action prediction, so the arm race is informative.

## 1k arms

All runs use 1000 generated episodes, 3000 train steps, batch size 4, `action_dropout=0`, and the same 64-episode analysis battery. The motion-frame zero baseline is `5.241 px`.

| run | motion delta MAE px | margin vs zero px | pred abs mean px | cursor h4 ladder | click acc | key acc | key-event acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/notepad-1k` | 5.146 | 0.095 | 3.261 | 4.413e-4 | 1.000 | 1.000 | 1.000 |
| `runs/notepad-1k-delta-weight` | 5.006 | 0.235 | 3.137 | 1.762e-3 | 1.000 | 1.000 | 1.000 |
| `runs/notepad-1k-motion-oversample` | 4.760 | 0.482 | 2.764 | 5.132e-4 | 1.000 | 1.000 | 1.000 |
| `runs/notepad-1k-binned-delta-ce` | 4.456 | 0.785 | 4.149 | 2.693e-6 | 1.000 | 1.000 | 1.000 |

## Read

Binned CE is the best inverse-action predictor and escapes much of the regression shrinkage, but it nearly erases cursor video authority in the chunk-local ladder. Delta weighting gives the strongest video authority and keeps all discrete channels intact, but only modestly improves motion-frame delta prediction. Motion oversampling is the most balanced continuous arm: better delta prediction than weighting and a nonzero ladder close to the baseline.

The current evidence does not support launching the full 100k run unchanged. The next targeted experiment should combine the useful parts: motion oversampling plus stronger cursor supervision, while separately testing whether smaller patches or an auxiliary cursor-position loss can repair the trunk representation exposed by the probe.
