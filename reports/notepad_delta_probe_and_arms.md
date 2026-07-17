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

## Hybrid synthesis arm

Run: `runs/notepad-1k-hybrid`

Configuration: motion oversampling, continuous cursor deltas kept in the flow-matching action stream, `delta_weight=4`, binned dx/dy CE heads on the same action hidden states, and an auxiliary cursor-position loss on video hidden states.

Mechanism check: the standalone binned CE arm replaced the continuous delta velocity head with CE-only dx/dy heads. Cursor delta tokens still entered the transformer as noisy inputs, but there was no flow-matching velocity target binding them into the joint denoising process. Its near-untrained cursor ladder is therefore best read as an early joint-vs-conditioning knockout: CE fixed inverse action prediction but removed the cursor channel from the authority-critical denoising objective.

| run | motion flow MAE px | margin vs zero px | eval CE MAE px | cursor h4 ladder | probe position MAE px | click/key |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `runs/notepad-1k-hybrid` | 4.611 | 0.630 | 3.628 | 1.093e-3 | 3.252 | 100% / 100% |

Read: the hybrid is a real improvement over motion oversampling on in-stream delta prediction and keeps much stronger authority than CE-only, but it does not clear the acceptance bar. Cursor authority remains below the delta-weight arm (`1.09e-3` vs `1.76e-3`), CE inverse prediction remains worse than the standalone CE arm, and the frozen probe regresses from `2.912 px` to `3.252 px`. The auxiliary cursor-position readout learned its own supervised task somewhat, but did not make the trunk linearly encode cursor position at the desired precision.

Next implication: do not launch 100k from this hybrid. The likely bottleneck is still spatial representation. The next cheap fork is either increasing cursor-position auxiliary weight substantially and probing again, or moving to 2x2 patches on a smaller model budget to test whether representation precision, not loss composition, is the hard blocker.

## Layer sweep and input-side screens

Runs:

- Layer sweeps: `runs/notepad-1k`, `runs/notepad-1k-hybrid`
- Short representation screens: `runs/notepad-screen-linear`, `runs/notepad-screen-coord`, `runs/notepad-screen-conv`

Aux head check: the hybrid cursor-position head is `Linear(d_model, 2)` plus sigmoid output squashing, not an MLP. The probe regression is therefore not explained by a high-capacity nonlinear aux head hiding geometry outside the trunk.

Layer sweep with 1000-step probes:

| checkpoint | best layer | best position MAE px | final-layer position MAE px |
| --- | ---: | ---: | ---: |
| `runs/notepad-1k` | 4 | 2.990 | 2.990 |
| `runs/notepad-1k-hybrid` | 4 | 3.329 | 3.329 |

Read: the hybrid aux loss did not move a better representation to an earlier layer. It made every layer worse or flat relative to the baseline, so the failure is not a probe-at-the-wrong-layer artifact.

Short 500-step representation screens, same 24x24 token grid:

| screen | best layer | best position MAE px | read |
| --- | ---: | ---: | --- |
| linear control | 4 | 3.427 | short-run baseline |
| coordinate channels | 4 | 2.950 | clear improvement over control, still above gate |
| conv stem | 3 | 3.573 | worse than control |

Read: coordinate channels are the only cheap input-side change that helped, and they helped a lot at the same token count. But they still did not approach the <=1 px probe gate in the short screen. The conv stem, as implemented here, is not worth promoting.

Next implication: promote coordinate channels only as part of the next representation test, not directly to 100k. The next gate should be either a longer coordinate-channel hybrid run or a 2x2 patch screen. If coordinate channels plateau near 3 px after a longer run, the 4x token 2x2 patch path becomes the right next experiment.
