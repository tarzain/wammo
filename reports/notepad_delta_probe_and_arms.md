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

## MLP probe audit

Runs:

- Standard checkpoints: `runs/notepad-1k`, `runs/notepad-1k-hybrid`
- Screen checkpoints: `runs/notepad-screen-linear`, `runs/notepad-screen-coord`, `runs/notepad-screen-conv`

Probe: one hidden layer MLP, GELU, hidden size 256, same train/eval episode split and 1000 probe steps as the layer-sweep linear probes.

| checkpoint | best linear layer | best linear position MAE px | best MLP layer | best MLP position MAE px |
| --- | ---: | ---: | ---: | ---: |
| `runs/notepad-1k` | 4 | 2.974 | 3 | 2.689 |
| `runs/notepad-1k-hybrid` | 3 | 3.327 | 4 | 3.025 |
| `runs/notepad-screen-linear` | 4 | 3.363 | 4 | 3.142 |
| `runs/notepad-screen-coord` | 4 | 2.864 | 4 | 2.672 |
| `runs/notepad-screen-conv` | 4 | 3.535 | 4 | 3.329 |

Read: the MLP probe improves position decoding by a few tenths of a pixel, but it does not reveal hidden ~1px position information. The best result is still `2.672 px` on the coord-channel screen, essentially tied with the baseline checkpoint's `2.689 px`. The linear probe was pessimistic, but not qualitatively wrong.

Decision: keep coord channels as a useful cheap improvement, but do not promote any current 4x4-patch configuration to 100k. The next representation test should be the sledgehammer screen: 2x2 patches, likely with a smaller width/depth budget to keep memory bounded. A separate cursor-size diagnostic is still useful: if a larger cursor solves the probe at 4x4 patches, it cleanly localizes the failure to sprite-scale-vs-patch-scale rather than the transformer.

## Probe label correction and cursor-size diagnostic

Important correction: the earlier position probes used `cursor_centroids(frames)` as labels. That pixel heuristic is badly contaminated by other near-white UI pixels such as toolbar glyphs and focus rings. Against exact cursor positions reconstructed from action integration, the centroid labels have very large error:

| cursor size | centroid-vs-true mean abs error px | p95 euclidean error px |
| ---: | ---: | ---: |
| 5 | 22.345 | 54.269 |
| 9 | 19.473 | 46.646 |

All future position probes and cursor auxiliary losses now use exact cursor positions from actions, not pixel centroids. This invalidates the absolute interpretation of the previous `~3 px` position-probe numbers. The delta probes and ladder metrics are unaffected.

True-label probes:

| run | pooling | best linear position MAE px | best MLP position MAE px | note |
| --- | --- | ---: | ---: | --- |
| `runs/notepad-1k` | mean | 12.703 | 11.384 | old checkpoint, no cursor aux |
| `runs/notepad-screen-coord-true` | mean | 12.917 | 11.695 | trained with exact cursor labels |
| `runs/notepad-screen-coord-cursor9-true` | mean | 12.860 | 11.818 | larger cursor does not help |
| `runs/notepad-screen-2x2-coord` | mean | 15.664 | 14.105 | reduced 2x2 screen, undertrained/small |
| `runs/notepad-screen-coord-true` | spatial moments | 9.455 | 7.326 | better probe pooling, still far from gate |

Read: making the cursor larger is possible, but in this diagnostic it does not solve the representation problem. The bigger finding is that mean-pooled probes were the wrong instrument for absolute position because they discard patch layout. Spatial-moment pooling is better and should replace mean pooling for future position probes, but even it does not reveal <=1 px cursor position in the current 4x4 coord model.

Decision update: the current evidence no longer supports a clean "3 px within-patch floor" story. The old 3 px floor was partly a bad label/probe artifact. With exact labels, the model is much worse at absolute cursor position. Next work should focus on a position-aware architecture or probe path, not only patch size: e.g. patch-level heatmap/objectness supervision, CLS/query token cross-attention to patch tokens, or a flattened/sparse patch-token probe that preserves layout more completely than spatial moments.

## CenterNet-style cursor localization screen

Change: the representation screen now trains a cursor auxiliary objective as patch classification over the 24x24 grid plus within-patch offset regression. The old normalized coordinate MSE head remains available, but the screen default is now heatmap+offset, and the acceptance metric is decoded cursor position error from the model head rather than a pooled linear probe.

Runs:

- `runs/notepad-screen-centernet-coord`
- `runs/notepad-screen-centernet-coord-cursor9`

Both runs use 1000 generated episodes, 500 train steps, coordinate-channel patchify, `cursor_weight=0`, `cursor_heatmap_weight=1`, and `cursor_offset_weight=1`.

| run | cursor size | held-out decoded MAE px | held-out decoded euclidean px | patch accuracy | best MLP probe MAE px |
| --- | ---: | ---: | ---: | ---: | ---: |
| `runs/notepad-screen-centernet-coord` | 5 | 19.010 | 30.033 | 0.333 | 8.907 |
| `runs/notepad-screen-centernet-coord-cursor9` | 9 | 19.017 | 30.041 | 0.332 | 6.163 |

Read: heatmap+offset learns above chance but does not clear the localization gate in this short screen. The larger cursor improves the auxiliary MLP probe substantially, so visibility matters, but it barely changes decoded localization from the model's own head. That makes "cursor is simply too small to see" an incomplete explanation. The remaining failure is likely that the trunk/head is learning coarse cursor priors and local visual cues without reliably selecting the exact patch on held-out rollouts.

Decision: do not promote this configuration to the 100k run. The next diagnostic should inspect heatmap predictions directly: compare predicted patch distributions against the cursor marginal and visualize a few failure frames. If the head is following the marginal path prior, rebalance the localization loss or train a cursor-only detector on clean frames first; if it is visually tracking but shifted, audit target/render alignment again.

Note: the original saved larger-cursor decoder metric was produced from an in-process trainer path that had not switched the transformer back to eval mode after training/probing. Recomputing the checkpoint with `model.eval()` gives the corrected row above. The screen probe and decoder helpers now set and restore eval mode explicitly.

## Sigma stratification

Runs:

- `runs/notepad-screen-centernet-coord/analysis/sigma_stratification.json`
- `runs/notepad-screen-centernet-coord-cursor9/analysis/sigma_stratification.json`

Question: are the cursor aux head and delta inverse heads failing because they read noised current-chunk tokens at high flow σ?

Default cursor-size checkpoint:

| slice | σ=0 | σ=0.5 | σ=1.0 |
| --- | ---: | ---: | ---: |
| cursor decoded MAE, video σ varied, clean action | 19.010 | 19.742 | 20.684 |
| cursor patch acc, video σ varied, clean action | 0.333 | 0.270 | 0.063 |
| CE delta motion MAE, action σ varied, clean video | 1.987 | 3.493 | 5.136 |
| CE delta motion MAE, video σ varied, noisy action | 5.136 | 5.138 | 5.256 |
| flow delta motion MAE, action σ varied, clean video | 7.522 | 6.584 | 5.216 |
| flow delta motion MAE, video σ varied, noisy action | 5.216 | 5.223 | 5.248 |

The larger-cursor checkpoint has the same qualitative curves: cursor localization is already poor at σ=0 and worsens with video σ; delta CE is strong only when the action token itself is near-clean, and becomes baseline-like when action σ approaches 1; video σ barely changes delta inverse performance when action tokens are fully noisy.

Read: the cleanest version of the hypothesis is not confirmed. The cursor head is not secretly near-ceiling at low σ; it is already bad on clean video. But the sigma cut does explain part of the delta story: the CE inverse head mostly exploits low-noise action-token self-information, not video-derived inverse dynamics. When action tokens are fully noisy, clean video does not rescue delta prediction. That is a more specific failure than "video noising erases the cursor": the current action-prediction path is not forced to compute displacement from video.

Decision update: the next intervention should not be another cursor-size or patch-size screen. First inspect heatmap predictions and delta CE predictions against marginals, then test a stricter inverse-dynamics setup where action tokens are masked/noised while video is clean and the head must read frame-to-frame displacement. The result also strengthens the case for decoupled σ schedules or explicit low-σ/action-mask training for action prediction, but only after confirming the heads are not simply learning marginal priors.
