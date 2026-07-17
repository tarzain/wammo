# ONNX Export Status

Status: exporter implemented and validated on checkpoint `85000`.

Code:

- `src/wammo/export/onnx_notepad_hybrid.py`

The exporter converts NotePad hybrid checkpoints to ONNX and immediately validates the exported graph against PyTorch with ONNX Runtime CPU execution.

## Dependencies

Install the optional export dependencies:

```bash
source /venv/main/bin/activate
uv pip install -e ".[onnx]"
```

The `onnx` extra currently installs:

- `onnx`
- `onnxruntime`
- `onnxscript`

## Validated Artifact

Source checkpoint:

- `runs/notepad-long-corner-sigma-context/checkpoint_step_85000.pt`

Generated local artifact:

- `runs/notepad-long-corner-sigma-context/onnx/checkpoint_step_85000.onnx`
- `runs/notepad-long-corner-sigma-context/onnx/checkpoint_step_85000.json`

The generated artifacts are under `runs/`, so they are intentionally ignored by git. The tracked repo contains the exporter and this reproduction note, not the binary checkpoint exports.

Export command:

```bash
python -m wammo.export.onnx_notepad_hybrid \
  --run runs/notepad-long-corner-sigma-context \
  --checkpoint runs/notepad-long-corner-sigma-context/checkpoint_step_85000.pt \
  --out runs/notepad-long-corner-sigma-context/onnx/checkpoint_step_85000.onnx
```

Validation:

| metric | value |
| --- | ---: |
| ONNX opset | 18 |
| max abs diff vs PyTorch | 1.144e-5 |
| mean abs diff vs PyTorch | 1.142e-6 |

The exported model is a single `.onnx` file. Earlier test exports briefly produced external `.onnx.data` weights, but the exporter now sets `external_data=False` and removes stale sidecar data before export.

## Model Contract

Inputs:

| name | dtype | shape |
| --- | --- | --- |
| `video_patches` | float32 | `[batch, 4, 576, 48]` |
| `delta_actions` | float32 | `[batch, 4, 2]` |
| `button_ids` | int64 | `[batch, 4]` |
| `key_ids` | int64 | `[batch, 4]` |
| `sigma_video` | float32 | `[batch]` |
| `sigma_action` | float32 | `[batch]` |
| `chunk_ids` | int64 | `[batch]` |
| `context_video_patches` | float32 | `[batch, 1, 4, 576, 48]` |
| `context_actions` | float32 | `[batch, 1, 4, 4]` |
| `context_chunk_ids` | int64 | `[batch, 1]` |

Outputs:

| name | dtype | shape |
| --- | --- | --- |
| `video_velocity` | float32 | `[batch, 4, 576, 48]` |
| `delta_velocity` | float32 | `[batch, 4, 2]` |
| `dx_logits` | float32 | `[batch, 4, 17]` |
| `dy_logits` | float32 | `[batch, 4, 17]` |
| `button_logits` | float32 | `[batch, 4, 2]` |
| `key_logits` | float32 | `[batch, 4, 18]` |
| `cursor_xy_norm` | float32 | `[batch, 4, 2]` |

The graph takes normalized, already-patchified tensors. Pixel normalization, patchify/unpatchify, action denormalization, and rollout policy remain host-side responsibilities.

## Bulk Export

After the long run finishes, export every saved step checkpoint:

```bash
python -m wammo.export.onnx_notepad_hybrid \
  --run runs/notepad-long-corner-sigma-context \
  --checkpoint-glob "checkpoint_step_*.pt"
```

The exporter writes each ONNX file to:

- `runs/notepad-long-corner-sigma-context/onnx/<checkpoint-stem>.onnx`

and a JSON metadata sidecar next to it.

## Current Limitations

- This export is an eager full-context graph, not a KV-cache graph.
- The model accepts one previous context chunk because the current long-run config uses `--context-chunks 1`.
- The graph covers model inference only; browser/runtime code still needs patchification, action clamping, denoising loop control, and generated-context rollover.
- ONNX Runtime validation is CPU-only for now. The same graph should be usable by other providers, but WebGPU or platform-specific performance has not been measured yet.
