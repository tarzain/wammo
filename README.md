# Micro-WAM: Minimal World Action Model

`wammo` is a small, structurally faithful world-action-model experiment on a toy GUI domain. The goal is to verify the DreamZero/WAM mechanism at a scale where one GPU can train it and a browser can run it.

The important constraint is: simplify scale, not structure. Video tokens and action tokens are generated together by one chunk-autoregressive denoising transformer.

## Repository Layout

- `specs/cursor_world.json`: shared simulator constants for Python and TypeScript.
- `src/wammo/cursor_world`: deterministic Python simulator and scripted policies for dataset generation.
- `src/wammo/data`: episode generation and array serialization helpers.
- `src/wammo/model`: patch tokenizer, flow-matching utilities, and a DiT skeleton.
- `tests`: regression tests for simulator determinism, patch round trips, and flow direction.
- `web`: TypeScript/Canvas Cursor World demo using the same spec.

## Quick Start

```bash
cd /workspace/wammo
source /venv/main/bin/activate
uv pip install -e ".[dev]"
pytest
```

Generate a tiny dataset:

```bash
python -m wammo.data.generate --episodes 16 --out data/tiny.npz --seed 0
```

Run the browser simulator:

```bash
cd web
. /opt/nvm/nvm.sh
npm install
npm run dev -- --host 127.0.0.1
```

## Current Milestone

This repo starts at build-order step 1 plus the tokenizer and flow-convention tests:

1. Shared Cursor World spec.
2. Python simulator and dataset generator.
3. TypeScript/Canvas simulator shell.
4. Pixel patchifier with exact round-trip tests.
5. Flow matching convention locked to `x_t = (1 - t) * x0 + t * noise`, `v = noise - x0`.
6. DiT module skeleton for joint video/action chunk denoising.

The NotePad Desk v2 path uses mixed-objective joint modeling: video patches and cursor deltas are trained with flow matching, while discrete mouse/key channels use CE heads from the same joint action-token hidden states. This keeps symbol identity exact without moving button/key prediction into a separate inverse-dynamics model. The current divergence ladder is chunk-local (`h1`-`h4`); true h8/h16 authority requires the next autoregressive rollout evaluator.

Run the NotePad one-episode overfit with divergence ladder logging:

```bash
python -m wammo.train.overfit_notepad_one \
  --seed 0 \
  --steps 3000 \
  --batch-size 4 \
  --device cuda \
  --action-dropout 0 \
  --ladder-every 500 \
  --out runs/notepad-overfit-one
```

Run the first 1k-episode NotePad generalization pass:

```bash
python -m wammo.train.train_notepad \
  --episodes 1000 \
  --steps 3000 \
  --batch-size 4 \
  --device cuda \
  --action-dropout 0 \
  --ladder-every 500 \
  --out runs/notepad-1k
```
