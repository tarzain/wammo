from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from wammo.model.dit import MicroWAMConfig
from wammo.notepad_desk import load_spec
from wammo.train.train_notepad_hybrid import NotePadHybridModel


INPUT_NAMES = [
    "video_patches",
    "delta_actions",
    "button_ids",
    "key_ids",
    "sigma_video",
    "sigma_action",
    "chunk_ids",
    "context_video_patches",
    "context_actions",
    "context_chunk_ids",
]

OUTPUT_NAMES = [
    "video_velocity",
    "delta_velocity",
    "dx_logits",
    "dy_logits",
    "button_logits",
    "key_logits",
    "cursor_xy_norm",
]


class NotePadHybridOnnxWrapper(nn.Module):
    def __init__(self, model: NotePadHybridModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        context_video_patches: torch.Tensor,
        context_actions: torch.Tensor,
        context_chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_all(
            video_patches,
            delta_actions,
            button_ids,
            key_ids,
            sigma_video,
            sigma_action,
            chunk_ids,
            context_video_patches=context_video_patches,
            context_actions=context_actions,
            context_chunk_ids=context_chunk_ids,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a NotePad hybrid checkpoint to ONNX.")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing config.json.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to RUN/checkpoint.pt.")
    parser.add_argument(
        "--checkpoint-glob",
        default=None,
        help="Glob relative to RUN for exporting multiple checkpoints, e.g. 'checkpoint_step_*.pt'.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output ONNX path.")
    parser.add_argument("--metadata-out", type=Path, default=None, help="Output JSON metadata path.")
    parser.add_argument("--context-chunks", type=int, default=None, help="Override context chunk count.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for tracing.")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-validate", action="store_true", help="Skip ONNX Runtime parity check.")
    return parser.parse_args()


def load_hybrid_model(run: Path, checkpoint_path: Path, device: torch.device) -> tuple[NotePadHybridModel, dict[str, Any], dict[str, Any]]:
    config_payload = json.loads((run / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MicroWAMConfig(**checkpoint["config"])
    run_args = config_payload.get("args", {})
    if not isinstance(run_args, dict):
        run_args = {}
    model_kind = checkpoint.get("model_kind", config_payload.get("model_kind", "notepad_hybrid"))
    if model_kind != "notepad_hybrid":
        raise ValueError(f"expected a notepad_hybrid checkpoint, got {model_kind!r}")
    head_sigma_conditioned = bool(checkpoint.get("head_sigma_conditioned", run_args.get("head_sigma_conditioned", False)))
    model = NotePadHybridModel(config, key_count=len(load_spec()["keys"]), head_sigma_conditioned=head_sigma_conditioned).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    meta = {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "model_kind": model_kind,
        "head_sigma_conditioned": head_sigma_conditioned,
        "config": asdict(config),
    }
    return model, config_payload, meta


def dummy_inputs(
    config: MicroWAMConfig,
    context_chunks: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if context_chunks < 1:
        raise ValueError("ONNX export currently expects at least one context chunk; use zero-filled context for chunk 0.")
    return (
        torch.randn(batch_size, config.chunk_frames, config.patches_per_frame, config.patch_dim, device=device),
        torch.randn(batch_size, config.chunk_frames, 2, device=device).clamp(-1, 1),
        torch.zeros(batch_size, config.chunk_frames, dtype=torch.long, device=device),
        torch.zeros(batch_size, config.chunk_frames, dtype=torch.long, device=device),
        torch.ones(batch_size, device=device),
        torch.zeros(batch_size, device=device),
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.zeros(batch_size, context_chunks, config.chunk_frames, config.patches_per_frame, config.patch_dim, device=device),
        torch.zeros(batch_size, context_chunks, config.chunk_frames, 4, device=device),
        torch.zeros(batch_size, context_chunks, dtype=torch.long, device=device),
    )


def export_onnx(
    model: NotePadHybridModel,
    out: Path,
    context_chunks: int,
    batch_size: int,
    opset: int,
) -> dict[str, float]:
    out.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_external_data(out)
    wrapper = NotePadHybridOnnxWrapper(model).eval()
    example = dummy_inputs(model.config, context_chunks, batch_size, torch.device("cpu"))
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}
    torch.onnx.export(
        wrapper,
        example,
        out,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        external_data=False,
        do_constant_folding=True,
    )
    return validate_onnx(wrapper, example, out)


def remove_stale_external_data(out: Path) -> None:
    external_data = out.with_suffix(out.suffix + ".data")
    if external_data.exists():
        external_data.unlink()


def validate_onnx(wrapper: NotePadHybridOnnxWrapper, example: tuple[torch.Tensor, ...], out: Path) -> dict[str, float]:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit("Install ONNX export deps with `uv pip install -e '.[onnx]'`.") from error

    onnx_model = onnx.load(out)
    onnx.checker.check_model(onnx_model)
    with torch.no_grad():
        torch_outputs = wrapper(*example)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(out), providers=providers)
    ort_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUT_NAMES, example, strict=True)}
    ort_outputs = session.run(OUTPUT_NAMES, ort_inputs)
    max_abs = 0.0
    mean_abs = 0.0
    count = 0
    for torch_out, ort_out in zip(torch_outputs, ort_outputs, strict=True):
        diff = np.abs(torch_out.detach().cpu().numpy() - ort_out)
        max_abs = max(max_abs, float(diff.max()))
        mean_abs += float(diff.mean())
        count += 1
    return {
        "onnx_max_abs_diff": max_abs,
        "onnx_mean_abs_diff": mean_abs / max(count, 1),
        "onnx_opsets": {opset.domain or "": opset.version for opset in onnx_model.opset_import},
    }


def default_output_path(run: Path, checkpoint_path: Path) -> Path:
    stem = checkpoint_path.stem
    return run / "onnx" / f"{stem}.onnx"


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_glob and args.checkpoint:
        raise ValueError("pass either --checkpoint or --checkpoint-glob, not both")
    if args.checkpoint_glob:
        matches = sorted(args.run.glob(args.checkpoint_glob), key=lambda path: path.name)
        if not matches:
            raise FileNotFoundError(f"no checkpoints matched {args.run / args.checkpoint_glob}")
        return matches
    return [args.checkpoint or (args.run / "checkpoint.pt")]


def export_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    out = args.out or default_output_path(args.run, checkpoint_path)
    if args.out and len(checkpoint_paths(args)) > 1:
        raise ValueError("--out can only be used with a single checkpoint")
    metadata_out = args.metadata_out or out.with_suffix(".json")
    model, config_payload, model_meta = load_hybrid_model(args.run, checkpoint_path, torch.device("cpu"))
    run_args = config_payload.get("args", {})
    if not isinstance(run_args, dict):
        run_args = {}
    context_chunks = int(args.context_chunks if args.context_chunks is not None else run_args.get("context_chunks", 1))
    metrics = (
        {"onnx_validation_skipped": True}
        if args.no_validate
        else export_onnx(model, out, context_chunks=context_chunks, batch_size=args.batch_size, opset=args.opset)
    )
    if args.no_validate:
        out.parent.mkdir(parents=True, exist_ok=True)
        remove_stale_external_data(out)
        wrapper = NotePadHybridOnnxWrapper(model).eval()
        example = dummy_inputs(model.config, context_chunks, args.batch_size, torch.device("cpu"))
        torch.onnx.export(
            wrapper,
            example,
            out,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES},
            opset_version=args.opset,
            external_data=False,
            do_constant_folding=True,
        )
        try:
            import onnx

            onnx_model = onnx.load(out)
            metrics["onnx_opsets"] = {opset.domain or "": opset.version for opset in onnx_model.opset_import}
        except ImportError:
            pass
    metadata = {
        **model_meta,
        "run": str(args.run),
        "onnx_path": str(out),
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "context_chunks": context_chunks,
        "opset": args.opset,
        "validation": metrics,
    }
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    exports = [export_checkpoint(args, checkpoint) for checkpoint in checkpoint_paths(args)]
    payload: Any = exports[0] if len(exports) == 1 else {"exports": exports}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
