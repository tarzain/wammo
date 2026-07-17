from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from wammo.eval.notepad_pixels import cursor_centroids
from wammo.eval.probe_notepad import fit_linear_probe, fit_mlp_probe
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import interpolate, velocity_target
from wammo.model.tokenizer import patchify, patchify_with_coords
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import normalize_notepad_actions
from wammo.train.overfit_one import normalize_frames
from wammo.train.train_notepad import generate_training_dataset
from wammo.train.train_notepad_binned_delta import DELTA_BINS, delta_to_bins
from wammo.train.train_notepad_hybrid import normalize_positions


class RepresentationScreenDataset:
    def __init__(self, frames: np.ndarray, actions: np.ndarray, chunk_frames: int = 4, motion_oversample: bool = True):
        self.frames = frames
        self.actions = actions
        self.positions = cursor_centroids(frames)
        self.chunk_frames = chunk_frames
        self.chunks_per_episode = frames.shape[1] // chunk_frames
        self.motion_oversample = motion_oversample
        self.motion_pairs = self._motion_pairs()
        spec = load_spec()
        self.max_delta = float(spec["cursor"]["max_delta"])
        self.key_count = len(spec["keys"])
        self.width = int(spec["canvas"]["width"])
        self.height = int(spec["canvas"]["height"])

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.motion_oversample and len(self.motion_pairs):
            pair_idx = torch.randint(len(self.motion_pairs), (batch_size,), generator=generator).numpy()
            pairs = self.motion_pairs[pair_idx]
            episode_idx = pairs[:, 0]
            chunk_idx = pairs[:, 1]
        else:
            episode_idx = torch.randint(self.frames.shape[0], (batch_size,), generator=generator).numpy()
            chunk_idx = torch.randint(self.chunks_per_episode, (batch_size,), generator=generator).numpy()
        video_chunks, action_chunks, position_chunks = [], [], []
        for ep, chunk in zip(episode_idx, chunk_idx, strict=True):
            start = int(chunk) * self.chunk_frames
            video_chunks.append(self.frames[int(ep), start : start + self.chunk_frames])
            action_chunks.append(self.actions[int(ep), start : start + self.chunk_frames])
            position_chunks.append(self.positions[int(ep), start : start + self.chunk_frames])
        video = normalize_frames(np.stack(video_chunks)).to(device)
        actions = normalize_notepad_actions(np.stack(action_chunks), self.max_delta, self.key_count).to(device)
        positions = normalize_positions(np.stack(position_chunks), self.width, self.height).to(device)
        return video, actions, positions, torch.as_tensor(chunk_idx, dtype=torch.long, device=device)

    def all_chunks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video = normalize_frames(self.frames.reshape(-1, *self.frames.shape[2:])).reshape(
            -1, self.chunk_frames, *self.frames.shape[2:]
        )
        actions = normalize_notepad_actions(
            self.actions.reshape(-1, self.actions.shape[-1]), self.max_delta, self.key_count
        ).reshape(-1, self.chunk_frames, self.actions.shape[-1])
        positions = normalize_positions(self.positions.reshape(-1, self.chunk_frames, 2), self.width, self.height)
        chunk_ids = torch.arange(self.chunks_per_episode).repeat(self.frames.shape[0])
        return video.to(device), actions.to(device), positions.to(device), chunk_ids.to(device)

    def _motion_pairs(self) -> np.ndarray:
        pairs = []
        for ep in range(self.actions.shape[0]):
            for chunk in range(self.chunks_per_episode):
                start = chunk * self.chunk_frames
                deltas = self.actions[ep, start : start + self.chunk_frames, 0:2]
                if np.abs(deltas).max() > 0.5:
                    pairs.append((ep, chunk))
        return np.asarray(pairs, dtype=np.int64)


class ConvPatchStem(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(16, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = frames.shape
        x = frames.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        y = self.net(x)
        return y.permute(0, 2, 3, 1).reshape(b, t * y.shape[2] * y.shape[3], y.shape[1])


class RepresentationScreenModel(nn.Module):
    def __init__(self, config: MicroWAMConfig, key_count: int, input_mode: str):
        super().__init__()
        self.config = config
        self.key_count = key_count
        self.input_mode = input_mode
        if input_mode == "linear":
            self.video_in = nn.Linear(48, config.d_model)
        elif input_mode == "coord":
            self.video_in = nn.Linear(80, config.d_model)
        elif input_mode == "conv":
            self.video_in = ConvPatchStem(config.d_model)
        else:
            raise ValueError(f"unknown input mode {input_mode}")
        self.delta_in = nn.Linear(2, config.d_model)
        self.button_in = nn.Embedding(3, config.d_model)
        self.key_in = nn.Embedding(key_count + 1, config.d_model)
        self.video_out = nn.Linear(config.d_model, 48)
        self.delta_out = nn.Linear(config.d_model, 2)
        self.dx_out = nn.Linear(config.d_model, DELTA_BINS)
        self.dy_out = nn.Linear(config.d_model, DELTA_BINS)
        self.button_out = nn.Linear(config.d_model, 2)
        self.key_out = nn.Linear(config.d_model, key_count)
        self.cursor_out = nn.Linear(config.d_model, 2)
        self.video_pos = nn.Parameter(torch.zeros(1, config.chunk_frames * config.patches_per_frame, config.d_model))
        self.action_pos = nn.Parameter(torch.zeros(1, config.chunk_frames, config.d_model))
        self.chunk_pos = nn.Embedding(config.max_chunks, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.video_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))
        self.action_sigma = nn.Sequential(nn.Linear(1, config.d_model), nn.SiLU(), nn.Linear(config.d_model, config.d_model))

    def video_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "linear":
            return self.video_in(patchify(frames)).reshape(frames.shape[0], -1, self.config.d_model)
        if self.input_mode == "coord":
            return self.video_in(patchify_with_coords(frames)).reshape(frames.shape[0], -1, self.config.d_model)
        return self.video_in(frames)

    def encode(
        self,
        frames: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        return_layers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        b = frames.shape[0]
        c = self.config.chunk_frames
        p = self.config.patches_per_frame
        chunk_tokens = self.chunk_pos(chunk_ids).unsqueeze(1)
        video_tokens = self.video_tokens(frames)
        action_tokens = self.delta_in(delta_actions) + self.button_in(button_ids) + self.key_in(key_ids)
        video_tokens = video_tokens + self.video_pos + chunk_tokens + self.video_sigma(sigma_video.reshape(b, 1)).unsqueeze(1)
        action_tokens = action_tokens + self.action_pos + chunk_tokens + self.action_sigma(sigma_action.reshape(b, 1)).unsqueeze(1)
        hidden = torch.cat([video_tokens, action_tokens], dim=1)
        layers = [hidden[:, : c * p].reshape(b, c, p, self.config.d_model)] if return_layers else []
        for layer in self.backbone.layers:
            hidden = layer(hidden)
            if return_layers:
                layers.append(hidden[:, : c * p].reshape(b, c, p, self.config.d_model))
        if self.backbone.norm is not None:
            hidden = self.backbone.norm(hidden)
            if return_layers:
                layers.append(hidden[:, : c * p].reshape(b, c, p, self.config.d_model))
        video_hidden = hidden[:, : c * p].reshape(b, c, p, self.config.d_model)
        action_hidden = hidden[:, c * p :]
        return video_hidden, action_hidden, layers

    def forward_all(
        self,
        frames: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_hidden, action_hidden, _ = self.encode(
            frames, delta_actions, button_ids, key_ids, sigma_video, sigma_action, chunk_ids
        )
        frame_hidden = video_hidden.mean(dim=2)
        return (
            self.video_out(video_hidden),
            self.delta_out(action_hidden),
            self.dx_out(action_hidden),
            self.dy_out(action_hidden),
            self.button_out(action_hidden),
            self.key_out(action_hidden),
            self.cursor_out(frame_hidden).sigmoid(),
        )


def screen_training_step(
    model: RepresentationScreenModel,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    position_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
    delta_weight: float,
    delta_ce_weight: float,
    cursor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean.shape[0]
    delta_clean = action_clean[..., 0:2]
    delta_targets = delta_to_bins(delta_clean)
    button_target = action_clean[..., 2].long()
    key_target = action_clean[..., 3].long()
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    delta_noise = torch.randn(delta_clean.shape, device=action_clean.device, generator=generator)
    t_video = torch.rand((b,), device=video_clean.device, generator=generator)
    t_action = torch.rand((b,), device=video_clean.device, generator=generator)
    video_noisy = interpolate(video_clean, video_noise, t_video)
    delta_noisy = interpolate(delta_clean, delta_noise, t_action)
    video_target = velocity_target(patchify(video_clean), patchify(video_noise))
    delta_target = velocity_target(delta_clean, delta_noise)
    video_pred, delta_pred, dx_logits, dy_logits, button_logits, key_logits, cursor_pred = model.forward_all(
        video_noisy, delta_noisy, button_target, key_target, t_video, t_action, chunk_ids
    )
    video_loss = F.mse_loss(video_pred, video_target)
    delta_loss = F.mse_loss(delta_pred, delta_target)
    delta_ce_loss = 0.5 * (
        F.cross_entropy(dx_logits.reshape(-1, DELTA_BINS), delta_targets[..., 0].reshape(-1))
        + F.cross_entropy(dy_logits.reshape(-1, DELTA_BINS), delta_targets[..., 1].reshape(-1))
    )
    button_loss = F.cross_entropy(button_logits.reshape(-1, 2), button_target.reshape(-1))
    key_loss = F.cross_entropy(key_logits.reshape(-1, model.key_count), key_target.reshape(-1))
    cursor_loss = F.mse_loss(cursor_pred, position_clean)
    loss = video_loss + delta_weight * delta_loss + delta_ce_weight * delta_ce_loss + button_loss + key_loss + cursor_weight * cursor_loss
    return loss, {
        "loss": float(loss.detach()),
        "video_loss": float(video_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "delta_ce_loss": float(delta_ce_loss.detach()),
        "cursor_loss": float(cursor_loss.detach()),
        "button_loss": float(button_loss.detach()),
        "key_loss": float(key_loss.detach()),
    }


@torch.no_grad()
def extract_screen_layer_features(
    model: RepresentationScreenModel,
    dataset: RepresentationScreenDataset,
    device: torch.device,
    batch_chunks: int = 64,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    video_all, actions_all, _, chunk_ids_all = dataset.all_chunks(torch.device("cpu"))
    layer_parts: list[list[torch.Tensor]] | None = None
    for start in range(0, video_all.shape[0], batch_chunks):
        end = start + batch_chunks
        video = video_all[start:end].to(device)
        actions = actions_all[start:end].to(device)
        chunk_ids = chunk_ids_all[start:end].to(device)
        zero_actions = torch.zeros_like(actions)
        video_hidden, _, layers = model.encode(
            video,
            zero_actions[..., 0:2],
            zero_actions[..., 2].long(),
            zero_actions[..., 3].long(),
            torch.zeros((video.shape[0],), device=device),
            torch.zeros((video.shape[0],), device=device),
            chunk_ids,
            return_layers=True,
        )
        if not layers:
            layers = [video_hidden]
        if layer_parts is None:
            layer_parts = [[] for _ in layers]
        for layer_index, layer in enumerate(layers):
            layer_parts[layer_index].append(layer.mean(dim=2).cpu().reshape(-1, layer.shape[-1]))
    if layer_parts is None:
        raise ValueError("no chunks available for screen probe")
    positions_out = dataset.positions.reshape(-1, 2)
    deltas_out = dataset.actions.reshape(-1, dataset.actions.shape[-1])[..., 0:2]
    valid = np.isfinite(positions_out).all(axis=-1)
    valid_t = torch.as_tensor(valid)
    return (
        [torch.cat(parts)[valid_t] for parts in layer_parts],
        torch.as_tensor(positions_out[valid], dtype=torch.float32),
        torch.as_tensor(deltas_out[valid], dtype=torch.float32),
    )


def probe_screen_model(
    model: RepresentationScreenModel,
    train_dataset: RepresentationScreenDataset,
    eval_dataset: RepresentationScreenDataset,
    device: torch.device,
    steps: int,
    lr: float,
    mlp_hidden: int = 256,
) -> dict[str, object]:
    train_layers, train_pos, train_delta = extract_screen_layer_features(model, train_dataset, device)
    eval_layers, eval_pos, eval_delta = extract_screen_layer_features(model, eval_dataset, device)
    layer_results = []
    for layer_index, (train_x, eval_x) in enumerate(zip(train_layers, eval_layers, strict=True)):
        _, pos = fit_linear_probe(train_x, train_pos, eval_x, eval_pos, steps, lr, device)
        _, delta = fit_linear_probe(train_x, train_delta, eval_x, eval_delta, steps, lr, device)
        _, pos_mlp = fit_mlp_probe(train_x, train_pos, eval_x, eval_pos, steps, lr, device, hidden_dim=mlp_hidden)
        _, delta_mlp = fit_mlp_probe(train_x, train_delta, eval_x, eval_delta, steps, lr, device, hidden_dim=mlp_hidden)
        layer_results.append(
            {
                "layer": layer_index,
                "position_probe": pos,
                "delta_current_frame_probe": delta,
                "position_mlp_probe": pos_mlp,
                "delta_current_frame_mlp_probe": delta_mlp,
            }
        )
    best = min(layer_results, key=lambda item: item["position_probe"]["mae_mean"])
    best_mlp = min(layer_results, key=lambda item: item["position_mlp_probe"]["mae_mean"])
    return {"best_layer": best, "best_mlp_layer": best_mlp, "layer_sweep": layer_results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["linear", "coord", "conv"], required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=400_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--delta-weight", type=float, default=4.0)
    parser.add_argument("--delta-ce-weight", type=float, default=1.0)
    parser.add_argument("--cursor-weight", type=float, default=1.0)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--generate-progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false; pass --device cpu to run on CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator().manual_seed(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    train_frames, train_actions, train_metadata = generate_training_dataset(
        args.episodes, args.seed, progress_every=args.generate_progress_every
    )
    eval_frames, eval_actions, eval_metadata = generate_training_dataset(args.eval_episodes, args.eval_seed, progress_every=0)
    train_dataset = RepresentationScreenDataset(train_frames, train_actions)
    eval_dataset = RepresentationScreenDataset(eval_frames, eval_actions, motion_oversample=False)
    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        action_dim=4,
        patches_per_frame=24 * 24,
        max_chunks=16,
    )
    model = RepresentationScreenModel(config, key_count=len(load_spec()["keys"]), input_mode=args.input_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    first_metrics = None
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, positions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = screen_training_step(
                model,
                video,
                actions,
                positions,
                chunk_ids,
                noise_generator,
                args.delta_weight,
                args.delta_ce_weight,
                args.cursor_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == args.steps:
                row = {"step": step, "elapsed_sec": round(time.time() - start, 3), **metrics}
                if first_metrics is None:
                    first_metrics = row
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"step={step} loss={row['loss']:.4f} video={row['video_loss']:.4f} "
                    f"delta={row['delta_loss']:.4f} delta_ce={row['delta_ce_loss']:.4f} cursor={row['cursor_loss']:.4f}"
                )
    torch.save({"model": model.state_dict(), "config": asdict(config), "input_mode": args.input_mode}, args.out / "checkpoint.pt")
    probe = probe_screen_model(model, train_dataset, eval_dataset, device, args.probe_steps, 1e-2, mlp_hidden=args.mlp_hidden)
    output = {
        "model_kind": "notepad_representation_screen",
        "input_mode": args.input_mode,
        "args": vars(args) | {"out": str(args.out)},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
        "first_metrics": first_metrics,
        "final_probe": probe,
    }
    (args.out / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
