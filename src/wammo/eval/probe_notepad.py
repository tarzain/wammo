from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from wammo.data.notepad import generate_episode
from wammo.eval.analyze_notepad_run import load_model
from wammo.eval.notepad_pixels import cursor_centroids
from wammo.model.tokenizer import patchify
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import NotePadJointModel, normalize_notepad_actions
from wammo.train.overfit_one import normalize_frames


def make_probe_arrays(episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames, actions = [], []
    for i in range(episodes):
        ep_frames, ep_actions = generate_episode(seed + i)
        frames.append(ep_frames)
        actions.append(ep_actions)
    frames_np = np.stack(frames)
    actions_np = np.stack(actions)
    positions = cursor_centroids(frames_np)
    return frames_np, actions_np, positions


@torch.no_grad()
def extract_features(
    model: NotePadJointModel,
    frames: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    batch_chunks: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = load_spec()
    episode_count, steps = frames.shape[:2]
    chunk_frames = model.config.chunk_frames
    frames_t = normalize_frames(frames.reshape(-1, *frames.shape[2:])).reshape(episode_count, steps, *frames.shape[2:])
    actions_t = normalize_notepad_actions(
        actions.reshape(-1, actions.shape[-1]),
        float(spec["cursor"]["max_delta"]),
        len(spec["keys"]),
    ).reshape(episode_count, steps, actions.shape[-1])
    video_chunks = frames_t.reshape(-1, chunk_frames, *frames_t.shape[2:])
    action_chunks = actions_t.reshape(-1, chunk_frames, actions_t.shape[-1])
    position_chunks = torch.as_tensor(positions.reshape(-1, chunk_frames, 2), dtype=torch.float32)
    delta_chunks = torch.as_tensor(actions.reshape(-1, chunk_frames, actions.shape[-1])[..., 0:2], dtype=torch.float32)
    chunk_ids = torch.arange(steps // chunk_frames).repeat(episode_count)

    feature_parts = []
    position_parts = []
    delta_parts = []
    for start in range(0, video_chunks.shape[0], batch_chunks):
        end = start + batch_chunks
        video = patchify(video_chunks[start:end]).to(device)
        action = torch.zeros_like(action_chunks[start:end]).to(device)
        ids = chunk_ids[start:end].to(device)
        b = video.shape[0]
        sigma_video = torch.zeros((b,), device=device)
        sigma_action = torch.zeros((b,), device=device)
        video_hidden, _ = model.encode_hidden(
            video,
            action[..., 0:2],
            action[..., 2].long(),
            action[..., 3].long(),
            sigma_video,
            sigma_action,
            ids,
        )
        frame_features = video_hidden.mean(dim=2)
        feature_parts.append(frame_features.cpu().reshape(-1, frame_features.shape[-1]))
        position_parts.append(position_chunks[start:end].reshape(-1, 2))
        delta_parts.append(delta_chunks[start:end].reshape(-1, 2))
    features = torch.cat(feature_parts)
    positions_out = torch.cat(position_parts)
    deltas_out = torch.cat(delta_parts)
    valid = torch.isfinite(positions_out).all(dim=-1)
    return features[valid], positions_out[valid], deltas_out[valid]


def encode_hidden_layers(
    model: NotePadJointModel,
    video_patches: torch.Tensor,
    delta_actions: torch.Tensor,
    button_ids: torch.Tensor,
    key_ids: torch.Tensor,
    sigma_video: torch.Tensor,
    sigma_action: torch.Tensor,
    chunk_ids: torch.Tensor,
) -> list[torch.Tensor]:
    b, c, p, d = video_patches.shape
    if (c, p, d) != (model.config.chunk_frames, model.config.patches_per_frame, model.config.patch_dim):
        raise ValueError(f"unexpected video patch shape {tuple(video_patches.shape)}")
    chunk_tokens = model.chunk_pos(chunk_ids).unsqueeze(1)
    video_tokens = model.video_in(video_patches).reshape(b, c * p, model.config.d_model)
    action_tokens = model.delta_in(delta_actions) + model.button_in(button_ids) + model.key_in(key_ids)
    video_tokens = video_tokens + model.video_pos + chunk_tokens + model.video_sigma(sigma_video.reshape(b, 1)).unsqueeze(1)
    action_tokens = action_tokens + model.action_pos + chunk_tokens + model.action_sigma(sigma_action.reshape(b, 1)).unsqueeze(1)
    hidden = torch.cat([video_tokens, action_tokens], dim=1)
    layers = [hidden[:, : c * p].reshape(b, c, p, model.config.d_model)]
    for layer in model.backbone.layers:
        hidden = layer(hidden)
        layers.append(hidden[:, : c * p].reshape(b, c, p, model.config.d_model))
    if model.backbone.norm is not None:
        hidden = model.backbone.norm(hidden)
        layers.append(hidden[:, : c * p].reshape(b, c, p, model.config.d_model))
    return layers


@torch.no_grad()
def extract_layer_features(
    model: NotePadJointModel,
    frames: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray,
    device: torch.device,
    batch_chunks: int = 64,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    spec = load_spec()
    episode_count, steps = frames.shape[:2]
    chunk_frames = model.config.chunk_frames
    frames_t = normalize_frames(frames.reshape(-1, *frames.shape[2:])).reshape(episode_count, steps, *frames.shape[2:])
    actions_t = normalize_notepad_actions(
        actions.reshape(-1, actions.shape[-1]),
        float(spec["cursor"]["max_delta"]),
        len(spec["keys"]),
    ).reshape(episode_count, steps, actions.shape[-1])
    video_chunks = frames_t.reshape(-1, chunk_frames, *frames_t.shape[2:])
    action_chunks = torch.zeros_like(actions_t.reshape(-1, chunk_frames, actions_t.shape[-1]))
    position_chunks = torch.as_tensor(positions.reshape(-1, chunk_frames, 2), dtype=torch.float32)
    delta_chunks = torch.as_tensor(actions.reshape(-1, chunk_frames, actions.shape[-1])[..., 0:2], dtype=torch.float32)
    chunk_ids = torch.arange(steps // chunk_frames).repeat(episode_count)

    layer_parts: list[list[torch.Tensor]] | None = None
    position_parts = []
    delta_parts = []
    for start in range(0, video_chunks.shape[0], batch_chunks):
        end = start + batch_chunks
        video = patchify(video_chunks[start:end]).to(device)
        action = action_chunks[start:end].to(device)
        ids = chunk_ids[start:end].to(device)
        b = video.shape[0]
        sigma_video = torch.zeros((b,), device=device)
        sigma_action = torch.zeros((b,), device=device)
        layers = encode_hidden_layers(
            model,
            video,
            action[..., 0:2],
            action[..., 2].long(),
            action[..., 3].long(),
            sigma_video,
            sigma_action,
            ids,
        )
        if layer_parts is None:
            layer_parts = [[] for _ in layers]
        for layer_index, layer_hidden in enumerate(layers):
            frame_features = layer_hidden.mean(dim=2)
            layer_parts[layer_index].append(frame_features.cpu().reshape(-1, frame_features.shape[-1]))
        position_parts.append(position_chunks[start:end].reshape(-1, 2))
        delta_parts.append(delta_chunks[start:end].reshape(-1, 2))
    if layer_parts is None:
        raise ValueError("no chunks available for layer feature extraction")
    positions_out = torch.cat(position_parts)
    deltas_out = torch.cat(delta_parts)
    valid = torch.isfinite(positions_out).all(dim=-1)
    return [torch.cat(parts)[valid] for parts in layer_parts], positions_out[valid], deltas_out[valid]


@torch.no_grad()
def extract_chunk_video_features(
    model: NotePadJointModel,
    frames: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    batch_chunks: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = load_spec()
    episode_count, steps = frames.shape[:2]
    chunk_frames = model.config.chunk_frames
    frames_t = normalize_frames(frames.reshape(-1, *frames.shape[2:])).reshape(episode_count, steps, *frames.shape[2:])
    actions_t = normalize_notepad_actions(
        actions.reshape(-1, actions.shape[-1]),
        float(spec["cursor"]["max_delta"]),
        len(spec["keys"]),
    ).reshape(episode_count, steps, actions.shape[-1])
    video_chunks = frames_t.reshape(-1, chunk_frames, *frames_t.shape[2:])
    action_chunks = torch.zeros_like(actions_t.reshape(-1, chunk_frames, actions_t.shape[-1]))
    delta_chunks = torch.as_tensor(actions.reshape(-1, chunk_frames, actions.shape[-1])[..., 0:2], dtype=torch.float32)
    chunk_ids = torch.arange(steps // chunk_frames).repeat(episode_count)

    feature_parts = []
    delta_parts = []
    for start in range(0, video_chunks.shape[0], batch_chunks):
        end = start + batch_chunks
        video = patchify(video_chunks[start:end]).to(device)
        action = action_chunks[start:end].to(device)
        ids = chunk_ids[start:end].to(device)
        b = video.shape[0]
        sigma_video = torch.zeros((b,), device=device)
        sigma_action = torch.zeros((b,), device=device)
        video_hidden, _ = model.encode_hidden(
            video,
            action[..., 0:2],
            action[..., 2].long(),
            action[..., 3].long(),
            sigma_video,
            sigma_action,
            ids,
        )
        feature_parts.append(video_hidden.mean(dim=2).cpu())
        delta_parts.append(delta_chunks[start:end])
    return torch.cat(feature_parts), torch.cat(delta_parts)


def visible_delta_features(frame_features: torch.Tensor, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([frame_features[:, :-1], frame_features[:, 1:]], dim=-1).reshape(-1, frame_features.shape[-1] * 2)
    y = deltas[:, :-1].reshape(-1, 2)
    return x, y


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Linear, dict[str, float]]:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    x_train_n = ((x_train - x_mean) / x_std).to(device)
    y_train_n = ((y_train - y_mean) / y_std).to(device)
    x_eval_n = ((x_eval - x_mean) / x_std).to(device)
    y_eval = y_eval.to(device)
    model = nn.Linear(x_train.shape[-1], y_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train_n)
        loss = F.mse_loss(pred, y_train_n)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred_eval = model(x_eval_n) * y_std.to(device) + y_mean.to(device)
        mae = (pred_eval - y_eval).abs().mean(dim=0)
        euclidean = (pred_eval - y_eval).norm(dim=-1).mean()
    return model, {
        "mae_x": float(mae[0]),
        "mae_y": float(mae[1]),
        "mae_mean": float(mae.mean()),
        "euclidean_mean": float(euclidean),
    }


def fit_mlp_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device,
    hidden_dim: int = 256,
) -> tuple[nn.Module, dict[str, float]]:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-5)
    x_train_n = ((x_train - x_mean) / x_std).to(device)
    y_train_n = ((y_train - y_mean) / y_std).to(device)
    x_eval_n = ((x_eval - x_mean) / x_std).to(device)
    y_eval = y_eval.to(device)
    model = nn.Sequential(
        nn.Linear(x_train.shape[-1], hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, y_train.shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train_n)
        loss = F.mse_loss(pred, y_train_n)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred_eval = model(x_eval_n) * y_std.to(device) + y_mean.to(device)
        mae = (pred_eval - y_eval).abs().mean(dim=0)
        euclidean = (pred_eval - y_eval).norm(dim=-1).mean()
    return model, {
        "mae_x": float(mae[0]),
        "mae_y": float(mae[1]),
        "mae_mean": float(mae.mean()),
        "euclidean_mean": float(euclidean),
        "hidden_dim": hidden_dim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/notepad-1k"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=300_000)
    parser.add_argument("--eval-seed", type=int, default=400_000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument("--mlp-probe", action="store_true")
    parser.add_argument("--mlp-hidden", type=int, default=256)
    return parser.parse_args()


def run_probe(
    run_dir: Path,
    device: torch.device,
    train_episodes: int = 256,
    eval_episodes: int = 64,
    train_seed: int = 300_000,
    eval_seed: int = 400_000,
    steps: int = 2000,
    lr: float = 1e-2,
    layer_sweep: bool = False,
    mlp_probe: bool = False,
    mlp_hidden: int = 256,
) -> dict[str, object]:
    model = load_model(run_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_frames, train_actions, train_positions = make_probe_arrays(train_episodes, train_seed)
    eval_frames, eval_actions, eval_positions = make_probe_arrays(eval_episodes, eval_seed)
    x_train, pos_train, delta_train = extract_features(model, train_frames, train_actions, train_positions, device)
    x_eval, pos_eval, delta_eval = extract_features(model, eval_frames, eval_actions, eval_positions, device)
    chunk_x_train, chunk_delta_train = extract_chunk_video_features(model, train_frames, train_actions, device)
    chunk_x_eval, chunk_delta_eval = extract_chunk_video_features(model, eval_frames, eval_actions, device)
    visible_delta_x_train, visible_delta_y_train = visible_delta_features(chunk_x_train, chunk_delta_train)
    visible_delta_x_eval, visible_delta_y_eval = visible_delta_features(chunk_x_eval, chunk_delta_eval)
    _, position_metrics = fit_linear_probe(x_train, pos_train, x_eval, pos_eval, steps, lr, device)
    _, delta_current_metrics = fit_linear_probe(x_train, delta_train, x_eval, delta_eval, steps, lr, device)
    _, delta_visible_metrics = fit_linear_probe(
        visible_delta_x_train,
        visible_delta_y_train,
        visible_delta_x_eval,
        visible_delta_y_eval,
        steps,
        lr,
        device,
    )
    output: dict[str, object] = {
        "run": str(run_dir),
        "model": asdict(model.config),
        "train_examples": int(x_train.shape[0]),
        "eval_examples": int(x_eval.shape[0]),
        "position_probe": position_metrics,
        "delta_current_frame_probe": delta_current_metrics,
        "delta_visible_transition_probe": delta_visible_metrics,
    }
    if mlp_probe:
        _, position_mlp = fit_mlp_probe(x_train, pos_train, x_eval, pos_eval, steps, lr, device, hidden_dim=mlp_hidden)
        _, delta_current_mlp = fit_mlp_probe(
            x_train,
            delta_train,
            x_eval,
            delta_eval,
            steps,
            lr,
            device,
            hidden_dim=mlp_hidden,
        )
        _, delta_visible_mlp = fit_mlp_probe(
            visible_delta_x_train,
            visible_delta_y_train,
            visible_delta_x_eval,
            visible_delta_y_eval,
            steps,
            lr,
            device,
            hidden_dim=mlp_hidden,
        )
        output["position_mlp_probe"] = position_mlp
        output["delta_current_frame_mlp_probe"] = delta_current_mlp
        output["delta_visible_transition_mlp_probe"] = delta_visible_mlp
    if layer_sweep:
        train_layers, train_layer_pos, train_layer_delta = extract_layer_features(
            model, train_frames, train_actions, train_positions, device
        )
        eval_layers, eval_layer_pos, eval_layer_delta = extract_layer_features(model, eval_frames, eval_actions, eval_positions, device)
        layer_results = []
        for layer_index, (layer_train, layer_eval) in enumerate(zip(train_layers, eval_layers, strict=True)):
            _, layer_position = fit_linear_probe(layer_train, train_layer_pos, layer_eval, eval_layer_pos, steps, lr, device)
            _, layer_delta = fit_linear_probe(layer_train, train_layer_delta, layer_eval, eval_layer_delta, steps, lr, device)
            row = {
                "layer": layer_index,
                "position_probe": layer_position,
                "delta_current_frame_probe": layer_delta,
            }
            if mlp_probe:
                _, layer_position_mlp = fit_mlp_probe(
                    layer_train,
                    train_layer_pos,
                    layer_eval,
                    eval_layer_pos,
                    steps,
                    lr,
                    device,
                    hidden_dim=mlp_hidden,
                )
                _, layer_delta_mlp = fit_mlp_probe(
                    layer_train,
                    train_layer_delta,
                    layer_eval,
                    eval_layer_delta,
                    steps,
                    lr,
                    device,
                    hidden_dim=mlp_hidden,
                )
                row["position_mlp_probe"] = layer_position_mlp
                row["delta_current_frame_mlp_probe"] = layer_delta_mlp
            layer_results.append(row)
        output["layer_sweep"] = layer_results
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "linear_probe.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; pass --device cpu")
    device = torch.device(args.device)
    output = run_probe(
        args.run,
        device,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        steps=args.steps,
        lr=args.lr,
        layer_sweep=args.layer_sweep,
        mlp_probe=args.mlp_probe,
        mlp_hidden=args.mlp_hidden,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
