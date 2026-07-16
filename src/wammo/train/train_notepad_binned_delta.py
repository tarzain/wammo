from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from wammo.eval.divergence_ladder import notepad_divergence_ladder
from wammo.model.dit import MicroWAMConfig
from wammo.model.flow import interpolate, velocity_target
from wammo.notepad_desk import load_spec
from wammo.train.overfit_notepad_one import (
    NotePadJointModel,
    denormalize_notepad_actions,
    write_contact_sheet,
)
from wammo.train.train_notepad import (
    NotePadMultiEpisodeChunks,
    generate_training_dataset,
    make_eval_dataset,
)


DELTA_BINS = 17


def delta_to_bins(delta_norm: torch.Tensor, max_delta: float = 8.0) -> torch.Tensor:
    raw = (delta_norm * max_delta).round().clamp(-max_delta, max_delta).long()
    return (raw + int(max_delta)).clamp(0, DELTA_BINS - 1)


def bins_to_delta_norm(bins: torch.Tensor, max_delta: float = 8.0) -> torch.Tensor:
    return (bins.float() - max_delta) / max_delta


class NotePadBinnedDeltaModel(NotePadJointModel):
    delta_prediction_kind = "x0"

    def __init__(self, config: MicroWAMConfig, key_count: int):
        super().__init__(config, key_count)
        self.delta_out = nn.Identity()
        self.dx_out = nn.Linear(config.d_model, DELTA_BINS)
        self.dy_out = nn.Linear(config.d_model, DELTA_BINS)

    def forward_logits(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        action_drop: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_hidden, action_hidden = self.encode_hidden(
            video_patches,
            delta_actions,
            button_ids,
            key_ids,
            sigma_video,
            sigma_action,
            chunk_ids,
            action_drop,
        )
        return (
            self.video_out(video_hidden),
            self.dx_out(action_hidden),
            self.dy_out(action_hidden),
            self.button_out(action_hidden),
            self.key_out(action_hidden),
        )

    def forward(
        self,
        video_patches: torch.Tensor,
        delta_actions: torch.Tensor,
        button_ids: torch.Tensor,
        key_ids: torch.Tensor,
        sigma_video: torch.Tensor,
        sigma_action: torch.Tensor,
        chunk_ids: torch.Tensor,
        action_drop: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        video_out, dx_logits, dy_logits, button_logits, key_logits = self.forward_logits(
            video_patches,
            delta_actions,
            button_ids,
            key_ids,
            sigma_video,
            sigma_action,
            chunk_ids,
            action_drop,
        )
        delta_pred = torch.stack(
            [
                bins_to_delta_norm(dx_logits.argmax(dim=-1)),
                bins_to_delta_norm(dy_logits.argmax(dim=-1)),
            ],
            dim=-1,
        )
        return video_out, delta_pred, button_logits, key_logits


def binned_training_step(
    model: NotePadBinnedDeltaModel,
    video_clean: torch.Tensor,
    action_clean: torch.Tensor,
    chunk_ids: torch.Tensor,
    generator: torch.Generator,
    action_weight: float = 1.0,
    action_dropout: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    b = video_clean.shape[0]
    delta_clean = action_clean[..., 0:2]
    delta_targets = delta_to_bins(delta_clean)
    button_target = action_clean[..., 2].long()
    key_target = action_clean[..., 3].long()
    video_noise = torch.randn(video_clean.shape, device=video_clean.device, generator=generator)
    delta_noise = torch.randn(delta_clean.shape, device=action_clean.device, generator=generator)
    t_video = torch.rand((b,), device=video_clean.device, generator=generator)
    t_action = torch.ones((b,), device=video_clean.device)
    video_noisy = interpolate(video_clean, video_noise, t_video)
    action_drop = None
    if action_dropout > 0:
        action_drop = torch.rand((b, 1, 1), device=video_clean.device, generator=generator) < action_dropout
    video_target = velocity_target(video_clean, video_noise)
    video_pred, dx_logits, dy_logits, button_logits, key_logits = model.forward_logits(
        video_noisy, delta_noise, button_target, key_target, t_video, t_action, chunk_ids, action_drop
    )
    video_loss = F.mse_loss(video_pred, video_target)
    dx_loss = F.cross_entropy(dx_logits.reshape(-1, DELTA_BINS), delta_targets[..., 0].reshape(-1))
    dy_loss = F.cross_entropy(dy_logits.reshape(-1, DELTA_BINS), delta_targets[..., 1].reshape(-1))
    delta_loss = 0.5 * (dx_loss + dy_loss)
    button_loss = F.cross_entropy(button_logits.reshape(-1, 2), button_target.reshape(-1))
    key_loss = F.cross_entropy(key_logits.reshape(-1, model.key_count), key_target.reshape(-1))
    action_loss = delta_loss + button_loss + key_loss
    loss = video_loss + action_weight * action_loss
    return loss, {
        "loss": float(loss.detach()),
        "video_loss": float(video_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "delta_loss": float(delta_loss.detach()),
        "dx_loss": float(dx_loss.detach()),
        "dy_loss": float(dy_loss.detach()),
        "button_loss": float(button_loss.detach()),
        "key_loss": float(key_loss.detach()),
    }


@torch.no_grad()
def evaluate_binned(
    model: NotePadBinnedDeltaModel,
    dataset,
    device: torch.device,
    action_weight: float,
    action_dropout: float,
    seed: int = 999,
) -> dict[str, float]:
    spec = load_spec()
    key_count = len(spec["keys"])
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    video_clean, action_clean, chunk_ids = dataset.all_chunks(device)
    loss, metrics = binned_training_step(model, video_clean, action_clean, chunk_ids, generator, action_weight, action_dropout=0.0)
    del loss
    video_noise = torch.randn(video_clean.shape, device=device, generator=generator)
    delta_noise = torch.randn(action_clean[..., 0:2].shape, device=device, generator=generator)
    button_true = action_clean[..., 2].long()
    key_true = action_clean[..., 3].long()
    t_video = torch.ones((video_clean.shape[0],), device=device)
    t_action = torch.ones((video_clean.shape[0],), device=device)
    _, dx_logits, dy_logits, button_logits, key_logits = model.forward_logits(
        video_noise, delta_noise, button_true, key_true, t_video, t_action, chunk_ids
    )
    raw_true = denormalize_notepad_actions(action_clean, float(spec["cursor"]["max_delta"]), key_count)
    pred_delta = torch.stack(
        [
            bins_to_delta_norm(dx_logits.argmax(dim=-1)),
            bins_to_delta_norm(dy_logits.argmax(dim=-1)),
        ],
        dim=-1,
    )
    raw_pred = raw_true.clone()
    raw_pred[..., 0:2] = pred_delta * float(spec["cursor"]["max_delta"])
    raw_pred[..., 2] = button_logits.argmax(dim=-1)
    raw_pred[..., 3] = key_logits.argmax(dim=-1)
    click_true = raw_true[..., 2] >= 0.5
    click_pred = raw_pred[..., 2] >= 0.5
    key_true_raw = raw_true[..., 3].long()
    key_pred = raw_pred[..., 3].long().clamp(0, key_count - 1)
    key_event = key_true_raw != 0
    key_event_acc = (
        (key_pred[key_event] == key_true_raw[key_event]).float().mean()
        if key_event.any()
        else torch.tensor(1.0, device=device)
    )
    metrics.update(
        {
            "video_mae": float((video_noise - video_clean).abs().mean()),
            "delta_mae_px": float((raw_pred[..., 0:2] - raw_true[..., 0:2]).abs().mean()),
            "click_accuracy": float((click_pred == click_true).float().mean()),
            "key_accuracy": float((key_pred == key_true_raw).float().mean()),
            "key_event_accuracy": float(key_event_acc),
            "action_mae": float((pred_delta - action_clean[..., 0:2]).abs().mean()),
        }
    )
    model.train()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--out", type=Path, default=Path("runs/notepad-1k-binned-delta-ce"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ladder-every", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--action-dropout", type=float, default=0.0)
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
    train_dataset = NotePadMultiEpisodeChunks(train_frames, train_actions)
    eval_dataset, eval_metadata = make_eval_dataset(args.eval_seed)

    config = MicroWAMConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        action_dim=4,
        patches_per_frame=24 * 24,
        max_chunks=16,
    )
    model = NotePadBinnedDeltaModel(config, key_count=len(load_spec()["keys"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_kind": "notepad_binned_delta",
        "args": vars(args) | {"out": str(args.out)},
        "model": asdict(config),
        "train_dataset": train_metadata,
        "eval_dataset": eval_metadata,
    }
    (args.out / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    first_eval: dict[str, float] | None = None
    start = time.time()
    with (args.out / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            video, actions, chunk_ids = train_dataset.sample(args.batch_size, sample_generator, device)
            optimizer.zero_grad(set_to_none=True)
            loss, train_metrics = binned_training_step(
                model, video, actions, chunk_ids, noise_generator, args.action_weight, args.action_dropout
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                eval_metrics = evaluate_binned(model, eval_dataset, device, args.action_weight, args.action_dropout)
                if args.ladder_every > 0 and (step == 1 or step % args.ladder_every == 0 or step == args.steps):
                    video_all, action_all, chunk_ids_all = eval_dataset.all_chunks(device)
                    eval_metrics.update(
                        notepad_divergence_ladder(
                            model,
                            video_all,
                            action_all,
                            chunk_ids_all,
                            key_index=load_spec()["keys"].index("h"),
                        )
                    )
                if first_eval is None:
                    first_eval = eval_metrics
                row = {
                    "step": step,
                    "elapsed_sec": round(time.time() - start, 3),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"eval_{k}": v for k, v in eval_metrics.items()},
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"step={step} loss={row['eval_loss']:.4f} video={row['eval_video_loss']:.4f} "
                    f"action={row['eval_action_loss']:.4f} delta_ce={row['eval_delta_loss']:.4f} "
                    f"click={row['eval_click_accuracy']:.3f} key={row['eval_key_accuracy']:.3f} "
                    f"key_event={row['eval_key_event_accuracy']:.3f}"
                )

    torch.save({"model": model.state_dict(), "config": asdict(config), "model_kind": "notepad_binned_delta"}, args.out / "checkpoint.pt")
    write_contact_sheet(model, eval_dataset, args.out / "contact_sheet.png", device)
    final_eval = evaluate_binned(model, eval_dataset, device, args.action_weight, args.action_dropout)
    video_all, action_all, chunk_ids_all = eval_dataset.all_chunks(device)
    final_eval.update(
        notepad_divergence_ladder(model, video_all, action_all, chunk_ids_all, key_index=load_spec()["keys"].index("h"))
    )
    (args.out / "summary.json").write_text(json.dumps({"first_eval": first_eval, "final_eval": final_eval}, indent=2) + "\n")


if __name__ == "__main__":
    main()
