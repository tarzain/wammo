from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from wammo.notepad_desk import DeskAction, NotePadDesk, NotepadScriptedPolicy, load_spec


def generate_episode(seed: int, enforce_rare_quota: bool = True, cursor_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    base_spec = None
    if cursor_size is not None:
        base_spec = copy.deepcopy(load_spec())
        base_spec["cursor"]["size"] = int(cursor_size)
    for attempt in range(64):
        desk = NotePadDesk(spec=copy.deepcopy(base_spec) if base_spec is not None else None, seed=seed + attempt * 10_000)
        policy = NotepadScriptedPolicy(desk, seed=seed + 20_000 + attempt)
        steps = int(desk.spec["episode_steps"])
        frames = np.empty((steps, desk.height, desk.width, 3), dtype=np.uint8)
        actions = np.empty((steps, 4), dtype=np.float32)
        rare = 0
        for t in range(steps):
            action = policy.next_action(t)
            if action.mouse_down or action.key != 0:
                rare += 1
            desk.step(action)
            frames[t] = desk.render()
            actions[t] = encode_action(action)
        if not enforce_rare_quota or rare / steps >= float(desk.spec["policies"]["rare_event_min_rate"]):
            return frames, actions
    return frames, actions


def encode_action(action: DeskAction) -> np.ndarray:
    return np.array([action.dx, action.dy, float(action.mouse_down), float(action.key)], dtype=np.float32)


def rare_event_rate(actions: np.ndarray) -> float:
    return float(((actions[..., 2] > 0) | (actions[..., 3] > 0)).mean())


def generate_dataset(episodes: int, seed: int = 0, cursor_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    frames, actions = [], []
    for i in range(episodes):
        ep_frames, ep_actions = generate_episode(seed + i, cursor_size=cursor_size)
        frames.append(ep_frames)
        actions.append(ep_actions)
    return np.stack(frames), np.stack(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cursor-size", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames, actions = generate_dataset(args.episodes, args.seed, cursor_size=args.cursor_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, frames=frames, actions=actions)
    print(f"wrote {args.out} frames={frames.shape} actions={actions.shape} rare_event_rate={rare_event_rate(actions):.3f}")


if __name__ == "__main__":
    main()
