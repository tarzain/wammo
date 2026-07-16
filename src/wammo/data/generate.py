from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wammo.cursor_world.policies import ScriptedPolicy
from wammo.cursor_world.sim import CursorWorld


def generate_episode(seed: int) -> tuple[np.ndarray, np.ndarray]:
    world = CursorWorld(seed=seed)
    policy = ScriptedPolicy(world, seed=seed + 10_000)
    steps = int(world.spec["episode_steps"])
    frames = np.empty((steps, world.height, world.width, 3), dtype=np.uint8)
    actions = np.empty((steps, 3), dtype=np.float32)
    for t in range(steps):
        action = policy.next_action(t)
        world.step(action)
        frames[t] = world.render()
        actions[t] = (action.dx, action.dy, float(action.click))
    return frames, actions


def generate_dataset(episodes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    frames, actions = [], []
    for i in range(episodes):
        ep_frames, ep_actions = generate_episode(seed + i)
        frames.append(ep_frames)
        actions.append(ep_actions)
    return np.stack(frames), np.stack(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames, actions = generate_dataset(args.episodes, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, frames=frames, actions=actions)
    print(f"wrote {args.out} frames={frames.shape} actions={actions.shape}")


if __name__ == "__main__":
    main()

