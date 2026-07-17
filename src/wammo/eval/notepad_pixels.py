from __future__ import annotations

import numpy as np

from wammo.notepad_desk import load_spec


def cursor_positions_from_actions(actions: np.ndarray) -> np.ndarray:
    spec = load_spec()
    max_delta = float(spec["cursor"]["max_delta"])
    width = int(spec["canvas"]["width"])
    height = int(spec["canvas"]["height"])
    start_x, start_y = [float(v) for v in spec["cursor"]["start"]]
    positions = np.empty((*actions.shape[:2], 2), dtype=np.float32)
    for ep in range(actions.shape[0]):
        x, y = start_x, start_y
        for t in range(actions.shape[1]):
            dx = float(np.clip(actions[ep, t, 0], -max_delta, max_delta))
            dy = float(np.clip(actions[ep, t, 1], -max_delta, max_delta))
            x = float(np.clip(x + dx, 0, width - 1))
            y = float(np.clip(y + dy, 0, height - 1))
            positions[ep, t, 0] = x
            positions[ep, t, 1] = y
    return positions


def cursor_centroids(frames: np.ndarray) -> np.ndarray:
    spec = load_spec()
    colors = np.array([spec["cursor"]["color"], spec["cursor"]["hand_color"]], dtype=np.int16)
    flat = frames.astype(np.int16)
    positions = np.full((*frames.shape[:2], 2), np.nan, dtype=np.float32)
    yy, xx = np.indices(frames.shape[2:4])
    for ep in range(frames.shape[0]):
        for t in range(frames.shape[1]):
            pixel = flat[ep, t]
            mask = np.zeros(pixel.shape[:2], dtype=bool)
            for color in colors:
                mask |= np.abs(pixel - color).max(axis=-1) <= 8
            if mask.any():
                positions[ep, t, 0] = float(xx[mask].mean())
                positions[ep, t, 1] = float(yy[mask].mean())
    return positions
