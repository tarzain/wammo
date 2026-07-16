from __future__ import annotations

import numpy as np

from wammo.notepad_desk import load_spec


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
