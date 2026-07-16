from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SPEC_PATH = Path(__file__).resolve().parents[3] / "specs" / "cursor_world.json"


@dataclass(frozen=True)
class Action:
    dx: float
    dy: float
    click: bool


@dataclass
class Entity:
    entity_id: int
    shape: str
    color: tuple[int, int, int]
    x: float
    y: float
    size: int

    @property
    def half(self) -> float:
        return self.size / 2

    def contains(self, px: float, py: float) -> bool:
        return self.x - self.half <= px <= self.x + self.half and self.y - self.half <= py <= self.y + self.half


@dataclass
class WorldState:
    entities: list[Entity]
    cursor_x: float
    cursor_y: float
    mouse_down: bool
    grabbed_id: int | None
    background_alt: bool


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or SPEC_PATH).open("r", encoding="utf-8") as f:
        return json.load(f)


class CursorWorld:
    def __init__(self, spec: dict[str, Any] | None = None, seed: int = 0):
        self.spec = spec or load_spec()
        self.rng = np.random.default_rng(seed)
        self.state = self._initial_state()

    @property
    def width(self) -> int:
        return int(self.spec["canvas"]["width"])

    @property
    def height(self) -> int:
        return int(self.spec["canvas"]["height"])

    def _initial_state(self) -> WorldState:
        ent_spec = self.spec["entities"]
        n = int(self.rng.integers(ent_spec["min_count"], ent_spec["max_count"] + 1))
        colors = [tuple(c) for c in ent_spec["colors"]]
        shapes = ent_spec["shapes"]
        min_size, max_size = ent_spec["sizes"]
        entities: list[Entity] = []
        for i in range(n):
            size = int(self.rng.integers(min_size, max_size + 1))
            x = float(self.rng.uniform(size / 2, self.width - size / 2))
            y = float(self.rng.uniform(size / 2, self.height - size / 2))
            entities.append(
                Entity(
                    entity_id=i,
                    shape=str(shapes[i % len(shapes)]),
                    color=colors[i % len(colors)],
                    x=x,
                    y=y,
                    size=size,
                )
            )
        cursor_x, cursor_y = self.spec["cursor"]["start"]
        return WorldState(
            entities=entities,
            cursor_x=float(cursor_x),
            cursor_y=float(cursor_y),
            mouse_down=False,
            grabbed_id=None,
            background_alt=False,
        )

    def reset(self, seed: int | None = None) -> WorldState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self._initial_state()
        return self.state

    def step(self, action: Action) -> WorldState:
        max_delta = float(self.spec["cursor"]["max_delta"])
        dx = float(np.clip(action.dx, -max_delta, max_delta))
        dy = float(np.clip(action.dy, -max_delta, max_delta))
        prev_down = self.state.mouse_down
        self.state.cursor_x = float(np.clip(self.state.cursor_x + dx, 0, self.width - 1))
        self.state.cursor_y = float(np.clip(self.state.cursor_y + dy, 0, self.height - 1))
        self.state.mouse_down = bool(action.click)

        if action.click and not prev_down:
            if self._cursor_in_button():
                self.state.background_alt = not self.state.background_alt
            else:
                self.state.grabbed_id = self._top_entity_at_cursor()

        if not action.click:
            self.state.grabbed_id = None

        if self.state.grabbed_id is not None:
            ent = self.state.entities[self.state.grabbed_id]
            ent.x = float(np.clip(self.state.cursor_x, ent.half, self.width - ent.half))
            ent.y = float(np.clip(self.state.cursor_y, ent.half, self.height - ent.half))

        for ent in self.state.entities:
            ent.x = float(np.clip(ent.x, ent.half, self.width - ent.half))
            ent.y = float(np.clip(ent.y, ent.half, self.height - ent.half))

        return self.state

    def render(self) -> np.ndarray:
        bg_key = "background_alt" if self.state.background_alt else "background"
        bg = tuple(self.spec["canvas"][bg_key])
        img = Image.new("RGB", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)
        for ent in self.state.entities:
            box = [ent.x - ent.half, ent.y - ent.half, ent.x + ent.half, ent.y + ent.half]
            if ent.shape == "cube":
                draw.rectangle(box, fill=ent.color)
            elif ent.shape == "circle":
                draw.ellipse(box, fill=ent.color)
            elif ent.shape == "triangle":
                draw.polygon(
                    [(ent.x, ent.y - ent.half), (ent.x - ent.half, ent.y + ent.half), (ent.x + ent.half, ent.y + ent.half)],
                    fill=ent.color,
                )
        self._draw_button(draw)
        self._draw_cursor(draw)
        return np.asarray(img, dtype=np.uint8)

    def _draw_button(self, draw: ImageDraw.ImageDraw) -> None:
        button = self.spec["button"]
        if not button["enabled"]:
            return
        x, y, w, h = button["x"], button["y"], button["width"], button["height"]
        draw.rectangle([x, y, x + w, y + h], outline=tuple(button["color"]))

    def _draw_cursor(self, draw: ImageDraw.ImageDraw) -> None:
        size = float(self.spec["cursor"]["size"])
        color = tuple(self.spec["cursor"]["color"])
        x, y = self.state.cursor_x, self.state.cursor_y
        draw.line([x - size, y, x + size, y], fill=color)
        draw.line([x, y - size, x, y + size], fill=color)

    def _cursor_in_button(self) -> bool:
        button = self.spec["button"]
        if not button["enabled"]:
            return False
        x, y, w, h = button["x"], button["y"], button["width"], button["height"]
        return x <= self.state.cursor_x <= x + w and y <= self.state.cursor_y <= y + h

    def _top_entity_at_cursor(self) -> int | None:
        for ent in reversed(self.state.entities):
            if ent.contains(self.state.cursor_x, self.state.cursor_y):
                return ent.entity_id
        return None

