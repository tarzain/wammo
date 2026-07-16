from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .font import glyph_for


SPEC_PATH = Path(__file__).resolve().parents[3] / "specs" / "notepad_desk.json"


@dataclass(frozen=True)
class DeskAction:
    dx: float
    dy: float
    mouse_down: bool
    key: int = 0


@dataclass
class Note:
    note_id: int
    x: float
    y: float
    color_index: int
    text: list[str] = field(default_factory=list)

    @property
    def text_string(self) -> str:
        return "".join(self.text)


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    with Path(path or SPEC_PATH).open("r", encoding="utf-8") as f:
        return json.load(f)


class NotePadDesk:
    def __init__(self, spec: dict[str, Any] | None = None, seed: int = 0):
        self.spec = spec or load_spec()
        self.rng = np.random.default_rng(seed)
        self.cursor_x, self.cursor_y = [float(v) for v in self.spec["cursor"]["start"]]
        self.mouse_down = False
        self.notes: list[Note] = []
        self.focused_id: int | None = None
        self.dragging_id: int | None = None
        self.drag_offset: tuple[float, float] = (0.0, 0.0)
        self.next_note_id = 0

    @property
    def width(self) -> int:
        return int(self.spec["canvas"]["width"])

    @property
    def height(self) -> int:
        return int(self.spec["canvas"]["height"])

    @property
    def keys(self) -> list[str]:
        return list(self.spec["keys"])

    def step(self, action: DeskAction) -> None:
        max_delta = float(self.spec["cursor"]["max_delta"])
        dx = float(np.clip(action.dx, -max_delta, max_delta))
        dy = float(np.clip(action.dy, -max_delta, max_delta))
        was_down = self.mouse_down
        self.cursor_x = float(np.clip(self.cursor_x + dx, 0, self.width - 1))
        self.cursor_y = float(np.clip(self.cursor_y + dy, 0, self.height - 1))
        self.mouse_down = bool(action.mouse_down)

        if action.key:
            self._type_key(action.key)

        if self.mouse_down and not was_down:
            self._press()
        elif self.mouse_down and self.dragging_id is not None:
            self._drag()
        elif was_down and not self.mouse_down:
            self._release()

    def render(self) -> np.ndarray:
        img = Image.new("RGB", (self.width, self.height), tuple(self.spec["canvas"]["background"]))
        draw = ImageDraw.Draw(img)
        self._draw_toolbar(draw)
        hovered = self.hit_test_note(self.cursor_x, self.cursor_y)
        for note in self.notes:
            self._draw_note(draw, note, hovered_id=hovered[0] if hovered else None)
        self._draw_cursor(draw)
        return np.asarray(img, dtype=np.uint8)

    def hover_target(self) -> str | None:
        toolbar = self.hit_test_toolbar(self.cursor_x, self.cursor_y)
        if toolbar:
            return toolbar
        note_hit = self.hit_test_note(self.cursor_x, self.cursor_y)
        if note_hit:
            _, region = note_hit
            return f"note_{region}"
        return None

    def focused_note(self) -> Note | None:
        if self.focused_id is None:
            return None
        return self._note_by_id(self.focused_id)

    def hit_test_toolbar(self, x: float, y: float) -> str | None:
        for name, box in self.spec["toolbar"].items():
            if box["x"] <= x <= box["x"] + box["width"] and box["y"] <= y <= box["y"] + box["height"]:
                return name
        return None

    def hit_test_note(self, x: float, y: float) -> tuple[int, str] | None:
        w, h = self.spec["note"]["width"], self.spec["note"]["height"]
        title_h = self.spec["note"]["title_height"]
        for note in reversed(self.notes):
            if note.x <= x <= note.x + w and note.y <= y <= note.y + h:
                return note.note_id, "title" if y <= note.y + title_h else "body"
        return None

    def _press(self) -> None:
        toolbar = self.hit_test_toolbar(self.cursor_x, self.cursor_y)
        if toolbar == "new_note":
            self._spawn_note()
            return
        if toolbar == "color_cycle":
            focused = self.focused_note()
            if focused is not None:
                focused.color_index = (focused.color_index + 1) % len(self.spec["note"]["colors"])
            return

        note_hit = self.hit_test_note(self.cursor_x, self.cursor_y)
        if note_hit is None:
            self.focused_id = None
            return
        note_id, region = note_hit
        self._focus(note_id)
        if region == "title":
            note = self._note_by_id(note_id)
            assert note is not None
            self.dragging_id = note_id
            self.drag_offset = (self.cursor_x - note.x, self.cursor_y - note.y)

    def _drag(self) -> None:
        note = self._note_by_id(self.dragging_id)
        if note is None:
            return
        note_w, note_h = self.spec["note"]["width"], self.spec["note"]["height"]
        note.x = float(np.clip(self.cursor_x - self.drag_offset[0], 0, self.width - note_w))
        note.y = float(np.clip(self.cursor_y - self.drag_offset[1], self.spec["canvas"]["toolbar_height"], self.height - note_h))

    def _release(self) -> None:
        if self.dragging_id is not None and self.hit_test_toolbar(self.cursor_x, self.cursor_y) == "trash":
            self.notes = [note for note in self.notes if note.note_id != self.dragging_id]
            if self.focused_id == self.dragging_id:
                self.focused_id = None
        self.dragging_id = None

    def _type_key(self, key_index: int) -> None:
        focused = self.focused_note()
        if focused is None:
            return
        key = self.keys[key_index]
        if key == "none":
            return
        if key == "backspace":
            if focused.text:
                focused.text.pop()
            return
        if len(focused.text) < 12:
            focused.text.append(key)

    def _spawn_note(self) -> None:
        max_count = int(self.spec["note"]["max_count"])
        if len(self.notes) >= max_count:
            return
        positions = self.spec["note"]["spawn_positions"]
        used = {(round(note.x), round(note.y)) for note in self.notes}
        x, y = positions[len(self.notes) % len(positions)]
        for px, py in positions:
            if (px, py) not in used:
                x, y = px, py
                break
        note = Note(self.next_note_id, float(x), float(y), self.next_note_id % len(self.spec["note"]["colors"]))
        self.next_note_id += 1
        self.notes.append(note)
        self._focus(note.note_id)

    def _focus(self, note_id: int) -> None:
        self.focused_id = note_id
        for i, note in enumerate(self.notes):
            if note.note_id == note_id:
                self.notes.append(self.notes.pop(i))
                return

    def _note_by_id(self, note_id: int | None) -> Note | None:
        if note_id is None:
            return None
        for note in self.notes:
            if note.note_id == note_id:
                return note
        return None

    def _draw_toolbar(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rectangle([0, 0, self.width, self.spec["canvas"]["toolbar_height"]], fill=tuple(self.spec["canvas"]["toolbar"]))
        hover = self.hit_test_toolbar(self.cursor_x, self.cursor_y)
        for name, box in self.spec["toolbar"].items():
            fill = (92, 101, 116) if hover == name else (70, 78, 91)
            draw.rectangle([box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]], fill=fill, outline=(188, 196, 208))
        draw.line([5, 6, 9, 6], fill=(245, 245, 245))
        draw.line([7, 4, 7, 8], fill=(245, 245, 245))
        draw.rectangle([19, 4, 23, 8], fill=tuple(self.spec["note"]["colors"][0]))
        trash = self.spec["toolbar"]["trash"]
        draw.line([trash["x"] + 4, trash["y"] + 3, trash["x"] + 12, trash["y"] + 3], fill=(245, 245, 245))
        draw.rectangle([trash["x"] + 5, trash["y"] + 4, trash["x"] + 11, trash["y"] + 9], outline=(245, 245, 245))

    def _draw_note(self, draw: ImageDraw.ImageDraw, note: Note, hovered_id: int | None) -> None:
        note_spec = self.spec["note"]
        x, y = int(round(note.x)), int(round(note.y))
        w, h = note_spec["width"], note_spec["height"]
        title_h = note_spec["title_height"]
        color = tuple(note_spec["colors"][note.color_index])
        border = tuple(note_spec["hover"] if note.note_id == hovered_id else note_spec["border"])
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=border)
        draw.rectangle([x + 1, y + 1, x + w - 1, y + title_h], fill=tuple(max(0, c - 34) for c in color))
        if self.focused_id == note.note_id:
            draw.rectangle([x - 1, y - 1, x + w + 1, y + h + 1], outline=tuple(note_spec["focus"]))
        self._draw_text(draw, note)

    def _draw_text(self, draw: ImageDraw.ImageDraw, note: Note) -> None:
        color = tuple(self.spec["note"]["text"])
        start_x = int(round(note.x)) + 2
        start_y = int(round(note.y)) + self.spec["note"]["title_height"] + 2
        for idx, char in enumerate(note.text[:12]):
            col = idx % 6
            row = idx // 6
            self._draw_glyph(draw, char, start_x + col * 4, start_y + row * 6, color)

    def _draw_glyph(self, draw: ImageDraw.ImageDraw, char: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        for gy, line in enumerate(glyph_for(char)):
            for gx, value in enumerate(line):
                if value == "1":
                    draw.point((x + gx, y + gy), fill=color)

    def _draw_cursor(self, draw: ImageDraw.ImageDraw) -> None:
        color = tuple(self.spec["cursor"]["hand_color"] if self.hover_target() == "note_title" else self.spec["cursor"]["color"])
        x, y = self.cursor_x, self.cursor_y
        size = float(self.spec["cursor"]["size"])
        draw.line([x, y, x, y + size], fill=color)
        draw.line([x, y, x + size - 1, y + size - 1], fill=color)
        draw.line([x + 1, y + size, x + 3, y + size], fill=color)

