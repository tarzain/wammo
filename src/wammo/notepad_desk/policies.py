from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import DeskAction, NotePadDesk


@dataclass
class NotepadScriptedPolicy:
    desk: NotePadDesk
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.plan: list[DeskAction] = []

    def next_action(self, step: int) -> DeskAction:
        del step
        if self.plan:
            return self.plan.pop(0)
        mode = self._sample_mode()
        if mode == "create":
            self._plan_click_toolbar("new_note")
        elif mode == "type":
            self._plan_type()
        elif mode == "drag":
            self._plan_drag()
        elif mode == "trash":
            self._plan_trash()
        elif mode == "hover":
            self._plan_hover()
        else:
            self.plan.append(DeskAction(0.0, 0.0, False, 0))
        if not self.plan:
            self.plan.append(DeskAction(0.0, 0.0, False, 0))
        return self.plan.pop(0)

    def _sample_mode(self) -> str:
        probs = self.desk.spec["policies"]
        modes = ["idle", "hover", "create", "drag", "type", "trash"]
        weights = np.array(
            [
                probs["idle_probability"],
                probs["hover_probability"],
                probs["create_probability"],
                probs["drag_probability"],
                probs["type_probability"],
                probs["trash_probability"],
            ],
            dtype=np.float64,
        )
        weights = weights / weights.sum()
        return str(self.rng.choice(modes, p=weights))

    def _plan_click_toolbar(self, name: str) -> None:
        box = self.desk.spec["toolbar"][name]
        self._plan_move_to(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        self.plan.append(DeskAction(0.0, 0.0, True, 0))
        self.plan.append(DeskAction(0.0, 0.0, False, 0))

    def _plan_type(self) -> None:
        if self.desk.focused_note() is None:
            if not self.desk.notes:
                self._plan_click_toolbar("new_note")
                return
            note = self.desk.notes[-1]
            self._plan_move_to(note.x + 8, note.y + 10)
            self.plan.append(DeskAction(0.0, 0.0, True, 0))
            self.plan.append(DeskAction(0.0, 0.0, False, 0))
            return
        letters = [i for i, key in enumerate(self.desk.keys) if key not in {"none", "backspace"}]
        for _ in range(int(self.rng.integers(1, 4))):
            self.plan.append(DeskAction(0.0, 0.0, False, int(self.rng.choice(letters))))

    def _plan_drag(self) -> None:
        if not self.desk.notes:
            self._plan_click_toolbar("new_note")
            return
        note = self.rng.choice(self.desk.notes)
        self._plan_move_to(note.x + 5, note.y + 2)
        self.plan.append(DeskAction(0.0, 0.0, True, 0))
        target_x = float(self.rng.uniform(8, self.desk.width - self.desk.spec["note"]["width"]))
        target_y = float(self.rng.uniform(self.desk.spec["canvas"]["toolbar_height"] + 2, self.desk.height - self.desk.spec["note"]["height"]))
        self._plan_move_to(target_x, target_y, mouse_down=True)
        self.plan.append(DeskAction(0.0, 0.0, False, 0))

    def _plan_trash(self) -> None:
        if not self.desk.notes:
            self._plan_click_toolbar("new_note")
            return
        note = self.desk.notes[-1]
        trash = self.desk.spec["toolbar"]["trash"]
        self._plan_move_to(note.x + 5, note.y + 2)
        self.plan.append(DeskAction(0.0, 0.0, True, 0))
        self._plan_move_to(trash["x"] + trash["width"] / 2, trash["y"] + trash["height"] / 2, mouse_down=True)
        self.plan.append(DeskAction(0.0, 0.0, False, 0))

    def _plan_hover(self) -> None:
        targets: list[tuple[float, float]] = []
        for box in self.desk.spec["toolbar"].values():
            targets.append((box["x"] + box["width"] / 2, box["y"] + box["height"] / 2))
        for note in self.desk.notes:
            targets.append((note.x + 5, note.y + 2))
            targets.append((note.x + 10, note.y + 12))
        x, y = targets[int(self.rng.integers(0, len(targets)))]
        self._plan_move_to(x, y)

    def _plan_move_to(self, x: float, y: float, mouse_down: bool = False) -> None:
        max_delta = float(self.desk.spec["cursor"]["max_delta"])
        cx, cy = self.desk.cursor_x, self.desk.cursor_y
        while abs(x - cx) > 1 or abs(y - cy) > 1:
            dx = float(np.clip(x - cx, -max_delta, max_delta))
            dy = float(np.clip(y - cy, -max_delta, max_delta))
            self.plan.append(DeskAction(dx, dy, mouse_down, 0))
            cx += dx
            cy += dy
