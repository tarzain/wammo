from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import Action, CursorWorld


@dataclass
class ScriptedPolicy:
    world: CursorWorld
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.drag_target: tuple[float, float] | None = None
        self.drag_steps_left = 0

    def next_action(self, step: int) -> Action:
        del step
        probs = self.world.spec["policies"]
        mode = self.rng.choice(
            ["idle", "click", "drag", "walk"],
            p=[
                probs["idle_probability"],
                probs["click_probability"],
                probs["drag_probability"],
                probs["random_walk_probability"],
            ],
        )
        if self.drag_steps_left > 0:
            return self._continue_drag()
        if mode == "idle":
            return Action(0.0, 0.0, False)
        if mode == "click":
            return self._click_button()
        if mode == "drag":
            return self._start_drag()
        return self._random_walk()

    def _random_walk(self) -> Action:
        max_delta = self.world.spec["cursor"]["max_delta"]
        return Action(float(self.rng.uniform(-max_delta, max_delta)), float(self.rng.uniform(-max_delta, max_delta)), False)

    def _click_button(self) -> Action:
        button = self.world.spec["button"]
        target_x = button["x"] + button["width"] / 2
        target_y = button["y"] + button["height"] / 2
        near_button = abs(target_x - self.world.state.cursor_x) <= 1 and abs(target_y - self.world.state.cursor_y) <= 1
        return self._move_toward(target_x, target_y, click=near_button)

    def _start_drag(self) -> Action:
        ent = self.rng.choice(self.world.state.entities)
        if not ent.contains(self.world.state.cursor_x, self.world.state.cursor_y):
            return self._move_toward(ent.x, ent.y, click=False)
        self.drag_target = (
            float(self.rng.uniform(ent.half, self.world.width - ent.half)),
            float(self.rng.uniform(ent.half, self.world.height - ent.half)),
        )
        self.drag_steps_left = int(self.rng.integers(6, 18))
        return Action(0.0, 0.0, True)

    def _continue_drag(self) -> Action:
        assert self.drag_target is not None
        self.drag_steps_left -= 1
        return self._move_toward(*self.drag_target, click=self.drag_steps_left > 0)

    def _move_toward(self, x: float, y: float, click: bool) -> Action:
        max_delta = self.world.spec["cursor"]["max_delta"]
        dx = float(np.clip(x - self.world.state.cursor_x, -max_delta, max_delta))
        dy = float(np.clip(y - self.world.state.cursor_y, -max_delta, max_delta))
        return Action(dx, dy, click)
