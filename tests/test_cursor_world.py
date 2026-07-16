import numpy as np

from wammo.cursor_world.sim import Action, CursorWorld


def test_deterministic_render_for_seed():
    a = CursorWorld(seed=123)
    b = CursorWorld(seed=123)
    for _ in range(8):
        action = Action(1.5, -0.5, False)
        a.step(action)
        b.step(action)
    np.testing.assert_array_equal(a.render(), b.render())


def test_button_click_toggles_background():
    world = CursorWorld(seed=0)
    button = world.spec["button"]
    target_x = button["x"] + button["width"] / 2
    target_y = button["y"] + button["height"] / 2
    while abs(world.state.cursor_x - target_x) > 1 or abs(world.state.cursor_y - target_y) > 1:
        world.step(Action(target_x - world.state.cursor_x, target_y - world.state.cursor_y, False))
    world.step(Action(0, 0, True))
    assert world.state.background_alt is True
