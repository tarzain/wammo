import numpy as np

from wammo.data.notepad import generate_episode, rare_event_rate
from wammo.notepad_desk import DeskAction, NotePadDesk


def move_to(desk: NotePadDesk, x: float, y: float, mouse_down: bool = False) -> None:
    while abs(desk.cursor_x - x) > 1 or abs(desk.cursor_y - y) > 1:
        desk.step(DeskAction(x - desk.cursor_x, y - desk.cursor_y, mouse_down, 0))


def test_new_note_focus_and_type():
    desk = NotePadDesk(seed=0)
    button = desk.spec["toolbar"]["new_note"]
    move_to(desk, button["x"] + button["width"] / 2, button["y"] + button["height"] / 2)
    desk.step(DeskAction(0, 0, True, 0))
    desk.step(DeskAction(0, 0, False, 0))
    assert len(desk.notes) == 1
    assert desk.focused_note() is desk.notes[-1]
    h_key = desk.keys.index("h")
    desk.step(DeskAction(0, 0, False, h_key))
    assert desk.focused_note().text_string == "h"


def test_drag_to_trash_deletes_note():
    desk = NotePadDesk(seed=0)
    desk._spawn_note()
    note = desk.notes[-1]
    move_to(desk, note.x + 5, note.y + 2)
    desk.step(DeskAction(0, 0, True, 0))
    trash = desk.spec["toolbar"]["trash"]
    move_to(desk, trash["x"] + trash["width"] / 2, trash["y"] + trash["height"] / 2, mouse_down=True)
    desk.step(DeskAction(0, 0, False, 0))
    assert len(desk.notes) == 0
    assert desk.focused_id is None


def test_notepad_episode_shapes_and_rare_quota():
    frames, actions = generate_episode(seed=0)
    assert frames.shape == (64, 96, 96, 3)
    assert actions.shape == (64, 4)
    assert frames.dtype == np.uint8
    assert rare_event_rate(actions) >= 0.15
