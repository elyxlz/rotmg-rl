"""All synthetic input into the :99 display via XTest: clicks, typing, chat, shift-use, and the
policy action -> WASD/mouse/space mapping. The one place that touches Xlib.

Consolidates flash_click.py, flash_type.py, shiftuse.py, login_type.py and action_to_input.py.
A single persistent Display connection is reused (open_display) so held movement keys keep their
state across apply_action calls.

Xlib is a GPU-box-only dep (python-xlib); imported lazily so this module can be import-checked off-box.
"""

from __future__ import annotations

import math
import time

import config

_DISPLAY = None
_HELD_KEYS: set[str] = set()
_SHOOTING = False


def _d():
    global _DISPLAY
    if _DISPLAY is None:
        from Xlib import display

        _DISPLAY = display.Display(config.DISPLAY)
    return _DISPLAY


# ---- primitives ----------------------------------------------------------------------------
def move_mouse(x: int, y: int) -> None:
    from Xlib import X
    from Xlib.ext import xtest

    d = _d()
    xtest.fake_input(d, X.MotionNotify, x=x, y=y)
    d.sync()


def click(x: int, y: int, settle: float = 0.3) -> None:
    from Xlib import X
    from Xlib.ext import xtest

    d = _d()
    move_mouse(x, y)
    time.sleep(settle)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(0.08)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()
    time.sleep(settle)


def shift_click(x: int, y: int) -> None:
    """Hold Shift while clicking -- the in-game gesture that USES an inventory item (the portal key)."""
    from Xlib import X, XK
    from Xlib.ext import xtest

    d = _d()
    shift = d.keysym_to_keycode(XK.XK_Shift_L)
    move_mouse(x, y)
    time.sleep(0.2)
    xtest.fake_input(d, X.KeyPress, shift)
    d.sync()
    time.sleep(0.1)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(0.08)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()
    time.sleep(0.1)
    xtest.fake_input(d, X.KeyRelease, shift)
    d.sync()


def _tap(keysym: int, shift: bool = False) -> None:
    from Xlib import X, XK
    from Xlib.ext import xtest

    d = _d()
    keycode = d.keysym_to_keycode(keysym)
    if keycode == 0:
        return
    shift_kc = d.keysym_to_keycode(XK.XK_Shift_L)
    if shift:
        xtest.fake_input(d, X.KeyPress, shift_kc)
        d.sync()
        time.sleep(0.01)
    xtest.fake_input(d, X.KeyPress, keycode)
    d.sync()
    time.sleep(0.03)
    xtest.fake_input(d, X.KeyRelease, keycode)
    d.sync()
    time.sleep(0.03)
    if shift:
        xtest.fake_input(d, X.KeyRelease, shift_kc)
        d.sync()
        time.sleep(0.01)


def _char_keysym(ch: str):
    from Xlib import XK

    if ch == " ":
        return XK.XK_space, False
    if ch == "/":
        return XK.XK_slash, False
    if ch == "@":
        return XK.XK_2, True
    if ch == ".":
        return XK.XK_period, False
    if ch.isalpha():
        return XK.string_to_keysym(ch.lower()), ch.isupper()
    return XK.string_to_keysym(ch), False


def type_text(text: str) -> None:
    for ch in text:
        keysym, shift = _char_keysym(ch)
        _tap(keysym, shift)


def focus_stage() -> None:
    """Click the stage to grab Flash keyboard focus (needs miniwm running as the EWMH WM)."""
    click(*config.STAGE_FOCUS, settle=0.1)


def send_chat(line: str) -> None:
    """Open the in-client console (Escape), type a line, send it (Return), close the console.

    FIX 1: pass /give lines in lowercase (`/give snake pit key`). The capital-S routed through the
    XTest Shift modifier drops intermittently and yields "nake Pit Key" -> item-not-found; /give is
    case-insensitive so lowercase is always safe.
    """
    from Xlib import XK

    focus_stage()
    _tap(XK.XK_Escape)
    time.sleep(0.4)
    type_text(line)
    time.sleep(0.2)
    _tap(XK.XK_Return)
    time.sleep(0.2)
    _tap(XK.XK_Escape)
    time.sleep(0.1)


# ---- policy action -> input ----------------------------------------------------------------
_MOVE_DIRS = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]


def _wasd_for(dx: float, dy: float) -> set[str]:
    keys: set[str] = set()
    if dy < -0.3:
        keys.add("w")
    if dy > 0.3:
        keys.add("s")
    if dx > 0.3:
        keys.add("d")
    if dx < -0.3:
        keys.add("a")
    return keys


def _set_held(new_keys: set[str]) -> None:
    from Xlib import X, XK
    from Xlib.ext import xtest

    global _HELD_KEYS
    d = _d()
    for key in _HELD_KEYS - new_keys:
        xtest.fake_input(d, X.KeyRelease, d.keysym_to_keycode(XK.string_to_keysym(key)))
    for key in new_keys - _HELD_KEYS:
        xtest.fake_input(d, X.KeyPress, d.keysym_to_keycode(XK.string_to_keysym(key)))
    d.sync()
    _HELD_KEYS = new_keys


def apply_action(move: int, aim: int, shoot: bool, cast: bool) -> None:
    """Drive one policy action. move 0=stop, 1..8=direction; aim 0..31 -> mouse angle; shoot held;
    cast = a Space tap (ability)."""
    from Xlib import X, XK
    from Xlib.ext import xtest

    global _SHOOTING
    d = _d()

    if move == 0:
        _set_held(set())
    else:
        dx, dy = _MOVE_DIRS[move - 1]
        rot = math.radians(config.MOVE_FRAME_ROT_DEG)  # align the sim world frame to the client WASD frame
        rdx = dx * math.cos(rot) - dy * math.sin(rot)
        rdy = dx * math.sin(rot) + dy * math.cos(rot)
        _set_held(_wasd_for(rdx, rdy))

    cx, cy = config.AIM_CENTER
    radius = config.AIM_RADIUS_TILES * config.AIM_TILE
    angle = aim * 2 * math.pi / 32
    xtest.fake_input(d, X.MotionNotify, x=int(cx + radius * math.cos(angle)), y=int(cy + radius * math.sin(angle)))
    d.sync()

    if shoot and not _SHOOTING:
        xtest.fake_input(d, X.ButtonPress, 1)
        _SHOOTING = True
    elif not shoot and _SHOOTING:
        xtest.fake_input(d, X.ButtonRelease, 1)
        _SHOOTING = False
    d.sync()

    if cast:
        space = d.keysym_to_keycode(XK.XK_space)
        xtest.fake_input(d, X.KeyPress, space)
        d.sync()
        time.sleep(0.02)
        xtest.fake_input(d, X.KeyRelease, space)
        d.sync()


def release_all() -> None:
    """Release every held key and the mouse button -- call when stopping the policy."""
    global _SHOOTING
    from Xlib import X
    from Xlib.ext import xtest

    _set_held(set())
    if _SHOOTING:
        xtest.fake_input(_d(), X.ButtonRelease, 1)
        _SHOOTING = False
        _d().sync()
