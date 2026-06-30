"""The Flash client lifecycle: bring up Xvfb + miniwm + the Flash client, then log in to the Nexus
and verify it via the in-game gate. Also: grab a screenshot.

FIX 4: the obs pipeline is started fresh per run by run.py BEFORE this; the client connecting to a
fresh proxy is what keeps the obs live. This module only owns the X stack + client + login.
FIX 5: login is just Play -> select-character; the client auto-logs-in as "Wizardbot". No
logout/login/type-credentials cycle (that was the main source of login flakiness).
FIX 6: the in-game gate is obs.obs_is_fresh() -- the obs file exists and is updating -- never a sleep.
"""

from __future__ import annotations

import subprocess
import time

import config
import input as inp
import obs
import proc


def _patterns() -> dict[str, str]:
    return {
        "xvfb": "Xvfb %s" % config.DISPLAY,
        "miniwm": str(config.MINIWM_BIN),
        "flash": "flashplayer",
    }


def start_x_stack() -> None:
    """Idempotently (re)start Xvfb, miniwm and the Flash client. Kills any prior instance first."""
    pats = _patterns()
    proc.pkill(pats["flash"])
    proc.pkill(pats["miniwm"])
    proc.pkill(pats["xvfb"])
    time.sleep(1.0)

    env = config.env_for_subprocess()

    proc.spawn(["Xvfb", config.DISPLAY, "-screen", "0", config.SCREEN], config.XVFB_LOG, env=env)
    time.sleep(2.0)

    proc.spawn([str(config.MINIWM_BIN)], config.MINIWM_LOG, env=env)
    time.sleep(1.0)

    flash_env = dict(env)
    flash_env["LD_PRELOAD"] = str(config.REDIRECT_SO)  # reroute :2050 -> :2052 (the tee proxy)
    flash_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    proc.spawn([str(config.FLASH_BIN), str(config.CLIENT_SWF)], config.FLASH_LOG, env=flash_env)
    time.sleep(config.FLASH_BOOT_SECS)


def client_alive() -> bool:
    return proc.is_alive(_patterns()["flash"])


def _wait_ingame_gate(timeout: float = config.INGAME_GATE_TIMEOUT) -> bool:
    """FIX 6: poll the obs freshness gate instead of sleeping blindly."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if obs.obs_is_fresh():
            return True
        time.sleep(config.GATE_POLL_SECS)
    return False


def login() -> bool:
    """FIX 5 + FIX 8: Play -> select-character, retried until the in-game gate passes (max
    LOGIN_RETRY_MAX). Returns True once obs is live. Idempotent -- safe to call when already in-game."""
    if obs.obs_is_fresh():
        return True
    for attempt in range(1, config.LOGIN_RETRY_MAX + 1):
        inp.click(*config.PLAY_BUTTON)
        time.sleep(config.LOGIN_STEP_PLAY_SECS)
        inp.click(*config.CHAR_SELECT)
        time.sleep(config.LOGIN_STEP_CHAR_SECS)
        if _wait_ingame_gate():
            print("[client] in-game (attempt %d)" % attempt, flush=True)
            return True
        print("[client] login attempt %d: obs not live yet" % attempt, flush=True)
    print("[client] login failed after %d attempts" % config.LOGIN_RETRY_MAX, flush=True)
    return False


def screenshot(path: str) -> bool:
    """One-frame x11grab of :99 to `path`. Returns True on success."""
    res = subprocess.run(
        ["ffmpeg", "-y", "-f", "x11grab", "-video_size", config.SCREEN_SIZE, "-i", config.DISPLAY, "-frames:v", "1", path],
        capture_output=True,
        env=config.env_for_subprocess(),
        check=False,
    )
    return res.returncode == 0


def ensure_ingame() -> bool:
    """The idempotent client gate run.py calls: if already in-game, no-op; else (re)launch + login.

    Avoids a needless client restart when obs is already live (e.g. re-running mid-session)."""
    if obs.obs_is_fresh() and client_alive():
        print("[client] already in-game (obs live, client up)", flush=True)
        return True
    if not client_alive():
        start_x_stack()
    return login()
