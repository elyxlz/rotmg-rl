"""Get from the Nexus into the Snake Pit: /give the key, use it to open the portal, enter, and
verify the dungeon via a RELATIVE object-count jump gate.

FIX 1: the key is given with a lowercase chat line ("/give snake pit key").
FIX 6: 'in dungeon' is judged by the obs object-count jumping well above the Nexus baseline AND the
obs continuing to update -- never a fixed absolute count (absolute counts vary run to run).
FIX 8: give-key + open-portal is retried until the portal is usable, capped at GIVE_RETRY_MAX.
"""

from __future__ import annotations

import time

import config
import input as inp
import obs

GIVE_LINE = "/give snake pit key"  # FIX 1: lowercase, /give is case-insensitive


def nexus_baseline() -> int:
    """Capture the current Nexus object count to compare the post-Enter count against (FIX 6)."""
    count = obs.obs_object_count()
    return count if count is not None else 0


def give_and_open() -> bool:
    """Send the lowercase /give, then shift-click the key slot to use it and spawn the portal.

    Retries the give+use up to GIVE_RETRY_MAX (FIX 8). We cannot read the portal object directly, so
    'usable' is taken to mean the give+use sequence completed without the client dying; the real
    confirmation is the dungeon gate after Enter, which will fail and trigger a re-give if needed.
    """
    for attempt in range(1, config.GIVE_RETRY_MAX + 1):
        inp.focus_stage()  # grab Flash keyboard focus so the Shift modifier holds for the use-gesture
        inp.shift_click(*config.PORTAL_KEY_SLOT)  # use the pre-provisioned Snake Pit Key to open the portal
        time.sleep(config.KEY_USE_SETTLE_SECS)
        if obs.obs_is_fresh():
            print("[dungeon] give+open attempt %d done" % attempt, flush=True)
            return True
        print("[dungeon] give+open attempt %d: obs went stale, retrying" % attempt, flush=True)
    return False


def _wait_dungeon_gate(baseline: int, timeout: float = config.DUNGEON_GATE_TIMEOUT) -> bool:
    """FIX 6: pass only if the live object count climbs to >= baseline * DUNGEON_OBJ_JUMP while the
    obs keeps updating. A stale obs never passes."""
    threshold = max(int(baseline * config.DUNGEON_OBJ_JUMP), baseline + 1)
    deadline = time.time() + timeout
    while time.time() < deadline:
        count = obs.obs_object_count()
        if obs.obs_is_fresh() and count is not None and count >= threshold:
            print("[dungeon] entered: objs=%d (baseline=%d threshold=%d)" % (count, baseline, threshold), flush=True)
            return True
        time.sleep(config.GATE_POLL_SECS)
    print("[dungeon] gate timeout: count=%s baseline=%d threshold=%d" % (obs.obs_object_count(), baseline, threshold), flush=True)
    return False


def enter() -> bool:
    """Full Nexus->Snake Pit: capture baseline, give+open+enter, verify the dungeon gate; retry the
    whole sequence up to GIVE_RETRY_MAX. Idempotent: re-running just re-gives and re-enters."""
    baseline = nexus_baseline()
    print("[dungeon] nexus baseline objs=%d" % baseline, flush=True)
    for attempt in range(1, config.GIVE_RETRY_MAX + 1):
        if not give_and_open():
            continue
        inp.click(*config.PORTAL_ENTER)
        time.sleep(config.ENTER_SETTLE_SECS)
        if _wait_dungeon_gate(baseline):
            return True
        print("[dungeon] enter attempt %d failed the gate, retrying" % attempt, flush=True)
    print("[dungeon] failed to enter after %d attempts" % config.GIVE_RETRY_MAX, flush=True)
    return False
