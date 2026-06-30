"""Load the trained checkpoint and drive the client from the live obs. Run as a subprocess
(`python policy.py`) so its GPU/torch lifetime is isolated from the orchestrator.

It loads PolicyRunner (rotmg_rl.deploy.policy) on POLICY_GPU, then loops: read OBS_PATH, sample an
action, apply it via input.apply_action, ~10 Hz. torch + python-xlib are GPU-box-only deps, imported
lazily so this module import-checks off-box.

  python policy.py            drive until killed
  python policy.py --test     load + emit one action from the current obs, then exit (no driving)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

import config

os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.POLICY_GPU)


def _load_runner():
    sys.path.insert(0, str(config.POLICY_SRC))
    from rotmg_rl.deploy.policy import PolicyRunner

    runner = PolicyRunner(str(config.CKPT), hidden=config.HIDDEN, num_layers=config.NUM_LAYERS, device="cuda")
    print("[policy] loaded ckpt=%s hidden=%d layers=%d" % (config.CKPT, config.HIDDEN, config.NUM_LAYERS), flush=True)
    return runner


def _read_obs() -> np.ndarray | None:
    if not config.OBS_PATH.exists():
        return None
    data = np.fromfile(config.OBS_PATH, dtype=np.float32)
    if data.size != config.OBS_FLOATS:
        return None
    return data


def selftest() -> None:
    runner = _load_runner()
    for _ in range(40):
        data = _read_obs()
        if data is not None:
            action = runner.act(data, greedy=False)
            print("[policy] OBS OK nonzero=%d action=%s" % (int((data != 0).sum()), action), flush=True)
            return
        time.sleep(0.2)
    print("[policy] no valid obs at %s" % config.OBS_PATH, flush=True)


def drive() -> None:
    runner = _load_runner()
    runner.reset()
    import input as inp  # lazy: python-xlib is GPU-box-only
    from rotmg_rl.config import MOVE_DIRS  # world-space move directions the policy was trained on
    trace = open("/dev/shm/rotmg_trace.txt", "w")

    print("[policy] driving", flush=True)
    ticks = 0
    try:
        while True:
            data = _read_obs()
            if data is None:
                time.sleep(0.02)
                continue
            action = runner.act(data, greedy=False)
            inp.apply_action(action["move"], action["aim"], action["shoot"], action["cast"])
            mv = action["move"]
            mdx, mdy = (0.0, 0.0) if mv == 0 else (float(MOVE_DIRS[mv - 1][0]), float(MOVE_DIRS[mv - 1][1]))
            try:
                _px, _py = open("/dev/shm/rotmg_pos.txt").read().split()
            except Exception:
                _px, _py = "nan", "nan"
            trace.write("%d %d %.3f %.3f %s %s\n" % (ticks, mv, mdx, mdy, _px, _py))
            trace.flush()
            ticks += 1
            if ticks % 20 == 0:
                print("[policy] tick=%d action=%s" % (ticks, action), flush=True)
            time.sleep(0.1)
    finally:
        inp.release_all()


if __name__ == "__main__":
    if "--test" in sys.argv:
        selftest()
    else:
        drive()
