#!/usr/bin/env python3
"""Full-fight gate throughput probe. Like bench_gate.py but SHOOTS every tick (action
head 2 = shoot = 1) so BOTH collision directions are exercised at the full storm, and
aims around so agent bullets spread across enemies. Spawn the server with
SIM_SPAWN_GEO_DIST=0 SIM_RL_INVULN=1 so the agent sits at the boss (all enemies active)
and survives the bullet storm for a steady-state measurement."""
from __future__ import annotations
import argparse, ctypes, ctypes.util, os, time
import numpy as np

OBS_LEN = 9807
HEADER = 16

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--ticks", type=int, default=4000)
    p.add_argument("--warmup", type=int, default=600)
    p.add_argument("--shoot", type=int, default=1)
    a = p.parse_args()
    shm_path = os.environ.get("SIM_SHM_PATH", "/dev/shm/rotmg_sim_shm")
    n = a.agents

    raw = np.memmap(shm_path, dtype=np.uint8, mode="r+")
    act = np.frombuffer(raw, dtype=np.float32, count=n * 4, offset=HEADER + n * OBS_LEN * 4).reshape(n, 4)
    # move=0, aim rotates, shoot=1, cast=0
    act[:] = 0.0
    if a.shoot:
        act[:, 2] = 1.0

    ctrl_off = HEADER + n * OBS_LEN * 4 + n * 4 * 4 + n * 4 + n * 4
    ctrl = np.frombuffer(raw, dtype=np.int32, count=2, offset=ctrl_off)
    ctrl_addr = raw.ctypes.data + ctrl_off
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    SYS_futex, FUTEX_WAIT, FUTEX_WAKE = 202, 0, 1
    gen = int(ctrl[0])
    aim = 0

    def tick():
        nonlocal gen, aim
        if a.shoot:
            aim = (aim + 1) % 32
            act[:, 1] = float(aim)
        gen += 1
        ctrl[0] = gen
        libc.syscall(SYS_futex, ctypes.c_void_p(ctrl_addr), FUTEX_WAKE, 1, None, None, 0)
        done_addr = ctrl_addr + 4
        while int(ctrl[1]) < gen:
            cur = int(ctrl[1])
            if cur >= gen:
                break
            libc.syscall(SYS_futex, ctypes.c_void_p(done_addr), FUTEX_WAIT, cur, None, None, 0)

    for _ in range(a.warmup):
        tick()
    t0 = time.perf_counter()
    for _ in range(a.ticks):
        tick()
    dt = time.perf_counter() - t0
    tps = a.ticks / dt
    print(f"[fight shoot={a.shoot}] N={n} ticks={a.ticks} wall={dt:.3f}s ticks/s={tps:.0f} aggregate_SPS={tps*n:.0f} per_tick_ms={dt/a.ticks*1000:.3f}")

if __name__ == "__main__":
    main()
