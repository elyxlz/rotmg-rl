#!/usr/bin/env python3
"""Prove the policy's actions MOVE the real agents in-world. Drives the lockstep gate
DIRECTLY (no pufferl) with a forced 'move east + shoot' action written into the shm action
slots, then watches the fog-of-war minimap: the player cell shifts and the discovered-tile
count grows only if the agent actually walks. If the action path were broken (no apply /
wrong slot) the minimap player cell would never move. Run against a live SIM_SHM server.

    .venv/bin/python verify_motion.py --agents 16 --ticks 400
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import socket
import sys

import numpy as np

OBS_LEN = 9807
GRID_FLAT = 7 * 31 * 31
MM = 32
MM_PLAYER_BASE = GRID_FLAT + 1 * MM * MM  # minimap channel 1 = player cell
HEADER = 4 * 4


def mm_player_cell(row: np.ndarray) -> int:
    block = row[MM_PLAYER_BASE : MM_PLAYER_BASE + MM * MM]
    nz = np.nonzero(block)[0]
    return int(nz[0]) if nz.size else -1


def discovered_count(row: np.ndarray) -> int:
    block = row[GRID_FLAT : GRID_FLAT + MM * MM]  # minimap terrain channel 0
    return int(np.count_nonzero(block))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    a = p.parse_args()
    shm_path = os.environ.get("SIM_SHM_PATH", "/dev/shm/rotmg_sim_shm")
    n = a.agents

    raw = np.memmap(shm_path, dtype=np.uint8, mode="r+")
    hdr = np.frombuffer(raw, dtype=np.int32, count=4, offset=0)
    assert hdr[1] == n and hdr[2] == OBS_LEN, f"shm header n={hdr[1]} obs={hdr[2]} != {n}/{OBS_LEN}"
    obs = np.frombuffer(raw, dtype=np.float32, count=n * OBS_LEN, offset=HEADER).reshape(n, OBS_LEN)
    act = np.frombuffer(raw, dtype=np.float32, count=n * 4, offset=HEADER + n * OBS_LEN * 4).reshape(n, 4)

    use_barrier = os.environ.get("SIM_SHM_BARRIER", "0") == "1"

    if use_barrier:
        # Pure-shm futex barrier: bump req, FUTEX_WAKE it, wait until done catches up.
        # The two ctrl int32s sit at the region tail (after [obs][act][rew][done]).
        ctrl_off = HEADER + n * OBS_LEN * 4 + n * 4 * 4 + n * 4 + n * 4
        ctrl = np.frombuffer(raw, dtype=np.int32, count=2, offset=ctrl_off)
        ctrl_addr = raw.ctypes.data + ctrl_off
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        SYS_futex, FUTEX_WAIT, FUTEX_WAKE = 202, 0, 1
        gen = [int(ctrl[0])]

        def gate_tick(_tok: int):
            gen[0] += 1
            ctrl[0] = gen[0]  # req
            libc.syscall(SYS_futex, ctypes.c_void_p(ctrl_addr), FUTEX_WAKE, 1, None, None, 0)
            done_addr = ctrl_addr + 4
            while int(ctrl[1]) < gen[0]:
                cur = int(ctrl[1])
                if cur >= gen[0]:
                    break
                libc.syscall(SYS_futex, ctypes.c_void_p(done_addr), FUTEX_WAIT, cur, None, None, 0)
    else:
        s = socket.create_connection(("127.0.0.1", a.redis_port))
        s.sendall(f"*2\r\n$6\r\nSELECT\r\n$1\r\n{a.redis_db}\r\n".encode())
        s.recv(64)

        def gate_tick(tok: int):
            s.sendall(f"*3\r\n$5\r\nLPUSH\r\n$12\r\nsim:step:cmd\r\n${len(str(tok))}\r\n{tok}\r\n".encode())
            s.recv(64)
            s.sendall(b"*3\r\n$5\r\nBLPOP\r\n$12\r\nsim:step:ack\r\n$1\r\n0\r\n")
            buf = b""
            while buf.count(b"\r\n") < 5:
                chunk = s.recv(256)
                if not chunk:
                    break
                buf += chunk

    start = None
    for t in range(a.ticks):
        act[:] = 0.0
        act[:, 0] = 1.0  # move dir 1 (east)
        act[:, 2] = 1.0  # shoot
        gate_tick(1000 + t)
        if t == 30:  # after agents have spawned + a few steps
            start = [(mm_player_cell(obs[i].copy()), discovered_count(obs[i].copy())) for i in range(n)]

    end = [(mm_player_cell(obs[i].copy()), discovered_count(obs[i].copy())) for i in range(n)]

    print("\n== motion proof: forced 'move east' over the gate -> agents walk in-world ==", flush=True)
    moved = 0
    for i in range(n):
        c0, d0 = start[i]
        c1, d1 = end[i]
        m = c0 != c1 or d1 > d0
        moved += int(m)
        if i < 8 or m != (c0 != c1 or d1 > d0):
            print(f"  agent {i}: player_mm_cell {c0}->{c1}  discovered_tiles {d0}->{d1}  {'MOVED' if m else 'static'}", flush=True)
    print(f"\nRESULT: {moved}/{n} agents MOVED in-world from the forced action (obs evolves -> actions reach the real game)", flush=True)
    if not use_barrier:
        s.close()
    sys.exit(0 if moved > 0 else 1)


if __name__ == "__main__":
    main()
