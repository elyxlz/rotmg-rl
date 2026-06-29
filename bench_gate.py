#!/usr/bin/env python3
"""Pure gate/barrier throughput probe (no policy, no GPU, no PPO). Drives N worlds one
tick at a time as fast as the sync primitive allows and reports ticks/sec * N == aggregate
SPS. Isolates the barrier-vs-redis cost from the trainer's GPU stalls.

    SIM_SHM_BARRIER=1 .venv/bin/python bench_gate.py --agents 16 --ticks 5000   # shm barrier
    .venv/bin/python bench_gate.py --agents 16 --ticks 5000                     # redis gate
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import socket
import time

import numpy as np

OBS_LEN = 9807
HEADER = 16


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--ticks", type=int, default=5000)
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    a = p.parse_args()
    shm_path = os.environ.get("SIM_SHM_PATH", "/dev/shm/rotmg_sim_shm")
    n = a.agents
    use_barrier = os.environ.get("SIM_SHM_BARRIER", "0") == "1"

    raw = np.memmap(shm_path, dtype=np.uint8, mode="r+")
    act = np.frombuffer(raw, dtype=np.float32, count=n * 4, offset=HEADER + n * OBS_LEN * 4).reshape(n, 4)
    act[:] = 0.0

    if use_barrier:
        ctrl_off = HEADER + n * OBS_LEN * 4 + n * 4 * 4 + n * 4 + n * 4
        ctrl = np.frombuffer(raw, dtype=np.int32, count=2, offset=ctrl_off)
        ctrl_addr = raw.ctypes.data + ctrl_off
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        SYS_futex, FUTEX_WAIT, FUTEX_WAKE = 202, 0, 1
        gen = int(ctrl[0])

        def tick():
            nonlocal gen
            gen += 1
            ctrl[0] = gen
            libc.syscall(SYS_futex, ctypes.c_void_p(ctrl_addr), FUTEX_WAKE, 1, None, None, 0)
            done_addr = ctrl_addr + 4
            while int(ctrl[1]) < gen:
                cur = int(ctrl[1])
                if cur >= gen:
                    break
                libc.syscall(SYS_futex, ctypes.c_void_p(done_addr), FUTEX_WAIT, cur, None, None, 0)
        label = "shm-barrier"
    else:
        s = socket.create_connection(("127.0.0.1", a.redis_port))
        s.sendall(f"*2\r\n$6\r\nSELECT\r\n$1\r\n{a.redis_db}\r\n".encode())
        s.recv(64)

        def tick():
            s.sendall(b"*3\r\n$5\r\nLPUSH\r\n$12\r\nsim:step:cmd\r\n$1\r\n1\r\n")
            s.recv(64)
            s.sendall(b"*3\r\n$5\r\nBLPOP\r\n$12\r\nsim:step:ack\r\n$1\r\n0\r\n")
            buf = b""
            while buf.count(b"\r\n") < 5:
                chunk = s.recv(256)
                if not chunk:
                    break
                buf += chunk
        label = "redis-gate"

    # warm up (let agents spawn)
    for _ in range(200):
        tick()
    t0 = time.perf_counter()
    for _ in range(a.ticks):
        tick()
    dt = time.perf_counter() - t0
    tps = a.ticks / dt
    print(f"[{label}] N={n}  ticks={a.ticks}  wall={dt:.3f}s  ticks/s={tps:.0f}  aggregate_SPS={tps * n:.0f}  per_tick_ms={dt / a.ticks * 1000:.3f}")


if __name__ == "__main__":
    main()
