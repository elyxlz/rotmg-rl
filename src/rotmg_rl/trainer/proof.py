#!/usr/bin/env python3
"""Learnability proof harness for the server-as-sim training loop.

Runs the real PufferLib PPO trainer (server_train.py) against the C# server and,
in parallel, tails the C# server log for [SIM-RL] EPISODE DONE lines (ground-truth
clear/death/timeout per episode). It aligns the two by wall-clock and writes a
learning-curve CSV: step, reward, done_rate (from the trainer) and the rolling
clear_rate + episode counts (from the server log). This is the make-or-break
evidence that a policy LEARNS to clear the real Snake Pit at the easy fixed config.

Usage:
    server_proof.py --agents 32 --steps 600000 --server-log /tmp/server_sim.log \
                    --out logs/server_proof_curve.csv
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

TRAIN_RE = re.compile(
    r"step=(\d+) updates=(\d+) SPS=([\d.]+) reward=(-?[\d.]+) done_rate=([\d.]+)"
)
EP_RE = re.compile(r"EPISODE DONE t=(\d+) world=(\S+) step=(\d+) reason=(\w+)")


def tail_episodes(path: Path, episodes: deque, stop: threading.Event) -> None:
    """Tail the server log, appending (wall_s, reason) for each EPISODE DONE."""
    # wait for the file
    for _ in range(600):
        if path.exists():
            break
        time.sleep(0.1)
    with path.open("r") as f:
        f.seek(0, 2)  # start at end: only count episodes from now on
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            m = EP_RE.search(line)
            if m:
                episodes.append((time.time(), m.group(4)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=32)
    p.add_argument("--steps", type=int, default=600_000)
    p.add_argument("--server-log", default="/tmp/server_sim.log")
    p.add_argument("--out", default="logs/server_proof_curve.csv")
    p.add_argument("--window", type=float, default=20.0, help="rolling clear-rate window (s)")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes: deque = deque(maxlen=100_000)
    stop = threading.Event()
    tailer = threading.Thread(
        target=tail_episodes, args=(Path(a.server_log), episodes, stop), daemon=True
    )
    tailer.start()

    cmd = [
        ".venv/bin/python", "server_train.py",
        "--agents", str(a.agents), "--steps", str(a.steps),
    ]
    env = {"SIM_SHM_BARRIER": "1", "CUDA_VISIBLE_DEVICES": "1"}
    import os
    # pass SIM_ASYNC through so the same proof harness measures the async-overlap loop
    # (the trainer reads SIM_ASYNC; the server must be launched with it too).
    if os.environ.get("SIM_ASYNC") == "1":
        env["SIM_ASYNC"] = "1"
    full_env = dict(os.environ, **env)
    print(f"== proof: {' '.join(cmd)} (server log {a.server_log}) ==", flush=True)
    t0 = time.time()
    rows: list[str] = []
    rows.append("wall_s,step,updates,sps,reward,done_rate,clear_rate,n_clear,n_death,n_timeout,n_ep_window\n")

    proc = subprocess.Popen(cmd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    total_clear = total_death = total_timeout = 0
    last_seen_idx = 0
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = TRAIN_RE.search(line)
        if not m:
            continue
        now = time.time()
        wall = now - t0
        step, updates, sps, reward, done_rate = m.groups()
        # tally episodes in the rolling window
        snap = list(episodes)
        window = [r for (ts, r) in snap if now - ts <= a.window]
        n_clear = sum(1 for r in window if r == "clear")
        n_death = sum(1 for r in window if r == "death")
        n_timeout = sum(1 for r in window if r == "timeout")
        n_ep = len(window)
        clear_rate = n_clear / n_ep if n_ep else 0.0
        # running totals over all episodes seen so far
        for (ts, r) in snap[last_seen_idx:]:
            if r == "clear":
                total_clear += 1
            elif r == "death":
                total_death += 1
            elif r == "timeout":
                total_timeout += 1
        last_seen_idx = len(snap)
        rows.append(
            f"{wall:.1f},{step},{updates},{sps},{reward},{done_rate},"
            f"{clear_rate:.4f},{n_clear},{n_death},{n_timeout},{n_ep}\n"
        )
        out.write_text("".join(rows))  # flush each log so a kill still leaves a curve

    proc.wait()
    stop.set()
    print(
        f"== proof DONE: total episodes clear={total_clear} death={total_death} "
        f"timeout={total_timeout} -> {out} ==",
        flush=True,
    )


if __name__ == "__main__":
    main()
