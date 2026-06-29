#!/usr/bin/env python3
"""Server-as-sim EVAL LADDER -> curriculum-depth (the Protein objective for the sweep).

For each curriculum rung d in 0.1..1.0, set the LIVE difficulty config to that rung (over the shm
d-flow channel), drive the trained policy for a step budget, and count the ground-truth
clear/death/timeout episodes from the C# server's `[SIM-RL] EPISODE DONE reason=` log lines. The
per-rung clear rate feeds curriculum_depth() (reused UNCHANGED from schedule.py via server_difficulty)
-> one depth number in [0,1] = how far up the difficulty ladder the policy still clears >=50%.

Unlike the C-sim eval (CDungeonSingle, a Python-driven per-episode reset), the server owns episode
boundaries: episodes auto-reset inside the C# loop on done, and the GROUND TRUTH is the server log,
not a Python info dict. So the ladder drives the SAME PuffeRL eval-rollout path training uses (the
policy's forward_eval through the gated server_env) for a fixed budget per rung and reads the clears
the server reports in that window -- the same EPISODE DONE lines server_proof.py tallies.

Usage (the server must be up in server-as-sim mode; SIM_SHM_BARRIER must match):
    SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m server_eval \
        --checkpoint checkpoints/server_sim.pt --agents 32 \
        --server-log /tmp/server_sim.log --episodes-per-rung 12
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch  # noqa: F401  # torch first so its cudart loads before pufferlib's _C
from pufferlib import _C, pufferl
from pufferlib.torch_pufferl import PuffeRL, load_policy

from rotmg_rl.schedule import CURRICULUM_RUNGS
from rotmg_rl.server_difficulty import curriculum_depth, server_difficulty_config
from server_shm_config import ShmConfigChannel

EP_RE = re.compile(r"EPISODE DONE t=(\d+) world=(\S+) step=(\d+) reason=(\w+)")


def tail_episodes(path: Path, episodes: deque, stop: threading.Event) -> None:
    """Tail the server log, appending (wall_s, reason) for each EPISODE DONE (== server_proof.py)."""
    for _ in range(600):
        if path.exists():
            break
        time.sleep(0.1)
    with path.open("r") as f:
        f.seek(0, 2)
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.02)
                continue
            m = EP_RE.search(line)
            if m:
                episodes.append((time.time(), m.group(4)))


def build_trainer(checkpoint: str, agents: int, redis_port: int, redis_db: int):
    """Build the SAME PuffeRL + DungeonEncoder used for training, with the policy loaded from the
    checkpoint, pointed at the gated server_env. We only ever call rollouts() (eval-mode driving)."""
    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = agents
    args["env"]["redis_port"] = redis_port
    args["env"]["redis_db"] = redis_db
    if checkpoint:
        args["load_model_path"] = checkpoint  # load_policy reads this
    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    policy.eval()
    trainer = PuffeRL(args, vec, policy, verbose=False)
    return args, trainer


def run_ladder(
    checkpoint: str,
    agents: int,
    server_log: str,
    episodes_per_rung: int = 12,
    rungs: tuple[float, ...] = CURRICULUM_RUNGS,
    redis_port: int = 6390,
    redis_db: int = 5,
    settle_episodes: int = 4,
    max_rollouts_per_rung: int = 400,
) -> dict[float, float]:
    """Drive the trained policy at each rung; return {d: clear_rate}. Per rung: write the rung's
    d-config to shm, run eval rollouts until `episodes_per_rung` fresh episodes complete (or the
    rollout cap), tally clears. The first `settle_episodes`*agents episodes after a d-change are
    DISCARDED -- worlds mid-episode at the old config when d flips don't reflect the new rung; we
    count only episodes that started under the new d."""
    args, trainer = build_trainer(checkpoint, agents, redis_port, redis_db)

    chan = ShmConfigChannel(agents)
    episodes: deque = deque(maxlen=500_000)
    stop = threading.Event()
    tailer = threading.Thread(target=tail_episodes, args=(Path(server_log), episodes, stop), daemon=True)
    tailer.start()
    time.sleep(0.3)  # let the tailer reach the log tail

    rates: dict[float, float] = {}
    try:
        for d in rungs:
            cfg = server_difficulty_config(d)
            chan.write(cfg)
            print(f"[ladder] d={d:.2f} -> cfg={cfg}", flush=True)

            # settle: drive a few rollouts so in-flight old-config episodes drain.
            settle_target = len(episodes) + settle_episodes * agents
            t_settle = time.time()
            while len(episodes) < settle_target and time.time() - t_settle < 60:
                trainer.rollouts()

            mark = len(episodes)
            for _ in range(max_rollouts_per_rung):
                trainer.rollouts()
                if len(episodes) - mark >= episodes_per_rung:
                    break

            window = list(episodes)[mark:]
            n_clear = sum(1 for (_, r) in window if r == "clear")
            n_ep = len(window)
            rate = n_clear / n_ep if n_ep else 0.0
            rates[d] = rate
            print(f"[ladder] d={d:.2f} clear_rate={rate:.3f} ({n_clear}/{n_ep} episodes)", flush=True)
    finally:
        stop.set()
        chan.close()
        trainer.close()
    return rates


def main() -> None:
    p = argparse.ArgumentParser(description="Server-as-sim eval ladder -> curriculum depth.")
    p.add_argument("--checkpoint", default="checkpoints/server_sim.pt")
    p.add_argument("--agents", type=int, default=32)
    p.add_argument("--server-log", default="/tmp/server_sim.log")
    p.add_argument("--episodes-per-rung", type=int, default=12)
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    p.add_argument("--out", default="logs/server_ladder.csv")
    a = p.parse_args()

    rates = run_ladder(
        a.checkpoint, a.agents, a.server_log,
        episodes_per_rung=a.episodes_per_rung, redis_port=a.redis_port, redis_db=a.redis_db,
    )
    depth = curriculum_depth(rates)
    print("\n=== EVAL LADDER (server-as-sim) ===", flush=True)
    for d in sorted(rates):
        bar = "#" * int(rates[d] * 30)
        print(f"  d={d:.2f}  clear={rates[d]:.3f}  {bar}", flush=True)
    print(f"  CURRICULUM DEPTH = {depth:.4f}   (Protein objective)", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["d,clear_rate\n"] + [f"{d},{rates[d]}\n" for d in sorted(rates)] + [f"depth,{depth}\n"]
    out.write_text("".join(lines))
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
