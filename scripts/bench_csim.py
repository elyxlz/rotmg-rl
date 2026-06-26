"""Throughput benchmark for the C dungeon env (vectorized, single process).

Mirrors the PufferLib Ocean squared bench: one CDungeon owns `num_envs` C envs stepped in a tight
C loop. Reports aggregate steps/s. Compare against the numpy baseline (scripts/bench_env.py,
~20K SPS).

    uv run python scripts/bench_csim.py --num-envs 4096 --seconds 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rotmg_rl.csim.dungeon import CDungeon
from rotmg_rl.sim.dungeon import DungeonConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-boss-shoots", action="store_true")
    args = p.parse_args()

    cfg = DungeonConfig(n_snakes=args.n_snakes, boss_shoots=not args.no_boss_shoots)
    env = CDungeon(cfg, num_envs=args.num_envs)
    env.reset(seed=0)

    cache = 1024
    actions = np.random.randint(0, [9, 32, 2, 2], size=(cache, args.num_envs, 4)).astype(np.int32)

    steps = 0
    i = 0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        env.step(actions[i % cache])
        steps += args.num_envs
        i += 1
    elapsed = time.perf_counter() - start
    env.close()

    sps = steps / elapsed
    print(f"C env: {args.num_envs} envs, n_snakes={args.n_snakes}, boss_shoots={not args.no_boss_shoots}")
    print(f"{steps:,} steps in {elapsed:.2f}s -> {sps:,.0f} steps/s aggregate")


if __name__ == "__main__":
    main()
