"""Single-core throughput benchmark for the numpy Snake Pit env.

Informs the C-port decision: the goal targets >=1M steps/s/core, which numpy will not hit.
This measures the reference baseline so the speedup from the C port is quantified.

    uv run python scripts/bench_env.py --steps 100000
"""

import argparse
import time

from rotmg_rl.sim.snakepit import SnakePitEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    env = SnakePitEnv()
    env.reset(seed=0)
    actions = [env.action_space.sample() for _ in range(1000)]

    start = time.perf_counter()
    done_count = 0
    for i in range(args.steps):
        _, _, terminated, truncated, _ = env.step(actions[i % 1000])
        if terminated or truncated:
            env.reset()
            done_count += 1
    elapsed = time.perf_counter() - start

    sps = args.steps / elapsed
    print(f"{args.steps} steps in {elapsed:.2f}s -> {sps:,.0f} steps/s/core ({done_count} episodes)")


if __name__ == "__main__":
    main()
