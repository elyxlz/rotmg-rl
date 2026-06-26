"""Run the champion through the full deploy bridge and report clear-rate (M6 readiness).

    uv run --extra train python scripts/deploy_loop.py --checkpoint checkpoints/m3-final.pt --episodes 100
"""

import argparse

from rotmg_rl.deploy.loop import run_episode
from rotmg_rl.deploy.policy_server import PolicyRunner
from rotmg_rl.sim.snakepit import SnakePitConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--boss-hp", type=float, default=250.0)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()

    runner = PolicyRunner(args.checkpoint)
    cfg = SnakePitConfig(boss_hp_max=args.boss_hp, randomize=args.randomize)
    clears = sum(run_episode(runner, cfg, seed=10_000 + i) for i in range(args.episodes))
    print(f"DEPLOY-PATH clear_rate {clears / args.episodes:.3f} over {args.episodes} eps (boss_hp={args.boss_hp})")


if __name__ == "__main__":
    main()
