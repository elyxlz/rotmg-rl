"""M0 smoke test: prove the wandb logging path works end to end.

Logs a synthetic rising-with-noise curve so the dashboard shows a real time series.
Run online (needs WANDB_API_KEY) or offline (WANDB_MODE=offline) to validate the pipeline
without a key.

    uv run python scripts/smoke_wandb.py --steps 200
    WANDB_MODE=offline uv run python scripts/smoke_wandb.py --steps 200
"""

import argparse
import os
import random

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--project", default="rotmg-rl")
    parser.add_argument("--name", default="m0-smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    run = wandb.init(project=args.project, name=args.name, config=vars(args))

    reward = 0.0
    for step in range(args.steps):
        # Synthetic learning curve: asymptotes toward 1.0 with noise. Proves logging only.
        reward = reward + (1.0 - reward) * 0.02 + random.uniform(-0.02, 0.02)
        wandb.log({"smoke/reward": reward}, step=step)

    run.finish()
    mode = os.environ.get("WANDB_MODE", "online")
    print(f"logged {args.steps} steps to wandb (mode={mode})")


if __name__ == "__main__":
    main()
