"""Train the dungeon env on PufferLib 4.0's native trainer.

4.0 trains via `puffer train <env>` against the env compiled into its _C backend (see
scripts/setup_box_puffer4.sh, which bakes our env into the pinned clone at .pufferlib4). This is a
thin launcher: it maps our familiar knobs to 4.0's `--section.key` overrides and execs the .venv4
`puffer` CLI inside the clone. The 3.0 path stays `scripts/train_dungeon.py` (untouched).

    .venv4/bin/python scripts/train_dungeon4.py --total-timesteps 2000000 --num-envs 1024

(Runs with any Python; it only subprocess-execs the .venv4 puffer binary.)
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
CLONE = REPO / ".pufferlib4"
PUFFER = REPO / ".venv4" / "bin" / "puffer"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=1024, help="-> vec.total_agents (parallel games)")
    p.add_argument("--hidden", type=int, default=256, help="-> policy.hidden_size")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--boss-hp", type=float, default=None)
    p.add_argument("--n-snakes", type=int, default=None)
    p.add_argument("--spawn-in-room-prob", type=float, default=None)
    p.add_argument("--random-spawn-prob", type=float, default=None)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    if not PUFFER.exists():
        sys.exit(f"{PUFFER} not found — run scripts/setup_box_puffer4.sh first")

    cmd = [str(PUFFER), "train", "dungeon"]
    cmd += ["--train.total-timesteps", str(args.total_timesteps)]
    cmd += ["--vec.total-agents", str(args.num_envs)]
    cmd += ["--policy.hidden-size", str(args.hidden)]  # the LSTM network is sized from policy.hidden_size
    if args.learning_rate is not None:
        cmd += ["--train.learning-rate", str(args.learning_rate)]
    if args.ent_coef is not None:
        cmd += ["--train.ent-coef", str(args.ent_coef)]
    if args.boss_hp is not None:
        cmd += ["--env.boss-hp-max", str(args.boss_hp)]
    if args.n_snakes is not None:
        cmd += ["--env.n-snakes", str(args.n_snakes)]
    if args.spawn_in_room_prob is not None:
        cmd += ["--env.spawn-in-room-prob", str(args.spawn_in_room_prob)]
    if args.random_spawn_prob is not None:
        cmd += ["--env.random-spawn-prob", str(args.random_spawn_prob)]
    if args.no_grenades:
        cmd += ["--env.enable-grenades", "0"]
    if args.no_minions:
        cmd += ["--env.enable-minions", "0"]
    if args.no_boss_shoots:
        cmd += ["--env.boss-shoots", "0"]
    if args.wandb:
        cmd += ["--wandb", "--wandb-project", "rotmg-dungeon"]

    print("exec:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=str(CLONE)).returncode)


if __name__ == "__main__":
    main()
