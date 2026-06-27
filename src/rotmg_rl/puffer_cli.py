"""Launch a single configurable PufferLib `puffer train dungeon` run (the box.sh curriculum-iteration
path, distinct from the one-command sweep+schedule flow in rotmg_rl.training).

A thin launcher: maps our familiar knobs to 4.0's `--section.key` overrides and execs the venv
`puffer` CLI against the env compiled into _C. Three encoder paths (see pufferlib/README.md):
  --slowly (torch):  OUR DungeonEncoder CNN + renderable torch checkpoints (the recommended daily
    driver; rotmg_rl.video renders it).
  default (native _C, OUR CNN): a native CUDA port of the CNN, parity-verified but slower (im2col).
  native flat (set [torch] encoder=DefaultEncoder): puffernet's flat encoder, fast but the WRONG
    architecture for the spatial grid.

    python -m rotmg_rl.puffer_cli --total-timesteps 2000000 --slowly --wandb
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
CLONE = REPO / "pufferlib"
PUFFER = REPO / ".venv" / "bin" / "puffer"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=1024, help="-> vec.total_agents (parallel games)")
    p.add_argument("--num-workers", type=int, default=None, help="-> vec.num_threads")
    p.add_argument("--hidden", type=int, default=256, help="-> policy.hidden_size")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None, help="discount (LOWER ~0.95 cures the cautious finish)")
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--vf-coef", type=float, default=None)
    p.add_argument("--boss-hp", type=float, default=None)
    p.add_argument("--n-snakes", type=int, default=None)
    p.add_argument("--rew-boss-dmg", type=float, default=None, help="-> env.rew_boss_dmg")
    p.add_argument("--rew-clear", type=float, default=None, help="-> env.rew_clear")
    p.add_argument("--rew-death", type=float, default=None, help="-> env.rew_death")
    p.add_argument("--rew-approach", type=float, default=None, help="distance-to-boss shaping (~0.02 for nav)")
    p.add_argument("--spawn-in-room-prob", type=float, default=None)
    p.add_argument("--random-spawn-prob", type=float, default=None)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--slowly", action="store_true", help="torch backend = OUR CNN + renderable checkpoints")
    p.add_argument("--init-checkpoint", default=None, help="warm-start from this .bin (same backend only)")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="where 4.0 writes <env>/<run_id>/*.bin")
    p.add_argument("--checkpoint-interval", type=int, default=50, help="epochs between checkpoints (low = fresh videos)")
    p.add_argument("--tag", default=None, help="wandb tag")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-group", default=None, help="defaults to $WANDB_RUN_GROUP")
    args = p.parse_args()

    if not PUFFER.exists():
        sys.exit(f"{PUFFER} not found — run scripts/setup.sh first")

    cmd = [str(PUFFER), "train", "dungeon"]
    cmd += ["--train.total-timesteps", str(args.total_timesteps)]
    cmd += ["--vec.total-agents", str(args.num_envs)]
    cmd += ["--policy.hidden-size", str(args.hidden)]  # the LSTM network is sized from policy.hidden_size
    # absolute checkpoint dir: puffer runs with cwd=CLONE, so a relative path would land in pufferlib/
    ckpt_dir = args.checkpoint_dir if os.path.isabs(args.checkpoint_dir) else str(REPO / args.checkpoint_dir)
    cmd += ["--checkpoint-dir", ckpt_dir, "--checkpoint-interval", str(args.checkpoint_interval)]
    if args.num_workers is not None:
        cmd += ["--vec.num-threads", str(args.num_workers)]
    if args.slowly:
        cmd += ["--slowly"]
    if args.init_checkpoint is not None:
        cmd += ["--load-model-path", args.init_checkpoint]
    if args.learning_rate is not None:
        cmd += ["--train.learning-rate", str(args.learning_rate)]
    if args.ent_coef is not None:
        cmd += ["--train.ent-coef", str(args.ent_coef)]
    if args.gamma is not None:
        cmd += ["--train.gamma", str(args.gamma)]
    if args.gae_lambda is not None:
        cmd += ["--train.gae-lambda", str(args.gae_lambda)]
    if args.vf_coef is not None:
        cmd += ["--train.vf-coef", str(args.vf_coef)]
    if args.boss_hp is not None:
        cmd += ["--env.boss-hp-max", str(args.boss_hp)]
    if args.n_snakes is not None:
        cmd += ["--env.n-snakes", str(args.n_snakes)]
    if args.rew_boss_dmg is not None:
        cmd += ["--env.rew-boss-dmg", str(args.rew_boss_dmg)]
    if args.rew_clear is not None:
        cmd += ["--env.rew-clear", str(args.rew_clear)]
    if args.rew_death is not None:
        cmd += ["--env.rew-death", str(args.rew_death)]
    if args.rew_approach is not None:
        cmd += ["--env.rew-approach", str(args.rew_approach)]
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
    if args.tag is not None:
        cmd += ["--tag", args.tag]
    if args.wandb:
        group = args.wandb_group or os.environ.get("WANDB_RUN_GROUP") or "dungeon"
        cmd += ["--wandb", "--wandb-project", "rotmg-dungeon", "--wandb-group", group]

    print("exec:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.run(cmd, cwd=str(CLONE)).returncode)


if __name__ == "__main__":
    main()
