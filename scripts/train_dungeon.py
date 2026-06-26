"""Train the whole-dungeon policy with the latest PufferLib (PuffeRL) trainer.

Wires the custom DungeonEnv into PuffeRL via GymnasiumPufferEnv + pufferlib.vector, using the
default MLP+LSTM policy. Cold-start, no demos. Logs offline by default (no wandb key on the box).

    uv run python scripts/train_dungeon.py --total-timesteps 2000000 --num-envs 128
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

import pufferlib.emulation as emulation
import pufferlib.vector as pvector
from pufferlib import pufferl
from pufferlib.ocean import torch as ocean_torch

from rotmg_rl.puffer_policy import DungeonPolicy
from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--bptt-horizon", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=None, help="override PuffeRL's tuned LR (0.015); leave unset")
    p.add_argument("--backend", choices=["serial", "multiprocessing"], default="multiprocessing")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.0)
    p.add_argument("--random-spawn-prob", type=float, default=0.0)
    p.add_argument("--boss-hp", type=float, default=7500.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true", help="passive boss target (bootstrap aim+kill)")
    p.add_argument("--init-checkpoint", default=None, help="warm-start policy from this state_dict")
    p.add_argument("--save-path", default=None, help="save the trained policy state_dict here")
    p.add_argument("--wandb", action="store_true", help="log metrics to wandb (needs `wandb login`)")
    p.add_argument("--data-dir", default="checkpoints/run", help="PuffeRL checkpoint dir (watched by follow_along)")
    args = p.parse_args()
    sys.argv = [sys.argv[0]]  # pufferl.load_config parses sys.argv; keep our args out of its way

    cfg = pufferl.load_config("default")
    t = cfg["train"]
    t["device"] = "cuda"
    t["total_timesteps"] = args.total_timesteps
    if args.learning_rate is not None:  # else keep PuffeRL's tuned default (0.015)
        t["learning_rate"] = args.learning_rate
    # DON'T override batch_size/minibatch_size/bptt_horizon: PuffeRL's defaults (minibatch 8192,
    # bptt 64, batch auto) are tuned together with LR 0.015. Overriding them froze learning.
    t["use_rnn"] = True
    t["compile"] = False
    t["data_dir"] = args.data_dir
    t["checkpoint_interval"] = 20  # save often so follow_along renders fresh policies
    cfg["wandb"] = args.wandb
    cfg["wandb_project"] = "rotmg-dungeon"
    cfg["wandb_group"] = os.environ.get("WANDB_RUN_GROUP") or "dungeon"  # PuffeRL reads this, not the env var
    cfg["neptune"] = False
    cfg["env_name"] = "rotmg_dungeon"

    print(f"CONFIG lr={t['learning_rate']} batch={t['batch_size']} minibatch={t['minibatch_size']} bptt={t['bptt_horizon']} ent_coef={t['ent_coef']} vf_coef={t['vf_coef']}", flush=True)
    backend = pvector.Multiprocessing if args.backend == "multiprocessing" else pvector.Serial
    vec_kwargs = {"num_workers": args.num_workers} if args.backend == "multiprocessing" else {}
    env_cfg = DungeonConfig(
        spawn_in_room_prob=args.spawn_in_room_prob,
        random_spawn_prob=args.random_spawn_prob,
        boss_hp_max=args.boss_hp,
        n_snakes=args.n_snakes,
        enable_grenades=not args.no_grenades,
        enable_minions=not args.no_minions,
        boss_shoots=not args.no_boss_shoots,
    )
    vecenv = pvector.make(
        emulation.GymnasiumPufferEnv,
        env_kwargs={"env_creator": partial(DungeonEnv, env_cfg)},
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )

    import torch

    policy = DungeonPolicy(vecenv.driver_env, hidden_size=args.hidden)
    policy = ocean_torch.Recurrent(vecenv.driver_env, policy, input_size=args.hidden, hidden_size=args.hidden)
    policy = policy.to("cuda")
    if args.init_checkpoint:
        policy.load_state_dict(torch.load(args.init_checkpoint, map_location="cuda"))
        print(f"warm-started from {args.init_checkpoint}", flush=True)

    pufferl.train(cfg["env_name"], args=cfg, vecenv=vecenv, policy=policy)

    if args.save_path:
        torch.save(policy.state_dict(), args.save_path)
        print(f"saved policy to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
