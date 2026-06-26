"""Train the whole-dungeon policy with the latest PufferLib (PuffeRL) trainer.

Wires the custom DungeonEnv into PuffeRL via GymnasiumPufferEnv + pufferlib.vector, using the
default MLP+LSTM policy. Cold-start, no demos. Logs offline by default (no wandb key on the box).

    uv run python scripts/train_dungeon.py --total-timesteps 2000000 --num-envs 128
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--backend", choices=["serial", "multiprocessing"], default="multiprocessing")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.5, help="curriculum: prob of spawning in the boss room")
    args = p.parse_args()
    sys.argv = [sys.argv[0]]  # pufferl.load_config parses sys.argv; keep our args out of its way

    cfg = pufferl.load_config("default")
    t = cfg["train"]
    t["device"] = "cuda"
    t["total_timesteps"] = args.total_timesteps
    t["learning_rate"] = args.learning_rate
    t["bptt_horizon"] = args.bptt_horizon
    t["batch_size"] = args.num_envs * args.bptt_horizon * 4
    t["minibatch_size"] = args.num_envs * args.bptt_horizon
    t["use_rnn"] = True
    t["compile"] = False
    cfg["wandb"] = False
    cfg["neptune"] = False
    cfg["env_name"] = "rotmg_dungeon"

    backend = pvector.Multiprocessing if args.backend == "multiprocessing" else pvector.Serial
    vec_kwargs = {"num_workers": args.num_workers} if args.backend == "multiprocessing" else {}
    env_cfg = DungeonConfig(spawn_in_room_prob=args.spawn_in_room_prob)
    vecenv = pvector.make(
        emulation.GymnasiumPufferEnv,
        env_kwargs={"env_creator": partial(DungeonEnv, env_cfg)},
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )

    policy = DungeonPolicy(vecenv.driver_env, hidden_size=args.hidden)
    policy = ocean_torch.Recurrent(vecenv.driver_env, policy, input_size=args.hidden, hidden_size=args.hidden)
    policy = policy.to("cuda")

    pufferl.train(cfg["env_name"], args=cfg, vecenv=vecenv, policy=policy)


if __name__ == "__main__":
    main()
