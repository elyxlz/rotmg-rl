#!/usr/bin/env python3
"""Server-as-sim training entry: run a real PufferLib PPO loop against the throwaway
betterSkillys C# server through the server_env C-shim (shm + redis lockstep gate).

This is the SAME PuffeRL trainer + DungeonEncoder CNN-LSTM policy the dungeon flow uses
-- only the env binding changes (server_env instead of dungeon). The N policy agents map
1:1 onto the N Snake Pit worlds the C# server runs; obs/action/reward/done cross the shm
boundary, ticks cross the redis gate.

    # 1. start the C# server with SIM_SHM=1 SIM_WORLDS=N (see run-server-sim.sh)
    # 2. CUDA_VISIBLE_DEVICES=1 .venv/bin/python server_train.py --agents N --steps 20000

The agent count MUST match the launched SIM_WORLDS (the shm region is sized for it; the
binding hard-fails on a header mismatch).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch  # noqa: F401  # import torch FIRST so its cudart loads before pufferlib's _C
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy


def main() -> None:
    p = argparse.ArgumentParser(description="PPO against the C# server via server_env (shm + lockstep gate).")
    p.add_argument("--agents", type=int, default=4, help="N: must equal the server's SIM_WORLDS")
    p.add_argument("--steps", type=int, default=20_000, help="env steps (per agent) to run")
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    p.add_argument("--out", default="checkpoints/server_sim.pt")
    a = p.parse_args()

    # load_config parses sys.argv (an argparse over every ini key); clear our flags first.
    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    args["env"]["redis_port"] = a.redis_port
    args["env"]["redis_db"] = a.redis_db
    args["train"]["total_timesteps"] = a.steps * a.agents

    print(f"== server-as-sim PPO: N={a.agents} agents, ~{a.steps} steps/agent, redis 127.0.0.1:{a.redis_port} db{a.redis_db} ==", flush=True)
    print(f"   gpu={_C.gpu}  obs_size(server_env)={args['env']}", flush=True)

    vec = _C.create_vec(args, _C.gpu)  # connects to the C# server, ticks the gate once (c_reset)
    print(f"   vec ready: total_agents={vec.total_agents} obs_size={vec.obs_size} act_sizes={vec.act_sizes} dtype={vec.obs_dtype}", flush=True)

    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    horizon = args["train"]["horizon"]
    total = a.steps * a.agents
    last_log = 0.0
    updates = 0
    while trainer.global_step < total:
        trainer.rollouts()  # horizon gated ticks: obs -> policy -> action -> shm -> gate -> reward/done
        trainer.train()     # the PPO update (loss + grad step)
        updates += 1
        if time.time() - last_log > 1.0 or trainer.global_step >= total:
            last_log = time.time()
            flat = dict(unroll_nested_dict(trainer.log()))
            print(
                f"[server-sim] step={trainer.global_step} updates={updates} "
                f"SPS={flat.get('SPS', 0):.0f} reward={flat.get('env/reward', 0):.4f} "
                f"done_rate={flat.get('env/done_rate', 0):.4f} "
                f"perf[roll={flat.get('perf/rollout',0):.3f} gpu={flat.get('perf/eval_gpu',0):.3f} "
                f"env={flat.get('perf/eval_env',0):.3f} train={flat.get('perf/train',0):.3f}]",
                flush=True,
            )

    trainer.save_weights(a.out)
    print(f"== DONE: {updates} PPO updates, {trainer.global_step} steps -> {a.out} ==", flush=True)
    trainer.close()


if __name__ == "__main__":
    main()
