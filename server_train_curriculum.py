#!/usr/bin/env python3
"""Server-as-sim CURRICULUM training: the same PuffeRL PPO loop as server_train.py, but d RAMPS over
the run (cosine difficulty_at) and the d-derived difficulty config is written into shm each update, so
the C# server spawns the d-appropriate episode LIVE (no restart). This is the curriculum on top of the
proven per-episode loop -- the policy climbs from the easy anchor (d=0) toward the real conditions (d=1).

    # 1. start the C# server in server-as-sim mode (SIM_SHM=1 SIM_WORLDS=N; see run-server-sim.sh)
    # 2. SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python server_train_curriculum.py \
    #        --agents 32 --steps 1500000 --ramp-frac 0.6

d reaches the shm config block via ShmConfigChannel; the server's SimRlLoop reads it at every
spawn/reset. The applied-config-vs-d proof is in the server log ([SIM-RL] agent spawned ... applied cfg).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch  # noqa: F401  # torch first so its cudart loads before pufferlib's _C
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy

from rotmg_rl.server_difficulty import difficulty_at, server_difficulty_config
from server_shm_config import ShmConfigChannel


def main() -> None:
    p = argparse.ArgumentParser(description="Curriculum PPO against the C# server via server_env (d ramps live).")
    p.add_argument("--agents", type=int, default=32, help="N: must equal the server's SIM_WORLDS")
    p.add_argument("--steps", type=int, default=1_500_000, help="env steps (per agent) to run")
    p.add_argument("--ramp-frac", type=float, default=0.6, help="fraction of the run over which d ramps 0->1")
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    p.add_argument("--out", default="checkpoints/server_curriculum.pt")
    a = p.parse_args()

    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    args["env"]["redis_port"] = a.redis_port
    args["env"]["redis_db"] = a.redis_db
    args["train"]["total_timesteps"] = a.steps * a.agents

    print(f"== server-as-sim CURRICULUM PPO: N={a.agents}, ~{a.steps} steps/agent, ramp_frac={a.ramp_frac} ==", flush=True)
    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    chan = ShmConfigChannel(a.agents)
    total = a.steps * a.agents
    # seed d=0 before the first rollout so the very first episodes spawn at the easy anchor.
    chan.write(server_difficulty_config(0.0))

    last_log = 0.0
    updates = 0
    last_cfg = None
    try:
        while trainer.global_step < total:
            d = difficulty_at(trainer.global_step, total, a.ramp_frac)
            cfg = server_difficulty_config(d)
            if cfg != last_cfg:
                chan.write(cfg)  # publish the new d-config; the server picks it up next spawn/reset
                last_cfg = cfg
            trainer.rollouts()
            trainer.train()
            updates += 1
            if time.time() - last_log > 1.0 or trainer.global_step >= total:
                last_log = time.time()
                flat = dict(unroll_nested_dict(trainer.log()))
                print(
                    f"[curriculum] step={trainer.global_step} updates={updates} d={d:.3f} "
                    f"cfg(spawn={cfg['spawn_geo_dist']},hp={cfg['agent_hp']},def={cfg['agent_def']},boss={cfg['boss_hp']}) "
                    f"SPS={flat.get('SPS', 0):.0f} reward={flat.get('env/reward', 0):.4f} "
                    f"done_rate={flat.get('env/done_rate', 0):.4f}",
                    flush=True,
                )
        trainer.save_weights(a.out)
        print(f"== DONE: {updates} PPO updates, {trainer.global_step} steps -> {a.out} ==", flush=True)
    finally:
        chan.close()
        trainer.close()


if __name__ == "__main__":
    main()
