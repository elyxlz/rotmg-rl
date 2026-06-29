#!/usr/bin/env python3
"""Verify the d-flow takes effect server-side: write a sequence of d-configs into shm while driving
the gate, and assert the server's [SIM-RL] spawn lines report the matching applied config (spawn
distance / agent HP / boss HP) for each d -- proving a d change reaches the C# spawn per-episode with
NO restart. Run AFTER booting run-server-sim.sh 8 with SIM_SHM_BARRIER=1."""
from __future__ import annotations

import argparse
import sys
import time

import torch  # noqa: F401
from pufferlib import _C, pufferl
from pufferlib.torch_pufferl import PuffeRL, load_policy

from rotmg_rl.server_difficulty import server_difficulty_config
from server_shm_config import ShmConfigChannel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--ds", default="0.0,0.5,1.0")
    p.add_argument("--rollouts-per-d", type=int, default=40)
    a = p.parse_args()

    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    policy.eval()
    trainer = PuffeRL(args, vec, policy, verbose=False)

    chan = ShmConfigChannel(a.agents)
    for d in [float(x) for x in a.ds.split(",")]:
        cfg = server_difficulty_config(d)
        chan.write(cfg)
        readback = chan.read()
        print(f"\n[verify] WROTE d={d:.2f} cfg={cfg}", flush=True)
        print(f"[verify] shm READBACK  ={readback}", flush=True)
        assert readback["valid"], "config block not marked valid"
        assert readback["spawn_geo_dist"] == cfg["spawn_geo_dist"]
        assert readback["agent_hp"] == cfg["agent_hp"]
        assert readback["boss_hp"] == cfg["boss_hp"]
        for _ in range(a.rollouts_per_d):
            trainer.rollouts()
        time.sleep(0.5)
    chan.close()
    trainer.close()
    print("\n[verify] done. Check the server log for matching 'applied cfg' on the spawn/reset lines.", flush=True)


if __name__ == "__main__":
    main()
