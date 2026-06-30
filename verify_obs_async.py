#!/usr/bin/env python3
"""Async-aware obs-integrity proof (SIM_ASYNC). Under the free-run overlap the C# worlds tick
continuously, so a naive "read vec_obs then read shm" comparison races (a world can advance the
shm frame, or reset its episode, between the two reads -- a real game-state change, not corruption).

This proves the property that matters: the obs the policy receives through the C-shim is a TORN-FREE
bit-identical copy of a real SimObsBuilder frame. Method: after a rollout, STOP stepping. With no
new action posted, every free-run world parks in WaitForAction (it only ticks on a fresh action),
so the shm frame FREEZES at exactly the frame the policy's last c_step collected. We confirm the
freeze with two obs_seq reads (stable), then assert vec_obs == shm bit-for-bit for every slot. A
torn/corrupt copy would differ here; a one-frame-behind artifact cannot occur because nothing ticks.

    SIM_ASYNC=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python verify_obs_async.py --agents 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from pufferlib import _C, pufferl
from pufferlib.torch_pufferl import PuffeRL, load_policy

OBS_LEN = 9807
N_ATNS = 4  # move, aim, shoot, cast (staff+spell share the aim)
HEADER_INTS = 4
CTRL_INTS = 2
CONFIG_INTS = 5


def shm_layout(path: str, n: int):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    obs_off = HEADER_INTS * 4
    act_seq_off = obs_off + (n * OBS_LEN + n * N_ATNS + n + n) * 4 + (CTRL_INTS + CONFIG_INTS) * 4
    obs_seq_off = act_seq_off + n * 4
    return raw, obs_off, obs_seq_off


def rd_seq(raw, off, n):
    return np.frombuffer(raw[off : off + n * 4].tobytes(), dtype=np.int32).copy()


def rd_obs(raw, off, n):
    return np.frombuffer(raw[off : off + n * OBS_LEN * 4].tobytes(), dtype=np.float32).reshape(n, OBS_LEN).copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=64)
    p.add_argument("--warmup", type=int, default=150)
    a = p.parse_args()
    shm_path = os.environ.get("SIM_SHM_PATH", "/dev/shm/rotmg_sim_shm")

    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    args["train"]["total_timesteps"] = 10_000_000

    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)
    raw, obs_off, obs_seq_off = shm_layout(shm_path, a.agents)

    while trainer.global_step < a.warmup * a.agents:
        trainer.rollouts()
        trainer.train()

    # Last rollout, then FREEZE: do not step again. The worlds park (no fresh action), so shm holds
    # exactly the frame the policy's final c_step collected. Confirm the freeze, then compare.
    trainer.rollouts()
    vec_obs = trainer.vec_obs.detach().to("cpu").numpy().astype(np.float32).reshape(a.agents, OBS_LEN)
    time.sleep(0.5)  # let any in-flight tick settle; with no new action the worlds are now parked
    s0 = rd_seq(raw, obs_seq_off, a.agents)
    shm_obs = rd_obs(raw, obs_off, a.agents)
    s1 = rd_seq(raw, obs_seq_off, a.agents)

    frozen = bool(np.all(s0 == s1))
    print(f"\n== async obs integrity (frozen-worlds, N={a.agents}) ==", flush=True)
    print(f"   shm frozen across read (obs_seq stable): {frozen}", flush=True)
    all_ok = frozen
    n_id = 0
    for i in range(a.agents):
        d = float(np.max(np.abs(vec_obs[i] - shm_obs[i])))
        finite = bool(np.all(np.isfinite(vec_obs[i])))
        in_range = bool(np.all((vec_obs[i] >= -1.0001) & (vec_obs[i] <= 1.0001)))
        ok = d == 0.0 and finite and in_range
        n_id += int(d == 0.0)
        all_ok = all_ok and ok
        if i < 8 or not ok:
            print(f"  agent {i}: max|vec-shm|={d:.3e} finite={finite} in[-1,1]={in_range} {'OK' if ok else 'FAIL'}", flush=True)
    print(f"\n   bit-identical slots: {n_id}/{a.agents}", flush=True)
    print(f"RESULT: async obs {'MATCH (torn-free, bit-identical across the shm boundary)' if all_ok else 'MISMATCH'}", flush=True)
    trainer.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
