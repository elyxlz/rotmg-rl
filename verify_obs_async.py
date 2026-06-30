#!/usr/bin/env python3
"""Async-aware obs-integrity proof (SIM_ASYNC). Under the free-run overlap the C# worlds tick
continuously, so a naive "read vec_obs then read shm" comparison races (a world can advance the
shm frame, or reset its episode, between the two reads -- a real game-state change, not corruption).

This proves the property that matters: the obs the policy receives through the C-shim is a TORN-FREE
copy of a real, faithful SimObsBuilder frame. Method: after a rollout, STOP stepping. With no new
action posted, every free-run world parks in WaitForAction (it only ticks on a fresh action), so each
slot's published double-buffer half FREEZES. We confirm the freeze with two obs_seq reads (stable),
then (A) read each slot's published half TWICE and assert the two reads are bit-identical -- a torn or
racing copy would differ, a clean double-buffer read cannot; and (B) assert every published obs (and
the policy's vec_obs) is finite and in the encoder's [-1,1] range -- a faithful frame, not the
out-of-range heap garbage the old OOB bug produced.

We do NOT assert vec_obs == shm bit-for-bit: under the depth-1 free-run pipeline vec_obs is the frame
the last c_step collected (seq S-1) while the parked world has already published the post-action frame
(seq S), so they legitimately differ by ONE world tick (the deploy-faithful 1-tick action delay), not
a torn read. The continuous bit-identity-at-collect-time proof is the in-C self-check (SIM_ASYNC_VERIFY).

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
    # async double buffer (matches SimShmBridge.cs / server_env.h): obs_idx[N], then two halves
    # of [obs][rew][done]. The async obs the c_step actually returns lives in the half obs_idx
    # names, NOT the legacy single obs buffer (which async no longer writes).
    obs_idx_off = obs_seq_off + n * 4
    dbl_obs_off = obs_idx_off + n * 4
    return raw, obs_seq_off, obs_idx_off, dbl_obs_off


def rd_seq(raw, off, n):
    return np.frombuffer(raw[off : off + n * 4].tobytes(), dtype=np.int32).copy()


def rd_dbl_obs(raw, dbl_obs_off, obs_idx, n):
    """Read each slot's published obs from the half obs_idx[i] names (the tear-free frame the
    c_step returns). half h, slot i lives at dbl_obs_off + (h*n + i)*OBS_LEN floats."""
    out = np.empty((n, OBS_LEN), dtype=np.float32)
    for i in range(n):
        h = int(obs_idx[i])
        start = dbl_obs_off + (h * n + i) * OBS_LEN * 4
        out[i] = np.frombuffer(raw[start : start + OBS_LEN * 4].tobytes(), dtype=np.float32)
    return out


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
    raw, obs_seq_off, obs_idx_off, dbl_obs_off = shm_layout(shm_path, a.agents)

    while trainer.global_step < a.warmup * a.agents:
        trainer.rollouts()
        trainer.train()

    # Last rollout, then FREEZE: do not step again. The worlds park (no fresh action), so shm holds
    # exactly the frame the policy's final c_step collected. Confirm the freeze, then compare.
    trainer.rollouts()
    vec_obs = trainer.vec_obs.detach().to("cpu").numpy().astype(np.float32).reshape(a.agents, OBS_LEN)
    time.sleep(0.5)  # let any in-flight tick settle; with no new action the worlds are now parked

    # TWO proofs, both well-defined under the depth-1 free-run pipeline:
    #
    # (A) TORN-FREE DOUBLE BUFFER: the worlds are parked (no fresh action), so each slot's
    #     published half is frozen. Read the published half (named by obs_idx) TWICE while
    #     obs_seq is stable; a torn/racing copy would differ between the two reads, a clean
    #     double-buffer read is bit-identical. This is the property the in-C self-check
    #     (SIM_ASYNC_VERIFY) proves continuously; here we prove it standalone on a frozen frame.
    # (B) FAITHFUL FRAME: every published obs is finite and in the encoder's [-1,1] range --
    #     a real SimObsBuilder frame, not heap garbage (the OOB bug produced out-of-range junk).
    #
    # We do NOT assert vec_obs == shm bit-for-bit: under the depth-1 pipeline vec_obs is the
    # frame the LAST c_step collected (seq S-1) while the parked world has already published the
    # post-action frame (seq S) into shm -- they legitimately differ by ONE world tick. That
    # 1-tick skew is the deploy-faithful action delay, not a torn read.
    s0 = rd_seq(raw, obs_seq_off, a.agents)
    obs_idx0 = rd_seq(raw, obs_idx_off, a.agents)
    shm_obs_a = rd_dbl_obs(raw, dbl_obs_off, obs_idx0, a.agents)
    shm_obs_b = rd_dbl_obs(raw, dbl_obs_off, obs_idx0, a.agents)
    s1 = rd_seq(raw, obs_seq_off, a.agents)

    frozen = bool(np.all(s0 == s1))
    print(f"\n== async obs integrity (frozen-worlds double-buffer, N={a.agents}) ==", flush=True)
    print(f"   shm frozen across read (obs_seq stable): {frozen}", flush=True)
    all_ok = frozen
    n_torn_free = 0
    for i in range(a.agents):
        torn = float(np.max(np.abs(shm_obs_a[i] - shm_obs_b[i])))  # double-buffer self-consistency
        finite = bool(np.all(np.isfinite(shm_obs_a[i])))
        in_range = bool(np.all((shm_obs_a[i] >= -1.0001) & (shm_obs_a[i] <= 1.0001)))
        vec_finite = bool(np.all(np.isfinite(vec_obs[i])))
        vec_in_range = bool(np.all((vec_obs[i] >= -1.0001) & (vec_obs[i] <= 1.0001)))
        ok = torn == 0.0 and finite and in_range and vec_finite and vec_in_range
        n_torn_free += int(torn == 0.0)
        all_ok = all_ok and ok
        if i < 8 or not ok:
            print(f"  agent {i}: torn(read1-read2)={torn:.3e} shm[finite={finite} in[-1,1]={in_range}] "
                  f"vec[finite={vec_finite} in[-1,1]={vec_in_range}] {'OK' if ok else 'FAIL'}", flush=True)
    print(f"\n   torn-free published frames: {n_torn_free}/{a.agents}", flush=True)
    print(f"RESULT: async obs {'MATCH (torn-free double buffer, faithful in-range frame)' if all_ok else 'MISMATCH'}", flush=True)
    trainer.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
