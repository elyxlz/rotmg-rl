#!/usr/bin/env python3
"""Prove the obs the policy receives through the C-shim matches what SimObsBuilder wrote
into shm — no corruption across the boundary. Runs a few gated rollout steps, then reads
the shm obs region DIRECTLY (numpy over the same /dev/shm file) and compares it byte-for-
byte to the vec-buffer obs tensor pufferl hands the policy.

Run on GPU 1 against a live SIM_SHM server (same N).
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python verify_obs.py --agents 4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from pufferlib import _C, pufferl
from pufferlib.torch_pufferl import PuffeRL, load_policy

OBS_LEN = 9807
N_ATNS = 4  # move, aim, shoot, cast (staff+spell share the aim)
HEADER_INTS = 4


def shm_view(path: str, n: int):
    """numpy views into the shm region matching SimShmBridge.cs's layout."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    hdr = np.frombuffer(raw[: HEADER_INTS * 4].tobytes(), dtype=np.int32)
    base = HEADER_INTS * 4
    obs = np.frombuffer(raw[base : base + n * OBS_LEN * 4].tobytes(), dtype=np.float32).reshape(n, OBS_LEN)
    return hdr, obs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--warmup", type=int, default=200, help="gated steps before the obs snapshot")
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

    hdr, _ = shm_view(shm_path, a.agents)
    print(f"shm header: magic=0x{hdr[0] & 0xffffffff:x} n={hdr[1]} obs_len={hdr[2]} n_atns={hdr[3]}", flush=True)
    assert hdr[1] == a.agents and hdr[2] == OBS_LEN and hdr[3] == N_ATNS, "shm header mismatch"

    # drive warmup gated steps (lets agents spawn + the obs fill with real game state)
    horizon = args["train"]["horizon"]
    while trainer.global_step < a.warmup * a.agents:
        trainer.rollouts()
        trainer.train()

    # one more rollout, then snapshot: vec_obs is the LAST obs the policy saw this rollout,
    # which the C-shim copied from shm on the final gated tick. Re-read shm directly and
    # compare — the gate is idle now (trainer between rollouts), so shm holds that same frame.
    vec_obs = trainer.vec_obs.detach().to("cpu").numpy().astype(np.float32).reshape(a.agents, OBS_LEN)
    _, shm_obs = shm_view(shm_path, a.agents)

    print("\n== obs integrity: vec-buffer (policy input) vs shm (SimObsBuilder output) ==", flush=True)
    all_ok = True
    for i in range(a.agents):
        v = vec_obs[i]
        s = shm_obs[i]
        max_abs = float(np.max(np.abs(v - s))) if v.size else float("nan")
        finite = bool(np.all(np.isfinite(v)))
        in_range = bool(np.all((v >= -1.0001) & (v <= 1.0001)))
        nz = int(np.count_nonzero(v))
        # structural: the obs splits into grid / minimap / scalars (dungeon layout)
        grid_nz = int(np.count_nonzero(v[: 7 * 31 * 31]))
        mm_nz = int(np.count_nonzero(v[7 * 31 * 31 : 7 * 31 * 31 + 3 * 32 * 32]))
        ok = max_abs == 0.0 and finite and in_range
        all_ok = all_ok and ok
        print(
            f"  agent {i}: max|vec-shm|={max_abs:.3e}  finite={finite}  in[-1,1]={in_range}  "
            f"nonzero={nz} (grid={grid_nz} mm={mm_nz})  {'OK' if ok else 'FAIL'}",
            flush=True,
        )

    print(f"\nRESULT: obs {'MATCH (bit-identical across the shm boundary)' if all_ok else 'MISMATCH'}", flush=True)
    trainer.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
