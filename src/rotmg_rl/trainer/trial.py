#!/usr/bin/env python3
"""One server-as-sim SWEEP TRIAL in a single process: curriculum-train the policy (d ramps 0->1 over
the run) with a given hparam set, then run the eval ladder on the SAME warm policy/vec and emit the
curriculum depth. This is the per-trial worker server_sweep.py spawns; the C# server is already up
(server_sweep.py owns the server lifecycle + the reward env vars). Splitting train+eval out into this
worker keeps each trial's GPU/shm state in its own process (a clean teardown between trials -- no
CUDA-context or shm-handle leakage across 24 trials).

The hparams arrive as JSON on --hparams (the flat dict apply_hparams/server flow consume). The PPO +
policy knobs are applied to the pufferl args here; the reward knobs (rew_approach/rew_step) are applied
by the parent as server env vars (SIM_APPROACH_SCALE / SIM_STEP_PENALTY) BEFORE the server boots, since
the C# engine reads them at launch. The non-wired reward dims (rew_boss_dmg/rew_clear/rew_speed/rew_death)
stay in the search vector for warm-start/dimensionality parity but are inert in this engine (those reward
weights are fixed in SimRlLoop.cs); see server_sweep.py's header.

Emits one line `TRIAL_RESULT_JSON {...}` with {depth, rates, train_sps, train_steps} the parent parses.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch  # noqa: F401  # torch first so its cudart loads before pufferlib's _C
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy

from rotmg_rl.schedule import CURRICULUM_RUNGS
from rotmg_rl.sweep import SWEEP_POLICY, SWEEP_TRAIN, _INT_HP
from rotmg_rl.trainer.difficulty import curriculum_depth, difficulty_at, server_difficulty_config
from rotmg_rl.trainer.eval import EP_RE, tail_episodes
from rotmg_rl.trainer.shm_config import ShmConfigChannel


def apply_ppo_hparams(args: dict, hp: dict) -> None:
    """Apply ONLY the PPO/train + policy knobs (the ones pufferl actually reads) into the args dict.
    The env reward knobs are server-side env vars, applied by the parent, not here."""
    for k in SWEEP_TRAIN:
        if k in hp and k != "ramp_frac":  # ramp_frac is the schedule's own knob, consumed below
            args["train"][k] = int(hp[k]) if k in _INT_HP else hp[k]
    for k in SWEEP_POLICY:
        if k in hp:
            args["policy"][k] = int(hp[k]) if k in _INT_HP else hp[k]


def run_ladder_on_trainer(trainer, agents, server_log, episodes_per_rung, rungs,
                          settle_episodes=4, max_rollouts_per_rung=400):
    """The eval ladder driven on an ALREADY-BUILT trainer (the warm curriculum policy), tailing the
    same server log. Mirrors server_eval.run_ladder but reuses the in-process vec instead of attaching
    a fresh one -- so the trained policy evals against the same N worlds it trained on, no reconnect."""
    chan = ShmConfigChannel(agents)
    episodes: deque = deque(maxlen=500_000)
    stop = threading.Event()
    tailer = threading.Thread(target=tail_episodes, args=(Path(server_log), episodes, stop), daemon=True)
    tailer.start()
    time.sleep(0.3)

    rates: dict[float, float] = {}
    try:
        for d in rungs:
            cfg = server_difficulty_config(d)
            chan.write(cfg)
            print(f"[ladder] d={d:.2f} -> cfg={cfg}", flush=True)
            settle_target = len(episodes) + settle_episodes * agents
            t_settle = time.time()
            while len(episodes) < settle_target and time.time() - t_settle < 60:
                trainer.rollouts()
            mark = len(episodes)
            for _ in range(max_rollouts_per_rung):
                trainer.rollouts()
                if len(episodes) - mark >= episodes_per_rung:
                    break
            window = list(episodes)[mark:]
            n_clear = sum(1 for (_, r) in window if r == "clear")
            n_ep = len(window)
            rate = n_clear / n_ep if n_ep else 0.0
            rates[d] = rate
            print(f"[ladder] d={d:.2f} clear_rate={rate:.3f} ({n_clear}/{n_ep} episodes)", flush=True)
    finally:
        stop.set()
        chan.close()
    return rates


def main() -> None:
    p = argparse.ArgumentParser(description="One server-as-sim sweep trial: curriculum train + eval ladder -> depth.")
    p.add_argument("--agents", type=int, default=48)
    p.add_argument("--steps", type=int, required=True, help="env steps per agent for the curriculum train")
    p.add_argument("--hparams", required=True, help="JSON flat hparam dict (PPO + policy + reward)")
    p.add_argument("--server-log", required=True)
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    p.add_argument("--out", default="checkpoints/sweep/trial.pt")
    p.add_argument("--episodes-per-rung", type=int, default=12)
    p.add_argument("--eval-only-final", action="store_true", help="(default) eval the ladder once at d=1 end")
    a = p.parse_args()

    hp = json.loads(a.hparams)
    ramp_frac = float(hp["ramp_frac"]) if "ramp_frac" in hp else 0.6

    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    args["env"]["redis_port"] = a.redis_port
    args["env"]["redis_db"] = a.redis_db
    args["train"]["total_timesteps"] = a.steps * a.agents
    apply_ppo_hparams(args, hp)

    print(f"== TRIAL: N={a.agents} steps/agent={a.steps} ramp_frac={ramp_frac:.3f} "
          f"lr={args['train']['learning_rate']:.5f} gamma={args['train']['gamma']:.4f} "
          f"gae={args['train']['gae_lambda']:.4f} hidden={args['policy']['hidden_size']} "
          f"layers={args['policy']['num_layers']} mb={args['train']['minibatch_size']} ==", flush=True)

    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    chan = ShmConfigChannel(a.agents)
    total = a.steps * a.agents
    chan.write(server_difficulty_config(0.0))  # seed d=0 before the first rollout

    last_log = 0.0
    updates = 0
    last_cfg = None
    last_sps = 0.0
    try:
        while trainer.global_step < total:
            d = difficulty_at(trainer.global_step, total, ramp_frac)
            cfg = server_difficulty_config(d)
            if cfg != last_cfg:
                chan.write(cfg)
                last_cfg = cfg
            trainer.rollouts()
            trainer.train()
            updates += 1
            if time.time() - last_log > 2.0 or trainer.global_step >= total:
                last_log = time.time()
                flat = dict(unroll_nested_dict(trainer.log()))
                last_sps = flat["SPS"] if "SPS" in flat else 0.0
                print(f"[train] step={trainer.global_step} updates={updates} d={d:.3f} "
                      f"cfg(spawn={cfg['spawn_geo_dist']},hp={cfg['agent_hp']},def={cfg['agent_def']},boss={cfg['boss_hp']}) "
                      f"SPS={last_sps:.0f} reward={flat['env/reward'] if 'env/reward' in flat else 0:.4f} "
                      f"done_rate={flat['env/done_rate'] if 'env/done_rate' in flat else 0:.4f}", flush=True)
        trainer.save_weights(a.out)
        chan.close()
        print(f"== TRAIN DONE: {updates} updates, {trainer.global_step} steps -> {a.out} ==", flush=True)

        policy.eval()
        rates = run_ladder_on_trainer(
            trainer, a.agents, a.server_log, a.episodes_per_rung, CURRICULUM_RUNGS,
        )
        depth = curriculum_depth(rates)
        print("\n=== EVAL LADDER (server-as-sim) ===", flush=True)
        for d in sorted(rates):
            bar = "#" * int(rates[d] * 30)
            print(f"  d={d:.2f}  clear={rates[d]:.3f}  {bar}", flush=True)
        print(f"  CURRICULUM DEPTH = {depth:.4f}", flush=True)
        result = {"depth": depth, "rates": {str(k): v for k, v in rates.items()},
                  "train_sps": last_sps, "train_steps": trainer.global_step}
        print(f"TRIAL_RESULT_JSON {json.dumps(result)}", flush=True)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
