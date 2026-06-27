"""Protein hyperparameter sweep over the custom dungeon env + CNN-LSTM policy.

PufferLib's stock `pufferl.sweep` assumes an env registered through `train()`'s registry. Our env
(CDungeon) + policy (CDungeonPolicy) are built by hand exactly like train_dungeon.py, so we run the
Protein loop ourselves: suggest -> build vecenv+policy (warm-started) -> train() -> observe, with
cost = wall-clock uptime (cost-aware, per the PROTEIN method).

Metric: by default the sweep maximizes `environment/cleared` -- the per-step clear RATE, which is
the cautious-finish problem's own symptom (it decays as the agent lingers instead of finishing the
kill). The per-episode `environment/score` (1.0 if cleared, else the boss damage fraction) we added
SATURATES on this config: max_steps is 4000, so even a badly cautious agent finishes the kill
eventually -> every episode clears -> score ~1.0 while cleared has decayed to ~0.003. So `cleared`
is the metric with real gradient here; `score` stays logged for reference. Both are independent of
the tunable reward scale (the guide: "log a score, not raw reward"). Knobs: learning_rate, ent_coef,
gamma, gae_lambda, vf_coef and the reward balance (rew_clear, rew_death, rew_boss_dmg). Clip
coefficients are NOT swept (the guide warns against it).

    uv run python scripts/sweep_dungeon.py --max-runs 24 --trial-steps 15000000 --metric cleared \
        --num-envs 1024 --init-checkpoint checkpoints/passive.pt --wandb-group sweep-shooting
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import pufferlib.sweep
import pufferlib.vector as pvector
import torch
from pufferlib import pufferl
from pufferlib.ocean import torch as ocean_torch
from pufferlib.pufferl import downsample

from rotmg_rl.csim.dungeon import CDungeon
from rotmg_rl.csim.policy import CDungeonPolicy
from rotmg_rl.sim.dungeon import DungeonConfig


def _space(distribution: str, lo: float, hi: float, mean: float, scale="auto") -> dict:
    return {"distribution": distribution, "min": lo, "max": hi, "scale": scale, "mean": mean}


def build_sweep_config(metric: str) -> dict:
    """Protein search space. gamma gets the widest principled attention (the hypothesized cause of
    the cautious finish: the future +clear is over-discounted vs the immediate death risk)."""
    return {
        "method": "Protein",
        "metric": metric,
        "goal": "maximize",
        "downsample": 10,
        "train": {
            "learning_rate": _space("log_normal", 3e-4, 3e-2, 1.5e-2),
            "ent_coef": _space("log_normal", 1e-4, 5e-2, 1e-2),
            "gamma": _space("logit_normal", 0.95, 0.999, 0.997),  # push the discount up
            "gae_lambda": _space("logit_normal", 0.80, 0.99, 0.92),
            "vf_coef": _space("log_normal", 0.5, 5.0, 2.0),
        },
        "env": {
            "rew_clear": _space("uniform", 0.5, 3.0, 1.0),  # reward of finishing the kill
            "rew_death": _space("uniform", 0.1, 1.0, 0.5),  # penalty for dying (cautiousness knob)
            "rew_boss_dmg": _space("uniform", 0.3, 1.5, 1.0),
        },
    }


def make_trial_vecenv(args, num_envs: int):
    env_cfg = DungeonConfig(
        spawn_in_room_prob=1.0,
        random_spawn_prob=0.0,
        boss_hp_max=args.boss_hp,
        n_snakes=0,
        enable_grenades=False,
        enable_minions=False,
        boss_shoots=True,
        # swept reward balance (suggest filled cfg['env'])
        rew_clear=args.cfg_env["rew_clear"],
        rew_death=args.cfg_env["rew_death"],
        rew_boss_dmg=args.cfg_env["rew_boss_dmg"],
    )
    vecenv = pvector.make(
        CDungeon,
        env_kwargs={"config": env_cfg, "num_envs": num_envs},
        backend=pvector.PufferEnv,
        num_envs=1,
    )
    return vecenv, env_cfg


def make_trial_policy(vecenv, hidden: int, init_checkpoint: str | None):
    policy = CDungeonPolicy(vecenv.driver_env, hidden_size=hidden)
    policy = ocean_torch.Recurrent(vecenv.driver_env, policy, input_size=hidden, hidden_size=hidden)
    policy = policy.to("cuda")
    if init_checkpoint:
        policy.load_state_dict(torch.load(init_checkpoint, map_location="cuda"))
    return policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=30)
    p.add_argument("--trial-steps", type=int, default=10_000_000, help="timesteps per trial (short -> minutes)")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--boss-hp", type=float, default=300.0)
    p.add_argument("--metric", default="cleared", help="environment/<metric> to maximize (score saturates here; cleared has gradient)")
    p.add_argument("--init-checkpoint", default="checkpoints/passive.pt", help="warm-start each trial from this policy")
    p.add_argument("--wandb-group", default="sweep-shooting")
    args = p.parse_args()
    sys.argv = [sys.argv[0]]  # keep pufferl.load_config's argparse out of our args

    cfg = pufferl.load_config("default")
    t = cfg["train"]
    t["device"] = "cuda"
    t["use_rnn"] = True
    t["compile"] = False
    t["checkpoint_interval"] = 100000  # don't spam checkpoints during a sweep
    # DON'T touch batch_size/minibatch_size/bptt_horizon: tuned together with the optimizer defaults.
    cfg["wandb"] = True
    cfg["wandb_project"] = "rotmg-dungeon"
    cfg["wandb_group"] = args.wandb_group
    cfg["neptune"] = False
    cfg["env_name"] = "rotmg_dungeon"
    cfg["env"] = {}  # suggest fills the swept reward params here

    cfg["sweep"] = build_sweep_config(args.metric)
    metric = cfg["sweep"]["metric"]
    points_per_run = cfg["sweep"]["downsample"]
    target_key = f"environment/{metric}"
    method = cfg["sweep"].pop("method")
    sweep = getattr(pufferlib.sweep, method)(cfg["sweep"])

    for i in range(args.max_runs):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sweep.suggest(cfg)  # mutates cfg['train'] and cfg['env'] in place
        cfg["train"]["total_timesteps"] = args.trial_steps
        args.cfg_env = cfg["env"]
        hp = {k: round(cfg["train"][k], 6) for k in ("learning_rate", "ent_coef", "gamma", "gae_lambda", "vf_coef")}
        print(f"\n=== TRIAL {i + 1}/{args.max_runs} === train={hp} env={cfg['env']}", flush=True)

        vecenv, _ = make_trial_vecenv(args, args.num_envs)
        policy = make_trial_policy(vecenv, args.hidden, args.init_checkpoint)
        t0 = time.time()
        # train() closes the vecenv itself via PuffeRL.close(); a second close double-frees the C env.
        all_logs = pufferl.train(cfg["env_name"], args=cfg, vecenv=vecenv, policy=policy)
        elapsed = max(time.time() - t0, 1.0)  # cost (seconds); Protein takes log(cost), so keep it > 0

        all_logs = [e for e in all_logs if target_key in e]
        if not all_logs:
            print(f"TRIAL {i + 1}: no logs with {target_key}; recording failure", flush=True)
            sweep.observe(cfg, 0.0, elapsed, is_failure=True)
            continue
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = downsample([log["agent_steps"] for log in all_logs], points_per_run)
        best = max(log[target_key] for log in all_logs)
        print(f"TRIAL {i + 1}: best {target_key}={best:.4f} (final={all_logs[-1][target_key]:.4f})", flush=True)
        for score, cost, timestep in zip(scores, costs, timesteps):
            cfg["train"]["total_timesteps"] = int(timestep)
            sweep.observe(cfg, float(score), float(cost))
        cfg["train"]["total_timesteps"] = args.trial_steps


if __name__ == "__main__":
    main()
