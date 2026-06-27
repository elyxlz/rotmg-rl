#!/usr/bin/env python3
"""Protein (PufferLib cost-aware Bayesian) hyperparameter sweep over the faithful continuous-schedule
Snake Pit task, maximizing the TRUE full-difficulty (d=1) per-episode clear rate.

    python3 scripts/sweep_dungeon4.py --max-runs 16 --trial-steps 70_000_000 --launch-best

Self-bootstraps under .venv4 (the 4.0 stack), like train.py. Each trial: Protein suggests a config ->
we run train.py's continuous schedule for `--trial-steps` (cold start) -> we EVAL the d=1 clear rate
on the numpy env every `eval_every` steps -> Protein observes that cost-aware (uptime) trajectory.
Trials use a reduced `--sweep-boss-hp` so a good config can actually clear within the modest budget
(the metric needs gradient); the swept HYPERPARAMETERS then drive a full 7500-HP run via --launch-best.

Search space (8 knobs): learning_rate, gamma, gae_lambda, ent_coef (train), hidden_size (policy,
cost-aware), the difficulty ramp-rate ramp_frac (train, schedule-only), rew_approach + rew_boss_dmg
(env). We do NOT sweep clip coefficients (the Protein guide warns against it)."""

from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv4" / "bin" / "python"

if pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
    import subprocess

    if not VENV_PY.exists():
        print("== .venv4 missing -> provisioning the 4.0 stack (one-time) ==", flush=True)
        subprocess.run(["bash", str(REPO / "scripts" / "setup_box_puffer4.sh")], check=True)
    os.execv(str(VENV_PY), [str(VENV_PY), str(pathlib.Path(__file__).resolve()), *sys.argv[1:]])

import argparse
import time
from copy import deepcopy

import numpy as np
import torch

import pufferlib.sweep
from pufferlib import pufferl

sys.path.insert(0, str(REPO))
import train as T  # train.py's continuous schedule + d=1 eval (imported under .venv4, no re-exec)


def _space(distribution, lo, hi, mean, scale="auto"):
    return {"distribution": distribution, "min": lo, "max": hi, "mean": mean, "scale": scale}


def build_sweep_config(metric: str) -> dict:
    """Protein search space. gamma keeps the long-horizon attention (the lever that broke the 73%
    plateau); ramp_frac is the schedule's own knob (how fast d climbs vs how long it holds at 1)."""
    return {
        "method": "Protein",
        "metric": metric,
        "metric_distribution": "linear",  # clear rate is a linear 0..1 objective
        "goal": "maximize",
        "downsample": 5,
        "max_suggestion_cost": 3600,
        "early_stop_quantile": 0.3,
        "train": {
            "learning_rate": _space("log_normal", 3e-4, 3e-2, 1.5e-2),
            "gamma": _space("logit_normal", 0.95, 0.999, 0.97),
            "gae_lambda": _space("logit_normal", 0.80, 0.99, 0.88),
            "ent_coef": _space("log_normal", 1e-3, 5e-2, 2e-2),
            "ramp_frac": _space("uniform", 0.3, 0.85, 0.6),  # schedule-only knob (PuffeRL ignores it)
        },
        "policy": {
            "hidden_size": _space("uniform_pow2", 128, 512, 256),  # cost-aware
        },
        "env": {
            "rew_approach": _space("uniform", 0.0, 0.06, 0.02),
            "rew_boss_dmg": _space("uniform", 0.5, 2.0, 1.0),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-runs", type=int, default=16)
    p.add_argument("--trial-steps", type=int, default=70_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--sweep-boss-hp", type=float, default=4000.0, help="reduced boss HP for trials so the metric has gradient in-budget")
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--n-snakes-max", type=int, default=T.N_SNAKES_MAX)
    p.add_argument("--out-dir", default="checkpoints/sweep4")
    p.add_argument("--metric", default="clear_d1")
    p.add_argument("--launch-best", action="store_true", help="after the sweep, run the FULL 7500-HP schedule with the winner")
    p.add_argument("--full-steps", type=int, default=460_000_000)
    a = p.parse_args()
    sys.argv = [sys.argv[0]]  # keep pufferl.load_config's argparse out of our args

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = a.num_envs
    args["vec"]["num_buffers"] = 1
    # ramp_frac is a schedule-only knob (not an ini key); seed it so trial 1 (defaults, no suggest)
    # still carries every swept hyperparameter when we observe() it.
    args["train"]["ramp_frac"] = 0.6

    sweep_config = build_sweep_config(a.metric)
    method = sweep_config.pop("method")
    sweep_obj = getattr(pufferlib.sweep, method)(sweep_config)
    eval_every = max(1, a.trial_steps // 5)

    best = {"clear": -1.0, "hp": None}
    for i in range(a.max_runs):
        if i > 0:  # the first trial uses the config defaults; Protein suggests thereafter
            sweep_obj.suggest(args)
        seed = time.time_ns() & 0xFFFFFFFF
        np.random.seed(seed)
        torch.manual_seed(seed)

        tr = args["train"]
        ramp_frac = float(tr.get("ramp_frac", 0.6))
        hp = dict(learning_rate=round(tr["learning_rate"], 6), gamma=round(tr["gamma"], 5), gae_lambda=round(tr["gae_lambda"], 4),
                  ent_coef=round(tr["ent_coef"], 5), ramp_frac=round(ramp_frac, 3), hidden_size=int(args["policy"]["hidden_size"]),
                  rew_approach=round(args["env"]["rew_approach"], 4), rew_boss_dmg=round(args["env"]["rew_boss_dmg"], 3))
        print(f"\n=== TRIAL {i + 1}/{a.max_runs} === {hp}", flush=True)

        trial = deepcopy(args)
        trial["train"]["total_timesteps"] = a.trial_steps
        trial["env"]["boss_hp_max"] = a.sweep_boss_hp
        t0 = time.time()
        try:
            trainer, policy, evals = T.train_continuous(
                trial, a.trial_steps, ramp_frac, float(trial["env"]["rew_approach"]), a.n_snakes_max, out, use_wandb=False,
                eval_every=eval_every, eval_episodes=a.eval_episodes, boss_hp=a.sweep_boss_hp, verbose=True)
            trainer.close()
        except Exception as exc:
            print(f"TRIAL {i + 1} FAILED: {exc}", flush=True)
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue

        if not evals:
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue
        final = evals[-1]["clear_d1"]
        peak = max(e["clear_d1"] for e in evals)
        print(f"TRIAL {i + 1}: d=1 clear final={final:.3f} peak={peak:.3f}", flush=True)
        for e in evals:  # cost-aware trajectory: (clear_d1 at uptime e['uptime'], cost in timesteps)
            args["train"]["total_timesteps"] = e["step"]
            sweep_obj.observe(args, float(e["clear_d1"]), float(e["uptime"]))
        args["train"]["total_timesteps"] = a.trial_steps
        if peak > best["clear"]:
            best = {"clear": peak, "hp": hp}
            print(f"  >> new best d=1 clear {peak:.3f}", flush=True)

    print(f"\n==== SWEEP DONE. BEST d=1 clear (boss_hp={a.sweep_boss_hp:g}): {best['clear']:.3f} ====", flush=True)
    print(f"==== BEST CONFIG: {best['hp']} ====", flush=True)

    if a.launch_best and best["hp"]:
        b = best["hp"]
        print(f"\n== launching the FULL 7500-HP run with the winner ({a.full_steps / 1e6:.0f}M steps) ==", flush=True)
        import time as _time

        use_wandb = True
        try:
            import wandb

            wandb.init(project="rotmg-dungeon", name=f"continuous4-best-{int(_time.time())}", group="continuous4", config=b)
        except Exception as exc:  # a wandb hiccup must not waste the swept config -> train anyway
            print(f"[wandb] disabled ({exc}); training without logging", flush=True)
            use_wandb = False
        full = T.build_args(a.num_envs, b["hidden_size"], b["learning_rate"], b["gamma"], b["gae_lambda"], b["ent_coef"], a.full_steps, T.BOSS_HP, b["rew_boss_dmg"])
        finish = REPO / "checkpoints" / "curriculum4"
        finish.mkdir(parents=True, exist_ok=True)
        trainer, policy, _ = T.train_continuous(full, a.full_steps, b["ramp_frac"], b["rew_approach"], a.n_snakes_max, finish, use_wandb=use_wandb, boss_hp=T.BOSS_HP)
        trainer.save_weights(str(finish / "finish.pt"))
        trainer.close()
        rate = T.eval_clear_rate(policy, 100, d=1.0, boss_hp=T.BOSS_HP, n_snakes_max=a.n_snakes_max)
        print(f"== FULL RUN DONE -> {finish / 'finish.pt'} | d=1 clear rate {rate:.1%} ==", flush=True)
        if use_wandb:
            wandb.log({"final/clear_rate_d1": rate})
            wandb.finish()


if __name__ == "__main__":
    main()
