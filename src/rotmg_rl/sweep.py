"""Protein hyperparameter sweep (cost-aware Bayesian over CURRICULUM DEPTH -- how far up the difficulty
ladder the policy clears) + the hparam plumbing the training entry shares.

`run_sweep` drives the continuous-difficulty training loop (rotmg_rl.training) over a Protein search
space and returns the best hyperparameter dict. `python3 train.py` calls it then trains the winner;
this module's `main` is the sweep-alone entry (find + print the best config, no full run):

    python -m rotmg_rl.sweep --sweep-trials 16
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from copy import deepcopy

import pufferlib.sweep  # ty: ignore[unresolved-import]  pufferlib is pip-installed only on the GPU box
from pufferlib import pufferl  # ty: ignore[unresolved-import]  pufferlib is pip-installed only on the GPU box

from rotmg_rl.schedule import LADDER_EPISODES, N_SNAKES_MAX

# Swept hyperparameters, by the pufferl-config section they live in. ramp_frac is schedule-only
# (PuffeRL ignores it; train_continuous reads it). Every key here is verified to be read by 4.0
# PuffeRL / load_policy / the env Config -- no dead dimensions (update_epochs/vtrace are NOT read; we
# don't sweep horizon since minibatch_size % horizon must hold, nor the clip coeffs per the guide).
SWEEP_TRAIN = ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "minibatch_size", "ramp_frac")
SWEEP_POLICY = ("hidden_size", "num_layers")
SWEEP_ENV = ("rew_approach", "rew_boss_dmg", "rew_clear", "rew_speed", "rew_death", "rew_step")
_INT_HP = ("hidden_size", "num_layers", "minibatch_size")


def apply_hparams(args: dict, hp: dict) -> None:
    """Write whatever swept knobs are present in hp into the right pufferl-config section in place."""
    for section, keys in (("train", SWEEP_TRAIN), ("policy", SWEEP_POLICY), ("env", SWEEP_ENV)):
        for k in keys:
            if k in hp:
                args[section][k] = int(hp[k]) if k in _INT_HP else hp[k]


def _hparams_from_args(args) -> dict:
    """Pull the swept knobs back out of a (suggest-filled) pufferl args dict into the flat hp dict that
    apply_hparams + train_continuous consume."""
    hp = {k: args["train"][k] for k in SWEEP_TRAIN}
    hp.update({k: args["policy"][k] for k in SWEEP_POLICY})
    hp.update({k: args["env"][k] for k in SWEEP_ENV})
    for k in _INT_HP:
        hp[k] = int(hp[k])
    return hp


def _space(distribution, lo, hi, mean):
    return {"distribution": distribution, "min": lo, "max": hi, "mean": mean, "scale": "auto"}


def build_sweep_config(metric: str = "curriculum_depth") -> dict:
    """Protein search space (16 knobs) -- give Protein the heavy lifting, few hand-set assumptions.
    Every knob is read by 4.0 PuffeRL / load_policy / the env Config (verified). gamma keeps the
    long-horizon attention (the lever that broke the 73% plateau); ramp_frac is the schedule's own knob.
    NOTE: 16 dims wants more than the default trial budget -- raise --sweep-trials (~40+) for a thorough
    search; 16-24 is a coarse pass. We deliberately do NOT sweep the clip coefficients (guide warning)
    nor horizon (minibatch_size % horizon must hold)."""
    return {
        "method": "Protein",
        "metric": metric,
        "metric_distribution": "linear",  # curriculum depth is a linear 0..1 objective
        "goal": "maximize",
        "downsample": 5,
        "max_suggestion_cost": 3600,
        "early_stop_quantile": 0.3,
        "train": {
            "learning_rate": _space("log_normal", 2e-4, 5e-2, 1.5e-2),
            "gamma": _space("logit_normal", 0.95, 0.999, 0.97),
            "gae_lambda": _space("logit_normal", 0.80, 0.99, 0.88),
            "ent_coef": _space("log_normal", 5e-4, 8e-2, 2e-2),
            "vf_coef": _space("log_normal", 0.3, 3.0, 1.0),
            "max_grad_norm": _space("log_normal", 0.3, 5.0, 1.0),
            "minibatch_size": _space("uniform_pow2", 512, 2048, 1024),  # stays divisible by horizon 64
            "ramp_frac": _space("uniform", 0.3, 0.85, 0.6),  # schedule-only knob (PuffeRL ignores it)
        },
        "policy": {
            "hidden_size": _space("uniform_pow2", 128, 1024, 256),  # cost-aware
            "num_layers": _space("int_uniform", 1, 3, 1),
        },
        "env": {
            "rew_approach": _space("uniform", 0.0, 0.06, 0.02),
            "rew_boss_dmg": _space("uniform", 0.5, 2.0, 1.0),
            "rew_clear": _space("uniform", 0.5, 3.0, 1.0),
            "rew_speed": _space("uniform", 0.0, 0.6, 0.2),  # small fast-clear bonus, tuned alongside the clear reward
            "rew_death": _space("uniform", 0.1, 1.0, 0.5),
            "rew_step": _space("uniform", -0.003, 0.0, -0.001),
        },
    }


def run_sweep(
    num_envs, sweep_trials, trial_steps, sweep_boss_hp, eval_episodes, n_snakes_max, out, ladder_episodes=LADDER_EPISODES
) -> dict | None:
    """Protein sweep maximizing CURRICULUM DEPTH -- how far up the difficulty ladder the policy clears
    (eval'd every eval_every steps -> the observed cost-aware uptime trajectory). Depth gives gradient on
    the authored map where the d=1 clear rate is 0 for every config at a sweep budget. The objective is
    purely clear-based (per-rung clear rate only); the reward cocktail stays a search variable Protein
    tunes to maximize it. Returns the best hyperparameter dict (None if every trial failed)."""
    from rotmg_rl.training import train_continuous  # deferred: training imports this module at top

    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = num_envs
    args["vec"]["num_buffers"] = 1
    args["train"]["ramp_frac"] = 0.6  # schedule-only knob (not an ini key); seed it so trial 1 (defaults) carries it through observe
    sweep_config = build_sweep_config()
    method = sweep_config.pop("method")
    sweep_obj = getattr(pufferlib.sweep, method)(sweep_config)
    eval_every = max(1, trial_steps // 5)

    best = {"depth": -1.0, "hp": None}
    for i in range(sweep_trials):
        if i > 0:  # the first trial uses the config defaults; Protein suggests thereafter
            sweep_obj.suggest(args)
        hp = _hparams_from_args(args)
        print(f"\n=== TRIAL {i + 1}/{sweep_trials} === { {k: round(v, 6) for k, v in hp.items()} }", flush=True)

        trial = deepcopy(args)
        trial["train"]["total_timesteps"] = trial_steps
        trial["env"]["boss_hp_max"] = sweep_boss_hp
        t0 = time.time()
        try:
            trainer, _, evals = train_continuous(
                trial,
                trial_steps,
                hp["ramp_frac"],
                hp["rew_approach"],
                n_snakes_max,
                out,
                use_wandb=False,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                ladder_episodes=ladder_episodes,
                boss_hp=sweep_boss_hp,
            )
            trainer.close()
        except Exception as exc:
            print(f"TRIAL {i + 1} FAILED: {exc}", flush=True)
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue

        if not evals:
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue
        peak = max(e["depth"] for e in evals)
        print(
            f"TRIAL {i + 1}: depth final={evals[-1]['depth']:.3f} peak={peak:.3f} (d=1 clear final={evals[-1]['clear_d1']:.3f})",
            flush=True,
        )
        for e in evals:  # cost-aware trajectory: curriculum depth at uptime e['uptime'], cost in timesteps
            args["train"]["total_timesteps"] = e["step"]
            sweep_obj.observe(args, float(e["depth"]), float(e["uptime"]))
        args["train"]["total_timesteps"] = trial_steps
        if peak > best["depth"]:
            best = {"depth": peak, "hp": hp}
            print(f"  >> new best curriculum depth {peak:.3f}", flush=True)

    print(f"\n==== SWEEP DONE. best curriculum depth (boss_hp={sweep_boss_hp:g}): {best['depth']:.3f} ====", flush=True)
    print(f"==== BEST CONFIG: {best['hp']} ====", flush=True)
    return best["hp"]


def main() -> None:
    p = argparse.ArgumentParser(description="Protein sweep only (no full run). For the one-command flow use train.py.")
    p.add_argument("--sweep-trials", type=int, default=16)
    p.add_argument("--trial-steps", type=int, default=35_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--sweep-boss-hp", type=float, default=7500.0)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--n-snakes-max", type=int, default=N_SNAKES_MAX)
    p.add_argument("--out-dir", default="checkpoints/sweep")
    a = p.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    sys.argv = [sys.argv[0]]  # keep pufferl.load_config's argparse out of our args

    out = repo / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    best = run_sweep(a.num_envs, a.sweep_trials, a.trial_steps, a.sweep_boss_hp, a.eval_episodes, a.n_snakes_max, out)
    print(
        f"\n== BEST CONFIG: {best} ==\n   run it full: python3 train.py --wandb --no-sweep "
        + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in (best or {}).items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
