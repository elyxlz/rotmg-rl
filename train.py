#!/usr/bin/env python3
"""One command, cold-start -> full Snake Pit clear on PufferLib 4.0 (--slowly CNN), as ONE process
and ONE continuous wandb run, driven by a SMOOTH difficulty schedule (no discrete phases).

    python3 train.py --wandb            # the whole schedule (~460M steps, ~2-3h on one 3090)

Self-bootstrapping: if `.venv4` (the 4.0 stack) is missing it runs scripts/setup_box_puffer4.sh
once, then re-execs itself under .venv4.

THE SCHEDULE (replaces the brittle 6-phase chain that cliffed 0.96 -> 0.00 the moment navigation +
all threats hit at once). A single difficulty d(t) in [0,1] ramps over the first `--ramp-frac` of
training (cosine ease), then holds at 1.0. d drives the env config JOINTLY so the policy always
faces slightly-harder-than-mastered, never a cliff:
  - spawn distance:  in-room (spawn_in_room_prob 1.0) -> entrance (0.05); the per-episode roll makes
                     a mid-d batch span both, a natural band of spawn distances.
  - threat density:  n_snakes 0 -> 40 (with per-episode n_snakes_jitter spreading the batch), and
                     grenades / minions toggling on once the early fight is mastered.
  - boss intensity:  passive point-blank target at the bottom (learn aim+kill+cast) -> shooting, the
                     blade cooldown tightening from gentle (40) to real (15).
We re-create the vecenv from the d-derived config every `--refresh` steps (the policy + optimizer +
global LR anneal persist across the swap -> one continuous run, LR alive to the end). Final policy
-> checkpoints/curriculum4/finish.pt; eval with scripts/eval_dungeon4.py.
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent
VENV_PY = REPO / ".venv4" / "bin" / "python"

# --- bootstrap: ensure the 4.0 stack exists, then run under .venv4 ---
if pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
    import subprocess

    if not VENV_PY.exists():
        print("== .venv4 missing -> provisioning the 4.0 stack (one-time) ==", flush=True)
        subprocess.run(["bash", str(REPO / "scripts" / "setup_box_puffer4.sh")], check=True)
    os.execv(str(VENV_PY), [str(VENV_PY), str(REPO / "train.py"), *sys.argv[1:]])

# --- below runs under .venv4 (PufferLib 4.0 + our env compiled into _C) ---
import argparse
import math
import time
from copy import deepcopy

import torch

import pufferlib
import pufferlib.sweep
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy
from pufferlib.torch_pufferl import _OBS_DTYPE_MAP, _CudaPtr, _cpu_tensor  # vec-buffer binding internals

N_SNAKES_MAX = 40
BOSS_HP = 7500.0
REFRESH_STEPS = 5_000_000  # re-derive the env config from d this often during the ramp
SAVE_EVERY = 8_000_000  # rolling checkpoint + POV video cadence


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def difficulty_at(step: int, total: int, ramp_frac: float) -> float:
    """d(t) in [0,1]: cosine ease-in-out over the first ramp_frac of training, then hold at 1.0."""
    ramp_steps = max(1.0, ramp_frac * total)
    x = _clamp01(step / ramp_steps)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def difficulty_config(d: float, n_snakes_max: int = N_SNAKES_MAX) -> dict:
    """Map a single difficulty d in [0,1] to the env-config levers. Monotonic in d, and every lever
    moves a little for each step of d so there is no cliff. The boolean toggles (boss_shoots /
    grenades / minions) fire at LOW d, while the rest of the difficulty is still gentle, so each new
    threat is met in near-isolation and then grows with d."""
    d = _clamp01(d)
    n = round(n_snakes_max * d)
    lerp = lambda a, b: a + (b - a) * d  # noqa: E731
    return dict(
        spawn_in_room_prob=lerp(1.0, 0.05),
        spawn_in_room_radius=lerp(6.0, 14.0),
        n_snakes=n,
        n_snakes_jitter=round(0.35 * n),
        enable_grenades=1 if d > 0.15 else 0,
        enable_minions=1 if d > 0.45 else 0,
        boss_shoots=1 if d > 0.05 else 0,
        blade_cd=round(15 + (40 - 15) * (1.0 - _clamp01(d / 0.4))),  # 40 (gentle) -> 15 (real) by d=0.4
    )


def _render_rollout(policy, max_frames=1200):
    """In-process POV video of one full-difficulty episode (the training C env is headless, so this
    runs the current policy on the numpy DungeonEnv). Logged to the SAME wandb run -- no follower."""
    import numpy as np

    from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv

    device = next(policy.parameters()).device
    env = DungeonEnv(DungeonConfig(boss_hp_max=BOSS_HP, n_snakes=N_SNAKES_MAX, spawn_in_room_prob=0.0), render_mode="rgb_array")
    obs, _ = env.reset(seed=777)
    state = policy.initial_state(1, device)
    frames = []
    with torch.no_grad():
        for _ in range(max_frames):
            flat = np.concatenate([obs["grid"].ravel(), obs["minimap"].ravel(), obs["scalars"]]).astype(np.float32)
            logits, _, state = policy.forward_eval(torch.tensor(flat, device=device).unsqueeze(0), state)
            action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
            obs, _, term, trunc, _ = env.step(action)
            frames.append(env.render())
            if term or trunc:
                break
    return frames


def eval_clear_rate(policy, episodes: int, d: float = 1.0, boss_hp: float = BOSS_HP, n_snakes_max: int = N_SNAKES_MAX, seed0: int = 50_000) -> float:
    """TRUE per-episode clear rate at difficulty d (default full) on the numpy DungeonEnv -- the sweep
    objective. Eval uses the deterministic full-difficulty spawn (always entrance) at d=1."""
    import numpy as np

    from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv

    device = next(policy.parameters()).device
    cfg_kw = difficulty_config(d, n_snakes_max)
    cfg_kw["spawn_in_room_prob"] = 0.0 if d >= 1.0 else cfg_kw["spawn_in_room_prob"]
    cfg_kw["n_snakes_jitter"] = 0  # eval the nominal density, not the training band
    cfg = DungeonConfig(boss_hp_max=boss_hp, **cfg_kw)
    clears = 0
    with torch.no_grad():
        for i in range(episodes):
            env = DungeonEnv(cfg)
            obs, _ = env.reset(seed=seed0 + i)
            state = policy.initial_state(1, device)
            for _ in range(cfg.max_steps):
                flat = np.concatenate([obs["grid"].ravel(), obs["minimap"].ravel(), obs["scalars"]]).astype(np.float32)
                logits, _, state = policy.forward_eval(torch.tensor(flat, device=device).unsqueeze(0), state)
                action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    clears += int(info["cleared"])
                    break
    return clears / max(1, episodes)


def _bind_vec(trainer: PuffeRL, vec) -> None:
    """Re-point a live PuffeRL trainer at a freshly-created vecenv (same shapes) without disturbing
    the policy/optimizer/global_step/epoch -> LR keeps annealing over the whole run. Mirrors the vec
    binding PuffeRL.__init__ does, so a difficulty refresh is just a buffer re-bind + reset."""
    n = trainer.total_agents
    trainer._vec = vec
    trainer.gpu = vec.gpu
    obs_dtype = _OBS_DTYPE_MAP.get(vec.obs_dtype, torch.uint8)
    if vec.gpu:
        trainer.vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr, (n, vec.obs_size), obs_dtype))
        trainer.vec_rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr, (n,), torch.float32))
        trainer.vec_terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr, (n,), torch.float32))
    else:
        trainer.vec_obs = _cpu_tensor(vec.obs_ptr, (n, vec.obs_size), obs_dtype)
        trainer.vec_rewards = _cpu_tensor(vec.rewards_ptr, (n,), torch.float32)
        trainer.vec_terminals = _cpu_tensor(vec.terminals_ptr, (n,), torch.float32)
    vec.reset()


def apply_difficulty(args: dict, d: float, rew_approach: float, n_snakes_max: int) -> None:
    """Write the d-derived env config (+ the swept reward levers) into args['env'] in place."""
    for k, v in difficulty_config(d, n_snakes_max).items():
        args["env"][k] = v
    args["env"]["boss_hp_max"] = args["env"].get("boss_hp_max", BOSS_HP)
    # navigation shaping is only useful once the spawn is far from the boss; off while the fight is
    # still in-room (low d), scaled by the swept base rate otherwise.
    args["env"]["rew_approach"] = rew_approach if d > 0.2 else 0.0


# Swept hyperparameters, by the pufferl-config section they live in. ramp_frac is schedule-only
# (PuffeRL ignores it; train_continuous reads it). Every key here is verified to be read by 4.0
# PuffeRL / load_policy / the env Config -- no dead dimensions (update_epochs/vtrace are NOT read; we
# don't sweep horizon since minibatch_size % horizon must hold, nor the clip coeffs per the guide).
SWEEP_TRAIN = ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "minibatch_size", "ramp_frac")
SWEEP_POLICY = ("hidden_size", "num_layers")
SWEEP_ENV = ("rew_approach", "rew_boss_dmg", "rew_clear", "rew_death", "rew_step")
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


def build_args(num_envs: int, hp: dict, total_steps: int, boss_hp: float) -> dict:
    """Load the dungeon config and apply the chosen hyperparameters (hp). load_config parses argv, so
    the caller must have cleared sys.argv of our own flags first."""
    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = num_envs
    args["vec"]["num_buffers"] = 1
    apply_hparams(args, hp)
    args["train"]["total_timesteps"] = total_steps  # drives LR annealing over the WHOLE run
    args["env"]["boss_hp_max"] = boss_hp
    return args


def train_continuous(args, total_steps, ramp_frac, rew_approach, n_snakes_max, out, use_wandb, cum_step=0, refresh_steps=REFRESH_STEPS, save_every=SAVE_EVERY, verbose=True, eval_every=0, eval_episodes=24, boss_hp=BOSS_HP):
    """Run the continuous-difficulty schedule under ONE PuffeRL trainer. Returns (trainer, policy,
    evals) where evals is a list of {step, uptime, clear_d1} sampled every `eval_every` steps (empty
    when eval_every<=0) -- the cost-aware trajectory the Protein sweep observes."""
    apply_difficulty(args, difficulty_at(0, total_steps, ramp_frac), rew_approach, n_snakes_max)
    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    evals: list[dict] = []
    last_log, last_save, last_refresh, last_eval, cur_d = 0.0, 0, 0, 0, -1.0
    while trainer.global_step < total_steps:
        if eval_every > 0 and (trainer.global_step - last_eval >= eval_every or trainer.global_step + trainer.total_agents * args["train"]["horizon"] >= total_steps):
            last_eval = trainer.global_step
            clear_d1 = eval_clear_rate(policy, eval_episodes, d=1.0, boss_hp=boss_hp, n_snakes_max=n_snakes_max)
            evals.append({"step": trainer.global_step, "uptime": trainer.uptime, "clear_d1": clear_d1})
            if verbose:
                print(f"[eval] step={trainer.global_step / 1e6:.0f}M d=1 clear_rate={clear_d1:.3f}", flush=True)
        if trainer.global_step - last_refresh >= refresh_steps or cur_d < 0.0:
            last_refresh = trainer.global_step
            d = difficulty_at(trainer.global_step, total_steps, ramp_frac)
            if abs(d - cur_d) > 1e-4:  # only rebuild the vecenv when d actually moved
                cur_d = d
                old = trainer._vec
                apply_difficulty(args, d, rew_approach, n_snakes_max)
                new_vec = _C.create_vec(args, _C.gpu)
                _bind_vec(trainer, new_vec)
                old.close()
                if verbose:
                    print(f"[schedule] step={trainer.global_step / 1e6:.0f}M d={d:.3f} env={difficulty_config(d, n_snakes_max)}", flush=True)

        trainer.rollouts()
        trainer.train()

        if trainer.global_step - last_save >= save_every:
            last_save = trainer.global_step
            trainer.save_weights(str(out / "latest.pt"))
            if use_wandb:
                import wandb
                import imageio.v2 as imageio

                try:
                    vid = str(out / "rollout.mp4")
                    imageio.mimsave(vid, _render_rollout(policy), fps=15)
                    wandb.log({"rollout": wandb.Video(vid, format="mp4")}, step=cum_step + trainer.global_step)
                except Exception as exc:  # a render hiccup must never kill training
                    print(f"[render] skipped: {exc}", flush=True)

        if time.time() - last_log > 1.0 or trainer.global_step >= total_steps:
            last_log = time.time()
            flat = dict(unroll_nested_dict(trainer.log()))  # env/ holds the legible end-of-episode metrics
            step = cum_step + trainer.global_step
            flat["difficulty"] = cur_d
            flat["global_step"] = step
            if verbose:
                print(f"[train] step={step / 1e6:.1f}M d={cur_d:.3f} SPS={flat.get('SPS', 0) / 1e3:.0f}K "
                      f"clear_rate={flat.get('env/clear_rate', 0):.2f} boss_left={flat.get('env/boss_hp_remaining', 0):.2f}", flush=True)
            if use_wandb:
                import wandb

                wandb.log(flat, step=step)
    return trainer, policy, evals


# --- Protein hyperparameter sweep (cost-aware Bayesian over the d=1 clear rate) ----------------------

def _space(distribution, lo, hi, mean):
    return {"distribution": distribution, "min": lo, "max": hi, "mean": mean, "scale": "auto"}


def build_sweep_config(metric: str = "clear_d1") -> dict:
    """Protein search space (15 knobs) -- give Protein the heavy lifting, few hand-set assumptions.
    Every knob is read by 4.0 PuffeRL / load_policy / the env Config (verified). gamma keeps the
    long-horizon attention (the lever that broke the 73% plateau); ramp_frac is the schedule's own knob.
    NOTE: 15 dims wants more than the default trial budget -- raise --sweep-trials (~40+) for a thorough
    search; 16-24 is a coarse pass. We deliberately do NOT sweep the clip coefficients (guide warning)
    nor horizon (minibatch_size % horizon must hold)."""
    return {
        "method": "Protein",
        "metric": metric,
        "metric_distribution": "linear",  # clear rate is a linear 0..1 objective
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
            "rew_death": _space("uniform", 0.1, 1.0, 0.5),
            "rew_step": _space("uniform", -0.003, 0.0, -0.001),
        },
    }


def run_sweep(num_envs, sweep_trials, trial_steps, sweep_boss_hp, eval_episodes, n_snakes_max, out) -> dict | None:
    """Protein sweep maximizing the TRUE d=1 clear rate (eval'd every eval_every steps -> the observed
    cost-aware uptime trajectory). Trials use the reduced sweep_boss_hp so the metric has gradient in a
    modest budget. Returns the best hyperparameter dict (None if every trial failed)."""
    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = num_envs
    args["vec"]["num_buffers"] = 1
    args["train"]["ramp_frac"] = 0.6  # schedule-only knob (not an ini key); seed it so trial 1 (defaults) carries it through observe
    sweep_config = build_sweep_config()
    method = sweep_config.pop("method")
    sweep_obj = getattr(pufferlib.sweep, method)(sweep_config)
    eval_every = max(1, trial_steps // 5)

    best = {"clear": -1.0, "hp": None}
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
            trainer, _, evals = train_continuous(trial, trial_steps, hp["ramp_frac"], hp["rew_approach"], n_snakes_max, out,
                                                 use_wandb=False, eval_every=eval_every, eval_episodes=eval_episodes, boss_hp=sweep_boss_hp)
            trainer.close()
        except Exception as exc:
            print(f"TRIAL {i + 1} FAILED: {exc}", flush=True)
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue

        if not evals:
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue
        peak = max(e["clear_d1"] for e in evals)
        print(f"TRIAL {i + 1}: d=1 clear final={evals[-1]['clear_d1']:.3f} peak={peak:.3f}", flush=True)
        for e in evals:  # cost-aware trajectory: clear_d1 at uptime e['uptime'], cost in timesteps
            args["train"]["total_timesteps"] = e["step"]
            sweep_obj.observe(args, float(e["clear_d1"]), float(e["uptime"]))
        args["train"]["total_timesteps"] = trial_steps
        if peak > best["clear"]:
            best = {"clear": peak, "hp": hp}
            print(f"  >> new best d=1 clear {peak:.3f}", flush=True)

    print(f"\n==== SWEEP DONE. best d=1 clear (boss_hp={sweep_boss_hp:g}): {best['clear']:.3f} ====", flush=True)
    print(f"==== BEST CONFIG: {best['hp']} ====", flush=True)
    return best["hp"]


def main() -> None:
    p = argparse.ArgumentParser(description="THE Snake Pit entry point: by default sweep hyperparameters then train the winner.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-sweep", action="store_true", help="skip the sweep; train the full schedule directly with the current good defaults")
    p.add_argument("--out-dir", default="checkpoints/curriculum4")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--full-steps", type=int, default=460_000_000, help="length of the final full-difficulty (7500-HP) run")
    # sweep knobs
    p.add_argument("--sweep-trials", type=int, default=24, help="Protein trials. The 15-knob space wants ~40+ for a thorough search; 24 is a coarse pass.")
    p.add_argument("--trial-steps", type=int, default=35_000_000, help="steps per sweep trial")
    p.add_argument("--sweep-boss-hp", type=float, default=7500.0, help="boss HP for sweep trials (default = the real 7500 target; the schedule's easy early-d gives gradient. Lower it only if a task genuinely has none)")
    p.add_argument("--eval-episodes", type=int, default=24, help="d=1 clear-rate eval episodes (per sweep eval + the final eval)")
    # full-run hyperparameters (defaults for --no-sweep / the fallback if the sweep finds nothing)
    p.add_argument("--ramp-frac", type=float, default=0.6, help="fraction of training spent ramping d 0->1 (rest at d=1)")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--gamma", type=float, default=0.97, help="long-horizon discount (the value that broke the 73% plateau)")
    p.add_argument("--gae-lambda", type=float, default=0.85)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--rew-approach", type=float, default=0.02, help="navigation distance-shaping rate (gated to d>0.2)")
    p.add_argument("--rew-boss-dmg", type=float, default=1.0)
    p.add_argument("--boss-hp", type=float, default=BOSS_HP, help="boss HP for the full run + final eval")
    p.add_argument("--n-snakes-max", type=int, default=N_SNAKES_MAX)
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    a = p.parse_args()

    mode = "DIRECT (--no-sweep)" if a.no_sweep else f"SWEEP ({a.sweep_trials} trials x {a.trial_steps / 1e6:.0f}M @ boss_hp {a.sweep_boss_hp:g}) -> winner"
    print(f"== train.py: {mode} -> full {a.full_steps / 1e6:.0f}M-step {a.boss_hp:g}-HP run, ramp_frac={a.ramp_frac} ==", flush=True)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = difficulty_at(int(frac * a.full_steps), a.full_steps, a.ramp_frac)
        print(f"  t={frac:.2f}  d={d:.3f}  {difficulty_config(d, a.n_snakes_max)}", flush=True)
    if not a.no_sweep:
        sc = build_sweep_config()
        knobs = {k: v["distribution"] for sect in ("train", "policy", "env") for k, v in sc[sect].items()}
        print(f"  sweep: {len(knobs)} knobs {knobs}", flush=True)
    if a.dry_run:
        return

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]  # load_config parses argv; keep our flags out of its way

    # the CLI-default hyperparameters (the --no-sweep config + the sweep-empty fallback). Only the
    # original good-default knobs; the rest stay at the ini defaults (apply_hparams sets only what's present).
    cli_hp = dict(learning_rate=a.lr, gamma=a.gamma, gae_lambda=a.gae_lambda, ent_coef=a.ent_coef, ramp_frac=a.ramp_frac,
                  hidden_size=a.hidden_size, rew_approach=a.rew_approach, rew_boss_dmg=a.rew_boss_dmg)

    # pick the hyperparameters: sweep for the winner, or the CLI defaults on --no-sweep.
    if a.no_sweep:
        hp = cli_hp
    else:
        hp = run_sweep(a.num_envs, a.sweep_trials, a.trial_steps, a.sweep_boss_hp, a.eval_episodes, a.n_snakes_max, out)
        if hp is None:  # every trial failed -> fall back to the CLI defaults rather than abort
            print("== sweep found no usable config; falling back to defaults ==", flush=True)
            hp = cli_hp

    # the full-difficulty run with the chosen hyperparameters (one continuous wandb run).
    print(f"\n== FULL {a.full_steps / 1e6:.0f}M-step run @ boss_hp {a.boss_hp:g} with {hp} ==", flush=True)
    if a.wandb:
        import wandb

        wandb.init(project="rotmg-dungeon", name=f"continuous4-{int(time.time())}", group="continuous4", config=hp)
    args = build_args(a.num_envs, hp, a.full_steps, a.boss_hp)
    trainer, policy, _ = train_continuous(args, a.full_steps, hp["ramp_frac"], hp["rew_approach"], a.n_snakes_max, out, a.wandb, boss_hp=a.boss_hp)
    final = out / "finish.pt"
    trainer.save_weights(str(final))
    trainer.close()

    rate = eval_clear_rate(policy, a.eval_episodes, d=1.0, boss_hp=a.boss_hp, n_snakes_max=a.n_snakes_max)
    print(f"\n== DONE -> {final} ({a.full_steps / 1e6:.0f}M steps) ==", flush=True)
    print(f"== FULL-DIFFICULTY (d=1) CLEAR RATE: {rate:.1%} over {a.eval_episodes} eps "
          f"(robust 100-ep number: scripts/eval_dungeon4.py --checkpoint {final}) ==", flush=True)
    if a.wandb:
        import wandb

        wandb.log({"final/clear_rate_d1": rate})
        wandb.finish()


if __name__ == "__main__":
    main()
