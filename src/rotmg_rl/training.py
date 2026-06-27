"""One command, cold-start -> full Snake Pit clear on PufferLib 4.0 (--slowly CNN), as ONE process
and ONE continuous wandb run, driven by a SMOOTH difficulty schedule (no discrete phases).

    python3 train.py --wandb            # the whole flow (~460M steps, ~2-3h on one 3090)

By default it runs a Protein hyperparameter sweep (rotmg_rl.sweep) then trains the winner; --no-sweep
trains the full schedule directly with the good defaults. A single difficulty d(t) in [0,1] ramps over
the first `--ramp-frac` of training (cosine ease, see rotmg_rl.schedule), then holds at 1.0, driving
spawn distance + threat density + boss intensity jointly so the policy always faces
slightly-harder-than-mastered. We re-create the vecenv from the d-derived config every `--refresh`
steps (the policy + optimizer + global LR anneal persist across the swap -> one continuous run). Final
policy -> checkpoints/curriculum/finish.pt; eval with `python -m rotmg_rl.eval`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import _OBS_DTYPE_MAP, _CudaPtr, _cpu_tensor, PuffeRL, load_policy  # vec-buffer binding internals

from rotmg_rl.eval import eval_clear_rate
from rotmg_rl.schedule import BOSS_HP, N_SNAKES_MAX, apply_difficulty, difficulty_at, difficulty_config
from rotmg_rl.sweep import apply_hparams, build_sweep_config, run_sweep
from rotmg_rl.video import render_rollout

REPO = pathlib.Path(__file__).resolve().parents[2]
REFRESH_STEPS = 5_000_000  # re-derive the env config from d this often during the ramp
SAVE_EVERY = 8_000_000  # rolling checkpoint + POV video cadence


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
                    imageio.mimsave(vid, render_rollout(policy), fps=15)
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


def main() -> None:
    p = argparse.ArgumentParser(description="THE Snake Pit entry point: by default sweep hyperparameters then train the winner.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-sweep", action="store_true", help="skip the sweep; train the full schedule directly with the current good defaults")
    p.add_argument("--out-dir", default="checkpoints/curriculum")
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

        wandb.init(project="rotmg-dungeon", name=f"continuous-{int(time.time())}", group="continuous", config=hp)
    args = build_args(a.num_envs, hp, a.full_steps, a.boss_hp)
    trainer, policy, _ = train_continuous(args, a.full_steps, hp["ramp_frac"], hp["rew_approach"], a.n_snakes_max, out, a.wandb, boss_hp=a.boss_hp)
    final = out / "finish.pt"
    trainer.save_weights(str(final))
    trainer.close()

    rate = eval_clear_rate(policy, a.eval_episodes, d=1.0, boss_hp=a.boss_hp, n_snakes_max=a.n_snakes_max)
    print(f"\n== DONE -> {final} ({a.full_steps / 1e6:.0f}M steps) ==", flush=True)
    print(f"== FULL-DIFFICULTY (d=1) CLEAR RATE: {rate:.1%} over {a.eval_episodes} eps "
          f"(robust 100-ep number: python -m rotmg_rl.eval --checkpoint {final}) ==", flush=True)
    if a.wandb:
        import wandb

        wandb.log({"final/clear_rate_d1": rate})
        wandb.finish()


if __name__ == "__main__":
    main()
