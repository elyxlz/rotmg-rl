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

import torch

import pufferlib
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


def build_args(num_envs: int, hidden_size: int, lr: float, gamma: float, gae_lambda: float, ent_coef: float, total_steps: int, boss_hp: float, rew_boss_dmg: float) -> dict:
    """Load the dungeon config and apply the (swept) hyperparameters. load_config parses argv, so the
    caller must have cleared sys.argv of our own flags first."""
    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = num_envs
    args["vec"]["num_buffers"] = 1
    args["policy"]["hidden_size"] = hidden_size
    t = args["train"]
    t["learning_rate"] = lr
    t["gamma"] = gamma
    t["gae_lambda"] = gae_lambda
    t["ent_coef"] = ent_coef
    t["total_timesteps"] = total_steps  # drives LR annealing over the WHOLE run
    args["env"]["boss_hp_max"] = boss_hp
    args["env"]["rew_boss_dmg"] = rew_boss_dmg
    return args


def train_continuous(args, total_steps, ramp_frac, rew_approach, n_snakes_max, out, use_wandb, cum_step=0, refresh_steps=REFRESH_STEPS, save_every=SAVE_EVERY, verbose=True):
    """Run the continuous-difficulty schedule under ONE PuffeRL trainer. Returns (trainer, policy)."""
    apply_difficulty(args, difficulty_at(0, total_steps, ramp_frac), rew_approach, n_snakes_max)
    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    last_log, last_save, last_refresh, cur_d = 0.0, 0, 0, -1.0
    while trainer.global_step < total_steps:
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
    return trainer, policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--out-dir", default="checkpoints/curriculum4")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=460_000_000)
    p.add_argument("--ramp-frac", type=float, default=0.6, help="fraction of training spent ramping d 0->1 (rest at d=1)")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--gamma", type=float, default=0.97, help="long-horizon discount (the value that broke the 73% plateau)")
    p.add_argument("--gae-lambda", type=float, default=0.85)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--rew-approach", type=float, default=0.02, help="navigation distance-shaping rate (gated to d>0.2)")
    p.add_argument("--rew-boss-dmg", type=float, default=1.0)
    p.add_argument("--boss-hp", type=float, default=BOSS_HP)
    p.add_argument("--n-snakes-max", type=int, default=N_SNAKES_MAX)
    p.add_argument("--eval-episodes", type=int, default=100, help="final full-difficulty (d=1) clear-rate eval")
    p.add_argument("--dry-run", action="store_true", help="print the schedule at a few d, run nothing")
    p.add_argument("--smoke", type=int, default=0, help="cap total steps to N to test the machinery")
    a = p.parse_args()

    total = a.smoke if a.smoke else a.total_steps
    print(f"== continuous schedule: {total / 1e6:.0f}M steps, ramp_frac={a.ramp_frac} "
          f"(ramp over {a.ramp_frac * total / 1e6:.0f}M, hold {(1 - a.ramp_frac) * total / 1e6:.0f}M at d=1) "
          f"ETA ~{total / 52_000 / 3600:.1f}h on one 3090 ==", flush=True)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = difficulty_at(int(frac * total), total, a.ramp_frac)
        print(f"  t={frac:.2f}  d={d:.3f}  {difficulty_config(d, a.n_snakes_max)}", flush=True)
    if a.dry_run:
        return

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0]]  # load_config parses argv; keep our flags out of its way

    if a.wandb:
        import wandb

        wandb.init(project="rotmg-dungeon", name=f"continuous4-{int(time.time())}", group="continuous4",
                   config={"ramp_frac": a.ramp_frac, "gamma": a.gamma, "lr": a.lr, "ent_coef": a.ent_coef,
                           "hidden_size": a.hidden_size, "rew_approach": a.rew_approach, "rew_boss_dmg": a.rew_boss_dmg})

    args = build_args(a.num_envs, a.hidden_size, a.lr, a.gamma, a.gae_lambda, a.ent_coef, total, a.boss_hp, a.rew_boss_dmg)
    trainer, policy = train_continuous(args, total, a.ramp_frac, a.rew_approach, a.n_snakes_max, out, a.wandb)
    final = out / "finish.pt"
    trainer.save_weights(str(final))
    trainer.close()

    rate = eval_clear_rate(policy, a.eval_episodes, d=1.0, boss_hp=a.boss_hp, n_snakes_max=a.n_snakes_max)
    print(f"\n== DONE -> {final} ({total / 1e6:.0f}M steps) ==", flush=True)
    print(f"== FULL-DIFFICULTY (d=1) CLEAR RATE: {rate:.1%} over {a.eval_episodes} episodes ==", flush=True)
    if a.wandb:
        import wandb

        wandb.log({"final/clear_rate_d1": rate})
        wandb.finish()


if __name__ == "__main__":
    main()
