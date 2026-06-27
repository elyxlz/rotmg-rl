#!/usr/bin/env python3
"""One command, cold-start -> ~95% full Snake Pit clear on PufferLib 4.0 (--slowly CNN), as ONE
process and ONE continuous wandb run.

    python3 train.py --wandb           # the whole curriculum (~460M steps, ~2-3h on one 3090)

Self-bootstrapping: if `.venv4` (the 4.0 stack) is missing it runs scripts/setup_box_puffer4.sh
once, then re-execs itself under .venv4. The proven curriculum (PROGRESS.md: 0->60->73->95%) runs as
six in-memory warm-started phases under a single wandb run, with the phase index logged as the
`phase` metric (not six separate runs). Drives the 4.0 trainer at the PuffeRL level: instantiate the
trainer fresh per phase (so LR anneals per phase) but reuse the policy in memory; the env config +
gamma/gae are the per-phase levers. Final policy -> checkpoints/curriculum4/finish.pt (a 4.0
torch state_dict; eval with scripts/eval_dungeon4.py). The 3.0 deploy keeps its own policy.
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
import time

import torch

import pufferlib
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy

# (name, env-config overrides, gamma, gae_lambda, total_timesteps). Mirrors scripts/train_curriculum.py
# (the proven 3.0 recipe): in-room point-blank kill -> all threats in-room -> spawn-distance ramp ->
# the gamma 0.97 "finish" that broke the 73% plateau. Each phase warm-starts from the previous policy.
PASSIVE = dict(boss_shoots=0, n_snakes=0, enable_grenades=0, enable_minions=0, spawn_in_room_prob=1.0, rew_approach=0.0)
FULL = dict(boss_shoots=1, n_snakes=40, enable_grenades=1, enable_minions=1)
STAGES = [
    ("passive", {**PASSIVE}, 0.95, 0.90, 30_000_000),
    ("shooting", {**PASSIVE, "boss_shoots": 1}, 0.95, 0.90, 50_000_000),
    ("combat", {**FULL, "spawn_in_room_prob": 1.0, "rew_approach": 0.0}, 0.95, 0.90, 50_000_000),
    ("combine1", {**FULL, "spawn_in_room_prob": 0.4, "rew_approach": 0.02}, 0.95, 0.90, 80_000_000),
    ("combine2", {**FULL, "spawn_in_room_prob": 0.1, "rew_approach": 0.02}, 0.95, 0.90, 100_000_000),
    ("finish", {**FULL, "spawn_in_room_prob": 0.05, "rew_approach": 0.02}, 0.97, 0.85, 150_000_000),
]


def run_phase(args, phase_idx, name, gamma, gae, steps, policy, cum_step, use_wandb):
    args["train"]["gamma"] = gamma
    args["train"]["gae_lambda"] = gae
    args["train"]["total_timesteps"] = steps
    args["vec"]["num_buffers"] = 1
    vec = _C.create_vec(args, _C.gpu)
    if policy is None:
        policy = load_policy(args, vec)  # cold start (phase 1)
    trainer = PuffeRL(args, vec, policy, verbose=False)
    last_log = 0.0
    while trainer.global_step < steps:
        trainer.rollouts()
        trainer.train()
        if time.time() - last_log > 1.0 or trainer.global_step >= steps:
            last_log = time.time()
            flat = dict(unroll_nested_dict(trainer.log()))
            step = cum_step + trainer.global_step
            flat["phase"] = phase_idx
            flat["global_step"] = step
            print(f"[{name}] step={step/1e6:.1f}M SPS={flat.get('SPS', 0)/1e3:.0f}K "
                  f"boss_hp={flat.get('env/boss_hp_frac', 0):.3f} score={flat.get('env/score', 0):.3f}", flush=True)
            if use_wandb:
                import wandb

                wandb.log(flat, step=step)
    end = cum_step + trainer.global_step
    return trainer, policy, end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--out-dir", default="checkpoints/curriculum4")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true", help="print the plan + ETA, run nothing")
    p.add_argument("--smoke", type=int, default=0, help="cap each phase to N steps to test the machinery")
    args_cli = p.parse_args()

    stages = STAGES if not args_cli.smoke else [(n, c, g, l, args_cli.smoke) for n, c, g, l, _ in STAGES]
    globals()["STAGES"] = stages
    total = sum(s for *_, s in STAGES)
    print(f"== curriculum: {len(STAGES)} phases, {total/1e6:.0f}M steps, ETA ~{total/52_000/3600:.1f}h on one 3090 ==", flush=True)
    for i, (name, cfg, gamma, gae, steps) in enumerate(STAGES):
        print(f"  phase {i} {name:9s} {steps/1e6:4.0f}M  gamma={gamma} gae={gae}  {cfg}", flush=True)
    if args_cli.dry_run:
        return

    out = REPO / args_cli.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = [sys.argv[0]]  # load_config parses argv; keep our flags out of its way
    args = pufferl.load_config("dungeon")
    args["vec"]["total_agents"] = args_cli.num_envs
    if args_cli.wandb:
        import wandb

        wandb.init(project="rotmg-dungeon", name=f"curriculum4-{int(time.time())}",
                   group="curriculum4", config={"stages": [s[0] for s in STAGES]})

    policy, cum = None, 0
    for i, (name, cfg, gamma, gae, steps) in enumerate(STAGES):
        for k, v in cfg.items():
            args["env"][k] = v
        args["env"]["boss_hp_max"] = 7500.0
        print(f"\n== PHASE {i}: {name} ({steps/1e6:.0f}M, gamma {gamma}) "
              f"{'COLD START' if policy is None else 'warm-start'} ==", flush=True)
        trainer, policy, cum = run_phase(args, i, name, gamma, gae, steps, policy, cum, args_cli.wandb)
        trainer.save_weights(str(out / f"{name}.pt"))
        trainer.close()

    final = out / "finish.pt"
    print(f"\n== DONE -> {final} ({cum/1e6:.0f}M steps) ==", flush=True)
    if args_cli.wandb:
        import wandb

        wandb.finish()
    print("eval the full-dungeon clear rate:\n"
          f"  {VENV_PY} scripts/eval_dungeon4.py --checkpoint {final} --episodes 100 "
          "--boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.0", flush=True)


if __name__ == "__main__":
    main()
