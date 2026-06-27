#!/usr/bin/env python3
"""Sweep-ONLY helper: find + print the best hyperparameter config WITHOUT launching the full run.

train.py is now THE entry point -- `python3 train.py --wandb` runs the Protein sweep then trains the
winner in one command (and `--no-sweep` skips straight to a direct run). This thin wrapper just calls
train.run_sweep (DRY: same search space + orchestration) for when you want to inspect the sweep result
on its own. Self-bootstraps under .venv4 like train.py."""

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

sys.path.insert(0, str(REPO))
import train as T  # the canonical sweep orchestration lives in train.py (imported under .venv4, no re-exec)


def main() -> None:
    p = argparse.ArgumentParser(description="Protein sweep only (no full run). For the one-command flow use train.py.")
    p.add_argument("--sweep-trials", type=int, default=16)
    p.add_argument("--trial-steps", type=int, default=35_000_000)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--sweep-boss-hp", type=float, default=4000.0)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--n-snakes-max", type=int, default=T.N_SNAKES_MAX)
    p.add_argument("--out-dir", default="checkpoints/sweep4")
    a = p.parse_args()
    sys.argv = [sys.argv[0]]  # keep pufferl.load_config's argparse out of our args

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    best = T.run_sweep(a.num_envs, a.sweep_trials, a.trial_steps, a.sweep_boss_hp, a.eval_episodes, a.n_snakes_max, out)
    print(f"\n== BEST CONFIG: {best} ==\n   run it full: python3 train.py --wandb --no-sweep "
          + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in (best or {}).items()), flush=True)


if __name__ == "__main__":
    main()
