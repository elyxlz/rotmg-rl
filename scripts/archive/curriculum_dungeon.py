"""Adaptive curriculum for the faithful dungeon: easy -> hard, warm-started each stage.

Flat cold-start does not clear the faithful env (proven: cleared=0). This ramps difficulty:
start in the fight vs a weak boss with no snakes/grenades, then add HP, grenades, minions, snakes,
and finally shift spawns to random-anywhere + entrance (navigation + coverage, less overfitting).
Each stage warm-starts from the previous stage's policy. Monitor `cleared` in the logs.

    WANDB_MODE=offline PYTHONUNBUFFERED=1 uv run python scripts/curriculum_dungeon.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

CKPT = pathlib.Path("checkpoints")
COMMON = ["--num-envs", "256", "--num-workers", "16", "--backend", "multiprocessing", "--wandb", "--data-dir", "checkpoints/run"]

# each stage: (name, steps, extra args). spawn-in-room/random-spawn shift fight->navigation.
STAGES = [
    ("s1", 12_000_000, ["--boss-hp", "500", "--n-snakes", "0", "--no-grenades", "--no-minions", "--spawn-in-room-prob", "1.0"]),
    ("s2", 12_000_000, ["--boss-hp", "1500", "--n-snakes", "0", "--no-grenades", "--no-minions", "--spawn-in-room-prob", "1.0"]),
    ("s3", 15_000_000, ["--boss-hp", "3000", "--n-snakes", "0", "--no-minions", "--spawn-in-room-prob", "1.0"]),
    ("s4", 15_000_000, ["--boss-hp", "5000", "--n-snakes", "12", "--spawn-in-room-prob", "0.9"]),
    ("s5", 20_000_000, ["--boss-hp", "7500", "--n-snakes", "40", "--spawn-in-room-prob", "0.6", "--random-spawn-prob", "0.3"]),
    ("s6", 25_000_000, ["--boss-hp", "7500", "--n-snakes", "40", "--spawn-in-room-prob", "0.25", "--random-spawn-prob", "0.45"]),
]


def main() -> None:
    CKPT.mkdir(exist_ok=True)
    py = sys.executable
    prev = None
    for name, steps, extra in STAGES:
        save = str(CKPT / f"dungeon-{name}.pt")
        cmd = [py, "scripts/train_dungeon.py", "--total-timesteps", str(steps), "--save-path", save, *COMMON, *extra]
        if prev:
            cmd += ["--init-checkpoint", prev]
        print(f"\n=== STAGE {name} ({steps} steps) ===\n$ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"stage {name} failed ({r.returncode}); stopping", flush=True)
            return
        prev = save
        print(f"=== STAGE {name} DONE -> {save} ===", flush=True)
    print(f"\ncurriculum complete; final policy: {prev}", flush=True)


if __name__ == "__main__":
    main()
