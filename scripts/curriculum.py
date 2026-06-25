"""Staged cold-start curriculum toward the full Snake Pit boss (the M3 gate).

Each stage warm-starts from the previous stage's checkpoint, trains at a fixed difficulty,
then greedy-evals at that difficulty. Difficulty ramps boss HP + fire rate up to the full
boss (HP 600, fire-interval 18), whose >=90% greedy clear is M3.

Runs scripts/train.py and scripts/eval_policy.py as subprocesses (same venv interpreter).

    uv run --extra train python scripts/curriculum.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

# (boss_hp, fire_interval, total_timesteps)
STAGES = [
    (40.0, 26, 8_000_000),
    (80.0, 22, 8_000_000),
    (160.0, 20, 10_000_000),
    (320.0, 18, 12_000_000),
    (600.0, 18, 14_000_000),  # full boss = M3 gate
]

COMMON_TRAIN = [
    "--num-envs", "256", "--num-steps", "64",
    "--backend", "multiprocessing", "--num-workers", "16",
    "--video-interval", "40",
]


def run(cmd: list[str]) -> str:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-2000:])
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-2000:])
        raise SystemExit(f"command failed ({proc.returncode})")
    return proc.stdout


def main() -> None:
    pathlib.Path("checkpoints").mkdir(exist_ok=True)
    py = sys.executable
    prev_ckpt: str | None = None
    for i, (hp, fire, steps) in enumerate(STAGES, start=1):
        name = f"curr-s{i}"
        train_cmd = [
            py, "scripts/train.py", "--name", name,
            "--boss-hp", str(hp), "--boss-fire-interval", str(fire),
            "--total-timesteps", str(steps), *COMMON_TRAIN,
        ]
        if prev_ckpt:
            train_cmd += ["--init-checkpoint", prev_ckpt]
        run(train_cmd)
        ckpt = f"checkpoints/{name}.pt"

        eval_out = run([
            py, "scripts/eval_policy.py", "--checkpoint", ckpt,
            "--episodes", "200", "--boss-hp", str(hp), "--boss-fire-interval", str(fire),
        ])
        m = re.search(r"clear_rate (\d+\.\d+)", eval_out)
        cr = float(m.group(1)) if m else float("nan")
        print(f"\n=== STAGE {i} (hp={hp}, fire={fire}) greedy clear_rate={cr:.3f} ===", flush=True)
        prev_ckpt = ckpt

    print("\ncurriculum complete; final checkpoint:", prev_ckpt, flush=True)


if __name__ == "__main__":
    main()
