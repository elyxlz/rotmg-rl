"""Adaptive cold-start curriculum toward the full Snake Pit boss (the M3 gate).

Difficulty is a single scalar d in [0,1]: boss HP 50 -> 250 and fire-interval 28 -> 18.
Each chunk warm-starts from the previous checkpoint and trains at the current d, then
greedy-evals. Advance d only when the level is solved; if a level stalls, keep training more
chunks at the same d. M3 = full boss (d=1.0) greedy clear >= 0.90.

    uv run --extra train python scripts/curriculum.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

HP_MIN, HP_MAX = 50.0, 250.0
FIRE_MIN, FIRE_MAX = 28, 18  # easy -> hard (smaller interval = denser fire)
INC = 0.125
CHUNK_STEPS = 6_000_000
ADVANCE_THRESH = 0.80
M3_THRESH = 0.90
MAX_CHUNKS = 40
EVAL_EPISODES = 150

COMMON_TRAIN = [
    "--num-envs", "256", "--num-steps", "64",
    "--backend", "multiprocessing", "--num-workers", "16",
    "--video-interval", "40",
]


def difficulty(d: float) -> tuple[float, int]:
    hp = HP_MIN + d * (HP_MAX - HP_MIN)
    fire = round(FIRE_MIN + d * (FIRE_MAX - FIRE_MIN))
    return hp, fire


def run(cmd: list[str]) -> str:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout[-1500:])
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr[-1500:])
        raise SystemExit(f"command failed ({proc.returncode})")
    return proc.stdout


def main() -> None:
    pathlib.Path("checkpoints").mkdir(exist_ok=True)
    py = sys.executable
    d = 0.0
    ckpt: str | None = None
    retries = 0
    for chunk_i in range(1, MAX_CHUNKS + 1):
        hp, fire = difficulty(d)
        name = f"curr-c{chunk_i:02d}"
        train_cmd = [
            py, "scripts/train.py", "--name", name,
            "--boss-hp", f"{hp:.1f}", "--boss-fire-interval", str(fire),
            "--total-timesteps", str(CHUNK_STEPS), *COMMON_TRAIN,
        ]
        if ckpt:
            train_cmd += ["--init-checkpoint", ckpt]
        run(train_cmd)
        ckpt = f"checkpoints/{name}.pt"

        eval_out = run([
            py, "scripts/eval_policy.py", "--checkpoint", ckpt,
            "--episodes", str(EVAL_EPISODES), "--boss-hp", f"{hp:.1f}", "--boss-fire-interval", str(fire),
        ])
        m = re.search(r"clear_rate (\d+\.\d+)", eval_out)
        cr = float(m.group(1)) if m else 0.0
        print(f"\n=== CHUNK {chunk_i} d={d:.3f} hp={hp:.0f} fire={fire} greedy_clear={cr:.3f} retries={retries} ===", flush=True)

        at_full = d >= 0.999
        if at_full and cr >= M3_THRESH:
            print(f"\n*** M3 REACHED: full boss greedy clear={cr:.3f} (checkpoint {ckpt}) ***", flush=True)
            break
        if (not at_full) and cr >= ADVANCE_THRESH:
            d = min(1.0, round(d + INC, 3))
            retries = 0
        else:
            retries += 1  # not solved (or full boss < M3 bar): train another chunk at this d
    else:
        print("\ncurriculum hit MAX_CHUNKS without M3; inspect and extend.", flush=True)

    print("final checkpoint:", ckpt, flush=True)


if __name__ == "__main__":
    main()
