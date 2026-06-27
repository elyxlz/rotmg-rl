"""One-command, cold-start curriculum: trains the full Snake Pit policy from SCRATCH to the final
~95% full-dungeon clear, by running the proven warm-start stages in sequence (no checkpoint needed).

    uv run --extra train python scripts/train_curriculum.py --c-env --wandb

Cold-starts stage 1, warm-starts each subsequent stage from the previous one, and ends with
`checkpoints/curriculum/finish.pt` (the deliverable policy). `--dry-run` prints the plan + ETA only.

The recipe (see PROGRESS.md, the 0% -> 60% -> 73% -> 95% trajectory): learn the in-room point-blank
nuke (gamma 0.95, myopic = fast kill), add all threats in-room, then a spawn-distance ramp to fuse
navigation onto combat, and finally the gamma 0.97 "finish" stage -- the lever that broke the 73%
plateau (the ~600-step navigate-then-clear is a long horizon that needs to value the eventual clear).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

# (name, env/hparam args, total_timesteps). Each stage warm-starts from the previous; stage 1 is cold.
STAGES: list[tuple[str, str, int]] = [
    ("passive", "--boss-hp 7500 --no-boss-shoots --n-snakes 0 --no-grenades --no-minions --spawn-in-room-prob 1.0 --gamma 0.95", 30_000_000),
    ("shooting", "--boss-hp 7500 --n-snakes 0 --no-grenades --no-minions --spawn-in-room-prob 1.0 --gamma 0.95", 50_000_000),
    ("combat", "--boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 1.0 --gamma 0.95", 50_000_000),
    ("combine1", "--boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.4 --rew-approach 0.02 --gamma 0.95", 80_000_000),
    ("combine2", "--boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.1 --rew-approach 0.02 --gamma 0.95", 100_000_000),
    ("finish", "--boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.05 --rew-approach 0.02 --gamma 0.97 --gae-lambda 0.85", 150_000_000),
]

SPS_ESTIMATE = 60_000  # observed C-env throughput on one 3090; used for the ETA only


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--c-env", action="store_true", help="train on the fast C env (recommended)")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--out-dir", default="checkpoints/curriculum")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the plan + ETA, run nothing")
    args = p.parse_args()

    total = sum(s for _, _, s in STAGES)
    eta_h = total / SPS_ESTIMATE / 3600
    print(f"=== curriculum: {len(STAGES)} stages, {total/1e6:.0f}M total steps, ETA ~{eta_h:.1f}h on one 3090 ===", flush=True)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prev: pathlib.Path | None = None
    for name, cfg, steps in STAGES:
        save = out / f"{name}.pt"
        cmd = ["uv", "run", "--extra", "train", "python", "scripts/train_dungeon.py", *cfg.split()]
        cmd += ["--total-timesteps", str(steps), "--num-envs", str(args.num_envs), "--save-path", str(save), "--data-dir", str(out / f"run_{name}")]
        if args.c_env:
            cmd.append("--c-env")
        if args.wandb:
            cmd.append("--wandb")
        if prev is not None:
            cmd += ["--init-checkpoint", str(prev)]
        print(f"\n=== STAGE {name}: {steps/1e6:.0f}M steps {'(cold start)' if prev is None else f'(warm <- {prev.name})'} ===", flush=True)
        if args.dry_run:
            print(" ", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)
        prev = save

    print(f"\n=== DONE -> {prev} ===", flush=True)
    print("eval the full-dungeon clear rate with:")
    print(f"  uv run --extra train python scripts/eval_dungeon.py --checkpoint {prev} {'--c-env ' if args.c_env else ''}--episodes 100 --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.0")
    if not args.dry_run:
        eval_cmd = ["uv", "run", "--extra", "train", "python", "scripts/eval_dungeon.py", "--checkpoint", str(prev), "--episodes", "100", "--boss-hp", "7500", "--n-snakes", "40", "--spawn-in-room-prob", "0.0"]
        if args.c_env:
            eval_cmd.append("--c-env")
        subprocess.run(eval_cmd, check=False)


if __name__ == "__main__":
    main()
