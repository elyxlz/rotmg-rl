"""POV rollout video of a dungeon policy, painted from the read-only render-state of the single-env C
wrapper (the same C dynamics training acts on -- never a re-simulation).

`render_rollout` renders one full-difficulty episode of a live in-process policy (the training loop
logs it to the same wandb run); `main` watches a checkpoint dir and renders the newest --slowly
checkpoint to an mp4 (+ wandb) on an interval, like a live training follow-along:

    python -m rotmg_rl.video --watch checkpoints/dungeon --wandb --run-id <id>
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import time

import imageio.v2 as imageio
import torch

from rotmg_rl.config import DungeonConfig
from rotmg_rl.csim.render import render_pov
from rotmg_rl.csim.single import CDungeonSingle
from rotmg_rl.eval import build_policy
from rotmg_rl.schedule import BOSS_HP, N_SNAKES_MAX
from rotmg_rl.sim.snakepit_map import load_jm


def render_rollout(policy, max_frames: int = 1200):
    """In-process POV video of one full-difficulty episode on the single-env C wrapper, painted by the
    read-only render-state. Returns the frame list (the caller logs it to wandb)."""
    device = next(policy.parameters()).device
    walkable = load_jm().walkable
    env = CDungeonSingle(DungeonConfig(boss_hp_max=BOSS_HP, n_snakes=N_SNAKES_MAX, spawn_in_room_prob=0.0), seed=777)
    obs = env.reset(seed=777)
    state = policy.initial_state(1, device)
    frames = []
    with torch.no_grad():
        for _ in range(max_frames):
            frames.append(render_pov(env.render_state(), walkable))  # render the state the policy acts on
            logits, _, state = policy.forward_eval(torch.tensor(obs, device=device).unsqueeze(0), state)
            action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    env.close()
    return frames


@torch.no_grad()
def rollout(policy, cfg, device, seed, max_frames):
    walkable = load_jm().walkable
    env = CDungeonSingle(cfg, seed=seed)
    obs = env.reset(seed=seed)
    state = policy.initial_state(1, device)
    frames, cleared = [], False
    try:
        for _ in range(max_frames):
            frames.append(render_pov(env.render_state(), walkable))  # render the state the policy acts on
            x = torch.tensor(obs, device=device).unsqueeze(0)
            logits, _, state = policy.forward_eval(x, state)
            action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                cleared = bool(info["cleared"])
                break
        return frames, cleared
    finally:
        env.close()


def newest_ckpt(watch):
    files = glob.glob(os.path.join(watch, "**", "*.bin"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def step_of(ckpt: str) -> int:
    return int(pathlib.Path(ckpt).stem)  # checkpoints are named <global_step>.bin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", default="checkpoints/dungeon", help="dir of <run_id>/<step>.bin checkpoints")
    p.add_argument("--interval", type=int, default=120, help="seconds between renders")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--boss-hp", type=float, default=300.0)
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--once", action="store_true", help="render the newest checkpoint once and exit")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-id", default=None, help="training run id; videos go to a run named rollouts-<id>")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(args.hidden, args.num_layers, device)
    # watchable fight: spawn in the boss room (matches the passive-boss validation config)
    cfg = DungeonConfig(boss_hp_max=args.boss_hp, spawn_in_room_prob=1.0, random_spawn_prob=0.0, boss_shoots=not args.no_boss_shoots)

    if args.wandb:
        import wandb

        name = f"rollouts-{args.run_id}" if args.run_id else "rollouts"
        wandb.init(project="rotmg-dungeon", name=name, job_type="eval", group=os.environ.get("WANDB_RUN_GROUP"))
    pathlib.Path("videos").mkdir(exist_ok=True)

    last = None
    while True:
        ckpt = newest_ckpt(args.watch)
        if ckpt and ckpt != last:
            try:
                policy.load_state_dict(torch.load(ckpt, map_location=device))
                policy.eval()
                frames, cleared = rollout(policy, cfg, device, seed=int(time.time()) % 10000, max_frames=args.max_frames)
                out = f"videos/follow_{step_of(ckpt)}.mp4"
                imageio.mimsave(out, frames, fps=args.fps)
                if args.wandb:
                    import wandb

                    wandb.log({"rollout": wandb.Video(out, format="mp4"), "rollout_cleared": float(cleared)}, step=step_of(ckpt))
                print(f"rendered {ckpt} -> {out} (cleared={cleared}, frames={len(frames)})", flush=True)
                last = ckpt
            except Exception as e:  # checkpoint mid-write or arch mismatch; retry next loop
                print(f"skip {ckpt}: {e}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
