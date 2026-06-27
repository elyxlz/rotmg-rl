"""Eval + video follower for a live `train.py` run. Polls the rolling `latest.pt` checkpoint and, on
each update (every ~8M steps), logs to wandb both LEGIBLE per-episode metrics and a fresh POV rollout
video -- so you can follow the policy getting better with real numbers and real footage, not the
per-step means PufferLib logs by default:

  clear_rate         fraction of episodes that KILL the boss (1.0 = always clears)
  boss_hp_remaining  mean boss HP fraction at episode END (0.0 = killed)
  episode_length     mean steps per episode
  death_rate         fraction of episodes where the player died
  rollout            a POV mp4 of one full-dungeon episode (entrance -> fight)

Logged for both the in-room combat config and the full deliverable config. Run alongside training:
    .venv4/bin/python scripts/eval_follow.py --wandb
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dungeon4 import build_policy, flatten, run_episode  # noqa: E402

from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv  # noqa: E402


def eval_config(policy, device, spawn_in_room: float, episodes: int) -> dict:
    cfg = DungeonConfig(boss_hp_max=7500.0, n_snakes=40, spawn_in_room_prob=spawn_in_room)
    clears, lengths, hp_end = [], [], []
    for i in range(episodes):
        cleared, steps, boss_hp = run_episode(policy, cfg, device, seed=20_000 + i, max_steps=cfg.max_steps)
        clears.append(cleared)
        lengths.append(steps)
        hp_end.append(boss_hp)
    clears, lengths, hp_end = np.array(clears), np.array(lengths), np.array(hp_end)
    died = (~clears) & (lengths < cfg.max_steps)  # terminated short of max_steps = player death
    return {
        "clear_rate": float(clears.mean()),
        "boss_hp_remaining": float(hp_end.mean()),
        "episode_length": float(lengths.mean()),
        "death_rate": float(died.mean()),
    }


@torch.no_grad()
def render_episode(policy, device, spawn_in_room: float, max_frames: int) -> tuple[list, bool]:
    cfg = DungeonConfig(boss_hp_max=7500.0, n_snakes=40, spawn_in_room_prob=spawn_in_room)
    env = DungeonEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=777)
    state = policy.initial_state(1, device)
    frames, cleared = [], False
    for _ in range(max_frames):
        x = torch.tensor(flatten(obs), device=device).unsqueeze(0)
        logits, _, state = policy.forward_eval(x, state)
        action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
        obs, _, term, trunc, info = env.step(action)
        frames.append(env.render())
        if term or trunc:
            cleared = bool(info["cleared"])
            break
    return frames, cleared


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch-dir", default="checkpoints/curriculum4")
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--poll", type=int, default=30)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.wandb:
        import wandb

        wandb.init(project="rotmg-dungeon", name=f"faithful-eval-{int(time.time())}", group="faithful-eval")
    seen, step = None, 0
    while True:
        ckpts = glob.glob(os.path.join(args.watch_dir, "*.pt"))
        latest = max(ckpts, key=os.path.getmtime) if ckpts else None
        sig = (latest, os.path.getmtime(latest)) if latest else None
        if latest and sig != seen:
            seen = sig
            policy = build_policy(args.hidden, args.num_layers, device)
            policy.load_state_dict(torch.load(latest, map_location=device))
            policy.eval()
            name = os.path.basename(latest).replace(".pt", "")
            inroom = eval_config(policy, device, 1.0, args.episodes)
            full = eval_config(policy, device, 0.0, args.episodes)
            frames, vid_cleared = render_episode(policy, device, 0.0, args.max_frames)
            print(
                f"[{name}] IN-ROOM clear={inroom['clear_rate']:.0%} boss_left={inroom['boss_hp_remaining']:.0%} "
                f"death={inroom['death_rate']:.0%} | FULL clear={full['clear_rate']:.0%} "
                f"boss_left={full['boss_hp_remaining']:.0%} len={full['episode_length']:.0f} | "
                f"vid {len(frames)}f cleared={vid_cleared}",
                flush=True,
            )
            if args.wandb:
                import wandb

                row = {f"inroom/{k}": v for k, v in inroom.items()}
                row.update({f"full/{k}": v for k, v in full.items()})
                vid_path = f"/tmp/follow_{step}.mp4"
                imageio.mimsave(vid_path, frames, fps=args.fps)
                row["rollout"] = wandb.Video(vid_path, format="mp4")
                wandb.log(row, step=step)
                step += 1
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
