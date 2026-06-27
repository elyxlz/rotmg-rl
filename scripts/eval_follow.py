"""Intuitive eval follower for a live `train.py` run. Polls the latest curriculum4 phase checkpoint
and logs LEGIBLE per-episode metrics to wandb every time a new one lands -- the numbers that
actually mean something, instead of the per-step means PufferLib logs by default:

  clear_rate         fraction of episodes that KILL the boss (1.0 = always clears)
  boss_hp_remaining  mean boss HP fraction at episode END (0.0 = killed)
  episode_length     mean steps per episode
  death_rate         fraction of episodes where the player died

Logged for both the in-room combat config and the full deliverable config (entrance + threats).
Run alongside training:  .venv4/bin/python scripts/eval_follow.py --wandb
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dungeon4 import build_policy, run_episode  # noqa: E402

from rotmg_rl.sim.dungeon import DungeonConfig  # noqa: E402


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch-dir", default="checkpoints/curriculum4")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--poll", type=int, default=45)
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
            print(
                f"[{name}] IN-ROOM clear={inroom['clear_rate']:.0%} boss_left={inroom['boss_hp_remaining']:.0%} "
                f"death={inroom['death_rate']:.0%} | FULL clear={full['clear_rate']:.0%} "
                f"boss_left={full['boss_hp_remaining']:.0%} len={full['episode_length']:.0f}",
                flush=True,
            )
            if args.wandb:
                import wandb

                row = {f"inroom/{k}": v for k, v in inroom.items()}
                row.update({f"full/{k}": v for k, v in full.items()})
                wandb.log(row, step=step)
                step += 1
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
