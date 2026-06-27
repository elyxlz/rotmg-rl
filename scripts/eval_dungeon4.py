"""TRUE per-episode clear-rate eval for a PufferLib 4.0 (--slowly / torch) dungeon checkpoint.

Mirrors scripts/eval_dungeon.py but for the 4.0 torch CNN: reconstructs
Policy(DungeonEncoder, DefaultDecoder, LSTM) (like follow_along4.py), loads the torch state_dict
checkpoint, and runs N stochastic episodes on the single-env C wrapper reporting the per-episode
clear fraction (the >=80% deliverable metric) -- NOT the per-step `cleared` rate the dashboard shows.

    .venv4/bin/python scripts/eval_dungeon4.py --checkpoint checkpoints4/dungeon/<run>/<step>.bin \
        --episodes 200 --boss-hp 300 --no-grenades --no-minions --no-boss-shoots --spawn-in-room-prob 1.0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import pufferlib.models as models
from rotmg_rl.config import DungeonConfig
from rotmg_rl.csim.single import OBS_SIZE, CDungeonSingle

ACT_SIZES = [9, 32, 2, 2]


def build_policy(hidden: int, num_layers: int, device):
    encoder = models.DungeonEncoder(OBS_SIZE, hidden)
    decoder = models.DefaultDecoder(ACT_SIZES, hidden)
    network = models.LSTM(hidden, num_layers=num_layers)
    return models.Policy(encoder, decoder, network).to(device)


@torch.no_grad()
def run_episode(policy, cfg, device, seed, max_steps):
    env = CDungeonSingle(cfg, seed=seed)
    obs = env.reset(seed=seed)
    state = policy.initial_state(1, device)
    try:
        for t in range(max_steps):
            x = torch.tensor(obs, device=device).unsqueeze(0)
            logits, _, state = policy.forward_eval(x, state)
            action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                return bool(info["cleared"]), t + 1, float(info["boss_hp_frac"])
        return False, max_steps, float(info["boss_hp_frac"])
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--boss-hp", type=float, default=7500.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.0)
    p.add_argument("--random-spawn-prob", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(args.hidden, args.num_layers, device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    cfg = DungeonConfig(
        boss_hp_max=args.boss_hp,
        n_snakes=args.n_snakes,
        enable_grenades=not args.no_grenades,
        enable_minions=not args.no_minions,
        boss_shoots=not args.no_boss_shoots,
        spawn_in_room_prob=args.spawn_in_room_prob,
        random_spawn_prob=args.random_spawn_prob,
    )
    clears, lengths, hp_left = [], [], []
    for i in range(args.episodes):
        c, n, hp = run_episode(policy, cfg, device, seed=10_000 + i, max_steps=cfg.max_steps)
        clears.append(c)
        lengths.append(n)
        hp_left.append(hp)
    rate = float(np.mean(clears))
    print(f"=== EVAL4 {args.checkpoint} ({args.episodes} eps, boss_hp={args.boss_hp:g}) ===")
    print(f"  CLEAR RATE: {rate:.1%}   ({sum(clears)}/{args.episodes})")
    print(f"  avg episode length: {np.mean(lengths):.0f} steps")
    print(f"  avg boss_hp_frac at end: {np.mean(hp_left):.3f}  (cleared eps end at 0)")
    print(f"  >=80% deliverable: {'MET' if rate >= 0.8 else 'not met'}")


if __name__ == "__main__":
    main()
