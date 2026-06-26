"""Stochastic evaluation: run a trained policy over N episodes and report the TRUE per-episode
clear rate (the >=80% deliverable criterion) + stats. Training per-step `cleared` is a rate, not a
clear fraction; this measures what we actually care about.

    uv run --extra train python scripts/eval_dungeon.py --checkpoint checkpoints/dungeon.pt \
        --c-env --episodes 200 --boss-hp 7500 --no-grenades --no-minions --spawn-in-room-prob 1.0
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import pufferlib.emulation as emulation
import pufferlib.vector as pvector
from pufferlib.ocean import torch as ocean_torch
from rotmg_rl.puffer_policy import DungeonPolicy
from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv


def flatten(obs) -> np.ndarray:  # matches the C obs layout: [grid, minimap, scalars]
    return np.concatenate([obs["grid"].ravel(), obs["minimap"].ravel(), obs["scalars"]]).astype(np.float32)


@torch.no_grad()
def run_episode(policy, cfg, device, seed, max_steps):
    env = DungeonEnv(cfg)
    obs, _ = env.reset(seed=seed)
    state = {"lstm_h": None, "lstm_c": None, "hidden": None}
    for t in range(max_steps):
        x = torch.tensor(flatten(obs), device=device).unsqueeze(0)
        logits, _ = policy.forward_eval(x, state)
        action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return bool(info["cleared"]), t + 1, float(env.boss_hp / cfg.boss_hp_max)
    return False, max_steps, float(env.boss_hp / cfg.boss_hp_max)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--c-env", action="store_true", help="checkpoint is from the C env -> CDungeonPolicy")
    p.add_argument("--boss-hp", type=float, default=7500.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.0)
    p.add_argument("--random-spawn-prob", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    venv = pvector.make(emulation.GymnasiumPufferEnv, env_kwargs={"env_creator": DungeonEnv}, num_envs=1, backend=pvector.Serial)
    if args.c_env:
        from rotmg_rl.csim.policy import CDungeonPolicy

        policy = CDungeonPolicy(venv.driver_env, hidden_size=args.hidden)
    else:
        policy = DungeonPolicy(venv.driver_env, hidden_size=args.hidden)
    policy = ocean_torch.Recurrent(venv.driver_env, policy, input_size=args.hidden, hidden_size=args.hidden).to(device)
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
    print(f"=== EVAL {args.checkpoint} ({args.episodes} eps, boss_hp={args.boss_hp:g}) ===")
    print(f"  CLEAR RATE: {rate:.1%}   ({sum(clears)}/{args.episodes})")
    print(f"  avg episode length: {np.mean(lengths):.0f} steps")
    print(f"  avg boss_hp_frac at end: {np.mean(hp_left):.3f}  (cleared eps end at 0)")
    print(f"  >=80% deliverable: {'MET' if rate >= 0.8 else 'not met'}")


if __name__ == "__main__":
    main()
