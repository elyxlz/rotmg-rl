"""Ground-truth evaluation for the milestone gates (M3/M4).

Runs N deterministic (greedy) episodes and reports clear-rate from `info['cleared']` (not the
training-time terminal-reward proxy). M3 target: >=90% clear over >=200 episodes.

    uv run --extra train python scripts/eval_policy.py --checkpoint checkpoints/m2-learn-bh50.pt \
        --episodes 200 --boss-hp 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from rotmg_rl.policy import Agent
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def flatten_obs(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([d["grid"].ravel(), d["scalars"]]).astype(np.float32)


@torch.no_grad()
def run_episode(agent: Agent, env: SnakePitEnv, device, seed: int) -> tuple[bool, float, int]:
    obs, _ = env.reset(seed=seed)
    lstm_state = agent.initial_state(1, device)
    done = torch.zeros(1, device=device)
    total = 0.0
    for step in range(env.cfg.max_steps):
        x = torch.tensor(flatten_obs(obs), device=device).unsqueeze(0)
        action, lstm_state = agent.act_greedy(x, lstm_state, done)
        obs, reward, term, trunc, info = env.step(action[0].cpu().numpy())
        total += reward
        done = torch.tensor([float(term or trunc)], device=device)
        if term or trunc:
            return info["cleared"], total, step + 1
    return False, total, env.cfg.max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--boss-hp", type=float, default=50.0)
    parser.add_argument("--boss-fire-interval", type=int, default=18)
    parser.add_argument("--boss-burst", type=int, default=12)
    parser.add_argument("--enemy-bullet-speed", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    env = SnakePitEnv(
        SnakePitConfig(
            boss_hp_max=args.boss_hp,
            boss_fire_interval=args.boss_fire_interval,
            boss_burst=args.boss_burst,
            enemy_bullet_speed=args.enemy_bullet_speed,
        )
    )
    clears, returns, lengths = [], [], []
    for i in range(args.episodes):
        cleared, total, length = run_episode(agent, env, device, seed=args.seed + i)
        clears.append(float(cleared))
        returns.append(total)
        lengths.append(length)

    print(
        f"episodes {args.episodes} boss_hp {args.boss_hp} | "
        f"clear_rate {np.mean(clears):.3f} | mean_return {np.mean(returns):.2f} | "
        f"mean_length {np.mean(lengths):.1f}"
    )


if __name__ == "__main__":
    main()
