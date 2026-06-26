"""Record the TRAINED policy clearing the sim boss to an mp4 (for visualization).

Tries seeds until it captures a clear (the policy is stochastic), so the video shows a win.

    uv run --extra train python scripts/record_policy.py --checkpoint checkpoints/m3-final.pt \
        --out videos/m3-final-clear.mp4 --boss-hp 250
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
import numpy as np
import torch

from rotmg_rl.policy import Agent
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def flatten(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([obs["grid"].ravel(), obs["scalars"]]).astype(np.float32)


@torch.no_grad()
def rollout(agent: Agent, cfg: SnakePitConfig, device, seed: int):
    env = SnakePitEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    lstm = agent.initial_state(1, device)
    done = torch.zeros(1, device=device)
    frames, cleared = [], False
    for _ in range(cfg.max_steps):
        x = torch.tensor(flatten(obs), device=device).unsqueeze(0)
        action, lstm = agent.act_sample(x, lstm, done)
        obs, _, term, trunc, info = env.step(action[0].cpu().numpy())
        frames.append(env.render())
        if term or trunc:
            cleared = info["cleared"]
            break
    return frames, cleared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="videos/policy-clear.mp4")
    parser.add_argument("--boss-hp", type=float, default=250.0)
    parser.add_argument("--boss-fire-interval", type=int, default=18)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--max-tries", type=int, default=15)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    cfg = SnakePitConfig(boss_hp_max=args.boss_hp, boss_fire_interval=args.boss_fire_interval, randomize=args.randomize)

    best, best_cleared = None, False
    for seed in range(args.max_tries):
        frames, cleared = rollout(agent, cfg, device, seed=seed)
        if cleared:
            best, best_cleared = frames, True
            break
        if best is None or len(frames) > len(best):
            best = frames  # fall back to the longest survival
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, best, fps=args.fps)
    print(f"wrote {out} ({len(best)} frames, cleared={best_cleared})")


if __name__ == "__main__":
    main()
