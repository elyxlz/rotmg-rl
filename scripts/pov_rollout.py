"""Record POV rollouts of a trained policy on the FULL dungeon (entrance spawn + threats) to mp4.

Unlike record_dungeon.py (a cheating BFS oracle), this drives the REAL policy with only its local
vision + fog-of-war minimap, so the video is honest evidence of what the checkpoint actually does.
Saves up to --keep cleared rollouts (falls back to the longest run if none clear) + prints a summary.

    uv run --extra train python scripts/pov_rollout.py --checkpoint checkpoints/full_dungeon_best.pt \
        --c-env --episodes 10 --n-snakes 40 --boss-hp 7500
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
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
def rollout(policy, cfg, device, seed, max_frames):
    env = DungeonEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    state = {"lstm_h": None, "lstm_c": None, "hidden": None}
    frames, cleared = [], False
    for _ in range(max_frames):
        x = torch.tensor(flatten(obs), device=device).unsqueeze(0)
        logits, _ = policy.forward_eval(x, state)
        action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
        obs, _, term, trunc, info = env.step(action)
        frames.append(env.render())
        if term or trunc:
            cleared = bool(info["cleared"])
            break
    return frames, cleared


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--keep", type=int, default=3, help="max cleared mp4s to save")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--c-env", action="store_true")
    p.add_argument("--boss-hp", type=float, default=7500.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=4000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--outdir", default="videos")
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
        boss_shoots=True,
        spawn_in_room_prob=args.spawn_in_room_prob,
    )
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved, clears = 0, 0
    for i in range(args.episodes):
        frames, cleared = rollout(policy, cfg, device, seed=20_000 + i, max_frames=args.max_frames)
        clears += int(cleared)
        if cleared and saved < args.keep:
            out = outdir / f"pov_clear_{i}.mp4"
            imageio.mimsave(out, frames, fps=args.fps)
            saved += 1
            print(f"ep {i}: CLEARED in {len(frames)} steps -> {out}", flush=True)
        else:
            print(f"ep {i}: cleared={cleared} ({len(frames)} steps)", flush=True)
    print(f"=== POV {args.checkpoint}: {clears}/{args.episodes} cleared, {saved} mp4(s) saved ===")


if __name__ == "__main__":
    main()
