"""Follow training visually: periodically render the LATEST policy checkpoint to an mp4 and log
it to wandb, so you watch the policy improve as it trains.

Watches the PuffeRL checkpoint dir for the newest model_*.pt, runs a stochastic rollout with the
debug renderer (HP/MP bars + local-view box), logs the video to wandb, and saves it locally.

    WANDB_MODE=online uv run --extra train python scripts/follow_along.py --watch checkpoints/run
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import time

import imageio.v2 as imageio
import numpy as np
import torch
import wandb

import pufferlib.emulation as emulation
import pufferlib.vector as pvector
from pufferlib.ocean import torch as ocean_torch
from rotmg_rl.puffer_policy import DungeonPolicy
from rotmg_rl.sim.dungeon import DungeonConfig, DungeonEnv


def flatten(obs) -> np.ndarray:
    return np.concatenate([obs["grid"].ravel(), obs["scalars"]]).astype(np.float32)


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
            cleared = info["cleared"]
            break
    return frames, cleared


def newest_ckpt(watch):
    files = glob.glob(os.path.join(watch, "**", "model_*.pt"), recursive=True) + glob.glob(os.path.join(watch, "*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", default="checkpoints/run", help="dir of PuffeRL model_*.pt checkpoints")
    p.add_argument("--interval", type=int, default=180, help="seconds between renders")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--boss-hp", type=float, default=2500.0)
    p.add_argument("--wandb", action="store_true", help="also log rollouts to wandb (needs `wandb login`)")
    p.add_argument("--run-id", default=None, help="resume this wandb run id so videos land in the training run")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # driver env (for policy construction + emulated obs dtype)
    venv = pvector.make(emulation.GymnasiumPufferEnv, env_kwargs={"env_creator": DungeonEnv}, num_envs=1, backend=pvector.Serial)
    policy = DungeonPolicy(venv.driver_env, hidden_size=args.hidden)
    policy = ocean_torch.Recurrent(venv.driver_env, policy, input_size=args.hidden, hidden_size=args.hidden).to(device)

    if args.wandb:
        if args.run_id:  # resume the training run so videos share its dashboard
            wandb.init(project="rotmg-dungeon", id=args.run_id, resume="allow")
        else:
            wandb.init(project="rotmg-dungeon", name="rollouts", job_type="eval")
    pathlib.Path("videos").mkdir(exist_ok=True)
    # show the fight (where the action is): spawn in the boss room
    cfg = DungeonConfig(boss_hp_max=args.boss_hp, spawn_in_room_prob=1.0)
    last = None
    while True:
        ckpt = newest_ckpt(args.watch)
        if ckpt and ckpt != last:
            try:
                policy.load_state_dict(torch.load(ckpt, map_location=device))
                policy.eval()
                frames, cleared = rollout(policy, cfg, device, seed=int(time.time()) % 10000, max_frames=args.max_frames)
                out = f"videos/follow_{pathlib.Path(ckpt).stem}.mp4"
                imageio.mimsave(out, frames, fps=30)
                if args.wandb:
                    wandb.log({"rollout": wandb.Video(out, fps=30, format="mp4"), "cleared": float(cleared), "frames": len(frames)})
                print(f"rendered {ckpt} -> {out} (cleared={cleared})", flush=True)
                last = ckpt
            except Exception as e:  # checkpoint mid-write or arch mismatch; retry next loop
                print(f"skip {ckpt}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
