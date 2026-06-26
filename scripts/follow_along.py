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
    p.add_argument("--run-id", default=None, help="training run id; videos go to a run named rollouts-<id>")
    p.add_argument("--fps", type=int, default=15, help="rollout video fps (lower = slower/realtime)")
    p.add_argument("--c-env", action="store_true", help="checkpoints are from the C env -> use CDungeonPolicy")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # driver env (for policy construction + emulated obs dtype)
    venv = pvector.make(emulation.GymnasiumPufferEnv, env_kwargs={"env_creator": DungeonEnv}, num_envs=1, backend=pvector.Serial)
    if args.c_env:  # C-env checkpoints use the flat-obs policy (same flat layout the numpy env produces)
        from rotmg_rl.csim.policy import CDungeonPolicy

        policy = CDungeonPolicy(venv.driver_env, hidden_size=args.hidden)
    else:
        policy = DungeonPolicy(venv.driver_env, hidden_size=args.hidden)
    policy = ocean_torch.Recurrent(venv.driver_env, policy, input_size=args.hidden, hidden_size=args.hidden).to(device)

    if args.wandb:
        # a dedicated run named for the training run (resuming the live run drops media on step conflicts)
        name = f"rollouts-{args.run_id}" if args.run_id else "rollouts"
        wandb.init(project="rotmg-dungeon", name=name, job_type="eval", group=os.environ.get("WANDB_RUN_GROUP"))
    pathlib.Path("videos").mkdir(exist_ok=True)
    # match the training env (train_dungeon writes env_config.json); start in the fight for a watchable video
    cfg_path = pathlib.Path(args.watch) / "env_config.json"
    if cfg_path.exists():
        import json

        d = json.loads(cfg_path.read_text())
        d["spawn_in_room_prob"], d["random_spawn_prob"] = 1.0, 0.0
        cfg = DungeonConfig(**d)
    else:
        cfg = DungeonConfig(boss_hp_max=args.boss_hp, spawn_in_room_prob=1.0)
    last, steps_per_epoch = None, None
    while True:
        ckpt = newest_ckpt(args.watch)
        if ckpt and ckpt != last:
            try:
                policy.load_state_dict(torch.load(ckpt, map_location=device))
                policy.eval()
                frames, cleared = rollout(policy, cfg, device, seed=int(time.time()) % 10000, max_frames=args.max_frames)
                out = f"videos/follow_{pathlib.Path(ckpt).stem}.mp4"
                imageio.mimsave(out, frames, fps=args.fps)  # encode fps controls playback speed
                if args.wandb:
                    # log at the ACTUAL training step of this checkpoint (epoch * steps/epoch from the live run)
                    epoch = int("".join(c for c in pathlib.Path(ckpt).stem if c.isdigit()) or "0")
                    if steps_per_epoch is None and args.run_id:
                        try:
                            s = wandb.Api().run(f"rotmg-dungeon/{args.run_id}").summary
                            steps_per_epoch = s["agent_steps"] / s["epoch"] if s["epoch"] else None
                        except Exception:
                            steps_per_epoch = None
                    step = int(epoch * steps_per_epoch) if steps_per_epoch else None
                    wandb.log({"rollout": wandb.Video(out, format="mp4"), "rollout_cleared": float(cleared), "checkpoint_epoch": epoch}, step=step)
                print(f"rendered {ckpt} -> {out} (cleared={cleared})", flush=True)
                last = ckpt
            except Exception as e:  # checkpoint mid-write or arch mismatch; retry next loop
                print(f"skip {ckpt}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
