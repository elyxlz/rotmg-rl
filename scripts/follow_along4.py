"""Follow a PufferLib 4.0 (--slowly / torch backend) training run visually: render the latest
checkpoint to a POV mp4 and log it to wandb, like follow_along.py does for the 3.0 runs.

4.0's torch backend saves checkpoints as `checkpoints4/dungeon/<run_id>/<global_step>.bin` — a plain
torch state_dict of pufferlib.models.Policy(DungeonEncoder, DefaultDecoder, LSTM) (our CNN). We
reconstruct that exact policy, load the weights, and drive the numpy DungeonEnv (its obs flattens to
the same [grid, scalars] layout the C env feeds the encoder), rendering the debug POV.

ONLY works for --slowly checkpoints (our CNN, torch state_dict). The native _C backend saves opaque
flat-weight dumps with puffernet's flat encoder — not renderable here (see docs/pufferlib4-migration.md).

    .venv4/bin/python scripts/follow_along4.py --watch checkpoints4/dungeon --wandb --run-id <id>
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

import pufferlib.models as models
from rotmg_rl.sim.dungeon import GRID, MM, NUM_CH, NUM_MM_CH, NUM_SCALARS, DungeonConfig, DungeonEnv

OBS_SIZE = NUM_CH * GRID * GRID + NUM_MM_CH * MM * MM + NUM_SCALARS
ACT_SIZES = [9, 32, 2, 2]


def flatten(obs) -> np.ndarray:  # C obs layout: [grid, minimap, scalars]
    return np.concatenate([obs["grid"].ravel(), obs["minimap"].ravel(), obs["scalars"]]).astype(np.float32)


def build_policy(hidden: int, num_layers: int, device) -> torch.nn.Module:
    encoder = models.DungeonEncoder(OBS_SIZE, hidden)
    decoder = models.DefaultDecoder(ACT_SIZES, hidden)
    network = models.LSTM(hidden, num_layers=num_layers)
    return models.Policy(encoder, decoder, network).to(device)


@torch.no_grad()
def rollout(policy, cfg, device, seed, max_frames):
    env = DungeonEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
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


def newest_ckpt(watch):
    files = glob.glob(os.path.join(watch, "**", "*.bin"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def step_of(ckpt: str) -> int:
    return int(pathlib.Path(ckpt).stem)  # 4.0 names checkpoints <global_step>.bin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", default="checkpoints4/dungeon", help="dir of 4.0 <run_id>/<step>.bin checkpoints")
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

        name = f"rollouts4-{args.run_id}" if args.run_id else "rollouts4"
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
                out = f"videos/follow4_{step_of(ckpt)}.mp4"
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
