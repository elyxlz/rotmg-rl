"""Record one Snake Pit episode to mp4 (headless). Proves the video pipeline for M1.

Policy is a simple scripted shooter (aim at boss, dodge nothing) so the video shows real
mechanics before any trained policy exists.

    uv run python scripts/record_episode.py --out videos/scripted.mp4
"""

import argparse
import pathlib

import imageio.v2 as imageio
import numpy as np

from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def aim_at_boss(env: SnakePitEnv) -> int:
    to_boss = env.boss_pos - env.player_pos
    ang = np.arctan2(to_boss[1], to_boss[0]) % (2 * np.pi)
    return int(np.round(ang / (np.pi / 4.0))) % 8 + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="videos/scripted.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = SnakePitEnv(SnakePitConfig(boss_hp_max=120.0), render_mode="rgb_array")
    env.reset(seed=args.seed)
    frames = []
    cleared = False
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step([0, aim_at_boss(env)])
        frames.append(env.render())
        if terminated or truncated:
            cleared = info["cleared"]
            break

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"wrote {len(frames)} frames to {out} (cleared={cleared})")


if __name__ == "__main__":
    main()
