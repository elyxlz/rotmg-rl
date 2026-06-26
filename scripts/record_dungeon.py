"""Record a Snake Pit run to mp4 with the debug renderer (to follow along visually).

Uses a scripted oracle (geodesic navigation -> reach boss -> staff + spell) so we can SEE the
full dungeon + the Wizard now, before a trained policy exists. Swap in a checkpoint later.

    uv run python scripts/record_dungeon.py --out videos/dungeon_demo.mp4
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
import numpy as np

from rotmg_rl.sim.dungeon import DIRS, DungeonConfig, DungeonEnv

_TILE_OFFSETS = [(int(round(d[0])), int(round(d[1]))) for d in DIRS]


def nav_move(env) -> int:
    px, py = int(env.player_pos[0]), int(env.player_pos[1])
    h, w = env.geodist.shape
    best_a, best_gd = 0, env._geodist_at(env.player_pos)
    for a in range(1, 9):
        ox, oy = _TILE_OFFSETS[a - 1]
        nx, ny = px + ox, py + oy
        if 0 <= nx < w and 0 <= ny < h and np.isfinite(env.geodist[ny, nx]) and env.geodist[ny, nx] < best_gd:
            best_gd, best_a = float(env.geodist[ny, nx]), a
    return best_a


def aim_at(src, dst) -> int:
    ang = np.arctan2(dst[1] - src[1], dst[0] - src[0]) % (2 * np.pi)
    return int(round(ang / (np.pi / 4.0))) % 8 + 1


def oracle(env) -> list[int]:
    if not env.fight_active:
        return [nav_move(env), 0, 0]
    # in the fight: dodge nearest bullet, shoot + nuke the boss
    move = 0
    if env.enemy_bullets.shape[0]:
        d = np.linalg.norm(env.enemy_bullets[:, :2] - env.player_pos, axis=1)
        if d.min() < 4.0:
            away = env.player_pos - env.enemy_bullets[int(np.argmin(d)), :2]
            move = aim_at(env.player_pos, env.player_pos + away)
    aim = aim_at(env.player_pos, env.boss_pos)
    cast = 1 if (env.player_mp >= env.cfg.spell_cost and env.spell_timer == 0) else 0
    return [move, aim, cast]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="videos/dungeon_demo.mp4")
    parser.add_argument("--boss-hp", type=float, default=2500.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=2500)
    args = parser.parse_args()

    env = DungeonEnv(DungeonConfig(boss_hp_max=args.boss_hp), render_mode="rgb_array")
    env.reset(seed=0)
    frames, cleared = [], False
    for _ in range(args.max_frames):
        _, _, term, trunc, info = env.step(oracle(env))
        frames.append(env.render())
        if term or trunc:
            cleared = info["cleared"]
            break
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"wrote {out} ({len(frames)} frames, cleared={cleared})")


if __name__ == "__main__":
    main()
