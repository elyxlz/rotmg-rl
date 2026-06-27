"""Record a faithful Snake Pit run to mp4 with the debug renderer (to follow along visually).

Uses a scripted oracle for the DEMO only (it cheats with a BFS path to the boss + dodges + shoots
nearest enemy + nukes the boss). The real RL policy gets none of this -- it has local vision only.
This is just to SEE the faithful env: snakes, grenades, the 3-phase boss.

    uv run python scripts/record_dungeon.py --out videos/faithful_demo.mp4
"""

from __future__ import annotations

import argparse
import pathlib
from collections import deque

import imageio.v2 as imageio
import numpy as np

from rotmg_rl.sim.dungeon import MOVE_DIRS, N_AIM, DungeonConfig, DungeonEnv


def bfs(walkable, goal):
    h, w = walkable.shape
    dist = np.full((h, w), np.inf, np.float32)
    dq = deque([goal])
    dist[goal[1], goal[0]] = 0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while dq:
        x, y = dq.popleft()
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx] and not np.isfinite(dist[ny, nx]):
                dist[ny, nx] = dist[y, x] + 1
                dq.append((nx, ny))
    return dist


def aim_at(src, dst) -> int:
    ang = np.arctan2(dst[1] - src[1], dst[0] - src[0]) % (2 * np.pi)
    return int(round(ang / (2 * np.pi / N_AIM))) % N_AIM


def oracle(env, geo) -> list[int]:
    p = env.player_pos
    move = 0
    # dodge nearest enemy bullet / grenade if close
    threats = []
    if env.enemy_bullets.shape[0]:
        threats.append(env.enemy_bullets[:, :2])
    if env.grenades.shape[0]:
        threats.append(env.grenades[:, :2])
    if threats:
        t = np.concatenate(threats, 0)
        d = np.linalg.norm(t - p, axis=1)
        if d.min() < 3.5:
            away = p - t[int(np.argmin(d))]
            move = int(np.argmax(MOVE_DIRS @ away)) + 1
    if move == 0 and not env.fight_active:  # navigate toward boss via BFS
        px, py = int(p[0]), int(p[1])
        best, ba = geo[py, px], 0
        for a in range(8):
            nx, ny = px + int(round(MOVE_DIRS[a, 0])), py + int(round(MOVE_DIRS[a, 1]))
            if 0 <= ny < geo.shape[0] and 0 <= nx < geo.shape[1] and geo[ny, nx] < best:
                best, ba = geo[ny, nx], a + 1
        move = ba
    # aim at nearest enemy (snake or boss); always shoot; nuke boss when able
    target = env.boss_pos
    alive = env.snakes[env.snakes[:, 2] > 0]
    if alive.shape[0]:
        d = np.linalg.norm(alive[:, :2] - p, axis=1)
        if d.min() < 10:
            target = alive[int(np.argmin(d)), :2]
    if env.fight_active:
        target = env.boss_pos
    cast = 1 if (env.fight_active and env.player_mp >= env.cfg.spell_cost and env.spell_timer == 0) else 0
    return [move, aim_at(p, target), 1, cast]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="videos/faithful_demo.mp4")
    parser.add_argument("--boss-hp", type=float, default=2500.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=3000)
    args = parser.parse_args()

    env = DungeonEnv(DungeonConfig(boss_hp_max=args.boss_hp), render_mode="rgb_array")
    env.reset(seed=0)
    geo = bfs(env.map.walkable, env.boss_xy)
    frames, cleared = [], False
    for _ in range(args.max_frames):
        _, _, term, trunc, info = env.step(oracle(env, geo))
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
