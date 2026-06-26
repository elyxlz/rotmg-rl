"""Whole-dungeon navigation env: the map is connected entrance->boss and mechanics work."""

import numpy as np

from rotmg_rl.sim.dungeon import DIRS, DungeonEnv


# 8 neighbour-tile offsets in DIRS order (descend the geodesic gradient by tile, not sub-tile).
_TILE_OFFSETS = [(int(round(d[0])), int(round(d[1]))) for d in DIRS]


def _oracle_move(env: DungeonEnv) -> int:
    """Move toward the neighbouring tile with the lowest geodesic distance to the boss."""
    px, py = int(env.player_pos[0]), int(env.player_pos[1])
    h, w = env.geodist.shape
    best_a, best_gd = 0, env._geodist_at(env.player_pos)
    for a in range(1, 9):
        ox, oy = _TILE_OFFSETS[a - 1]
        nx, ny = px + ox, py + oy
        if 0 <= nx < w and 0 <= ny < h and np.isfinite(env.geodist[ny, nx]) and env.geodist[ny, nx] < best_gd:
            best_gd, best_a = float(env.geodist[ny, nx]), a
    return best_a


def test_obs_shapes_and_space():
    env = DungeonEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert np.isfinite(env.prev_geodist)  # entrance is connected to the boss


def test_navigation_oracle_reaches_boss():
    env = DungeonEnv()
    env.reset(seed=0)
    total, reached = 0.0, False
    for _ in range(env.cfg.max_steps):
        obs, r, term, trunc, info = env.step(_oracle_move(env))
        total += r
        if term:
            reached = info["reached"]
            break
        if trunc:
            break
    assert reached
    assert total > 0
