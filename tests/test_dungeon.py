"""Whole-dungeon navigation env: the map is connected entrance->boss and mechanics work."""

import numpy as np

from rotmg_rl.sim.dungeon import DIRS, DungeonEnv


def _oracle_move(env: DungeonEnv) -> int:
    """Pick the move that most decreases geodesic distance to the boss."""
    best_a, best_gd = 0, env._geodist_at(env.player_pos)
    for a in range(1, 9):
        cand = env.player_pos.copy()
        sv = DIRS[a - 1] * env.cfg.player_speed
        for axis in (0, 1):
            c2 = cand.copy()
            c2[axis] += sv[axis]
            if env._walkable_at(c2):
                cand = c2
        gd = env._geodist_at(cand)
        if gd < best_gd:
            best_gd, best_a = gd, a
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
