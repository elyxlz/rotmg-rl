"""Whole-dungeon env: map connectivity/navigation works and the boss fight is winnable."""

import numpy as np

from rotmg_rl.sim.dungeon import DIRS, DungeonConfig, DungeonEnv

_TILE_OFFSETS = [(int(round(d[0])), int(round(d[1]))) for d in DIRS]


def _nav_move(env: DungeonEnv) -> int:
    """Descend the geodesic gradient toward the boss (neighbour tile with lowest geodist)."""
    px, py = int(env.player_pos[0]), int(env.player_pos[1])
    h, w = env.geodist.shape
    best_a, best_gd = 0, env._geodist_at(env.player_pos)
    for a in range(1, 9):
        ox, oy = _TILE_OFFSETS[a - 1]
        nx, ny = px + ox, py + oy
        if 0 <= nx < w and 0 <= ny < h and np.isfinite(env.geodist[ny, nx]) and env.geodist[ny, nx] < best_gd:
            best_gd, best_a = float(env.geodist[ny, nx]), a
    return best_a


def _aim_at(src, dst) -> int:
    ang = np.arctan2(dst[1] - src[1], dst[0] - src[0]) % (2 * np.pi)
    return int(round(ang / (np.pi / 4.0))) % 8 + 1


def test_obs_shapes_and_space():
    env = DungeonEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert np.isfinite(env.prev_geodist)  # entrance connected to boss


def test_navigation_reaches_boss_room():
    env = DungeonEnv()
    env.reset(seed=0)
    for _ in range(env.cfg.max_steps):
        env.step([_nav_move(env), 0])  # navigate, no shooting
        if env.fight_active:
            break
    assert env.fight_active  # reached the boss room over the real map


def test_boss_dies_when_shot():
    cfg = DungeonConfig(boss_hp_max=150.0, player_hp_max=5000.0, invuln_ticks=1)
    env = DungeonEnv(cfg)
    env.reset(seed=0)
    env.player_pos = (env.boss_pos + np.array([5.0, 0.0], np.float32)).astype(np.float32)  # skip navigation
    cleared = False
    for _ in range(env.cfg.max_steps):
        _, _, term, trunc, info = env.step([0, _aim_at(env.player_pos, env.boss_pos)])
        if info["cleared"]:
            cleared = True
            break
        if term or trunc:
            break
    assert cleared
