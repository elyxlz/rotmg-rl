"""Whole-dungeon env: navigation works, and the Wizard can kill the boss (staff + spell)."""

import numpy as np

from rotmg_rl.sim.dungeon import DIRS, DungeonConfig, DungeonEnv

_TILE_OFFSETS = [(int(round(d[0])), int(round(d[1]))) for d in DIRS]


def _nav_move(env: DungeonEnv) -> int:
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
    assert np.isfinite(env.prev_geodist)


def test_navigation_reaches_boss_room():
    env = DungeonEnv()
    env.reset(seed=0)
    for _ in range(env.cfg.max_steps):
        env.step([_nav_move(env), 0, 0])
        if env.fight_active:
            break
    assert env.fight_active


def test_wizard_staff_kills_boss():
    cfg = DungeonConfig(boss_hp_max=300.0, invuln_ticks=1)
    env = DungeonEnv(cfg)
    env.reset(seed=0)
    env.player_pos = (env.boss_pos + np.array([5.0, 0.0], np.float32)).astype(np.float32)
    cleared = False
    for _ in range(env.cfg.max_steps):
        _, _, term, trunc, info = env.step([0, _aim_at(env.player_pos, env.boss_pos), 0])
        if info["cleared"]:
            cleared = True
            break
        if term or trunc:
            break
    assert cleared


def test_wizard_spell_consumes_mp_and_damages_boss():
    cfg = DungeonConfig(boss_hp_max=7500.0, invuln_ticks=1)
    env = DungeonEnv(cfg)
    env.reset(seed=0)
    env.player_pos = (env.boss_pos + np.array([4.0, 0.0], np.float32)).astype(np.float32)
    env.fight_active, env.phase = True, 1  # in the fight
    mp_before, hp_before = env.player_mp, env.boss_hp
    env.step([0, _aim_at(env.player_pos, env.boss_pos), 1])  # cast spell
    for _ in range(8):  # let the burst reach the boss
        env.step([0, 0, 0])
    assert env.player_mp < mp_before  # spell consumed MP
    assert env.boss_hp < hp_before  # spell damaged the boss
