"""M1 verification: the Snake Pit reference sim steps correctly and obeys its contract."""

import numpy as np

from rotmg_rl.observation import GRID_SHAPE, NUM_SCALARS
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def test_reset_obs_shapes():
    env = SnakePitEnv()
    obs, info = env.reset(seed=0)
    assert obs["grid"].shape == GRID_SHAPE
    assert obs["scalars"].shape == (NUM_SCALARS,)
    assert obs["grid"].dtype == np.float32
    assert np.isfinite(obs["grid"]).all() and np.isfinite(obs["scalars"]).all()


def test_step_runs_and_observation_in_space():
    env = SnakePitEnv()
    obs, _ = env.reset(seed=1)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_determinism_same_seed():
    a = SnakePitEnv()
    b = SnakePitEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    actions = [a.action_space.sample() for _ in range(100)]
    for act in actions:
        ra = a.step(act)
        rb = b.step(act)
        assert ra[1] == rb[1]
        assert np.array_equal(ra[0]["grid"], rb[0]["grid"])


def test_boss_can_die_and_clear_is_rewarded():
    # Shrink boss HP so a focused-fire policy clears quickly; verifies the clear path + reward.
    cfg = SnakePitConfig(boss_hp_max=5.0, boss_fire_interval=10_000)  # boss barely shoots
    env = SnakePitEnv(cfg)
    env.reset(seed=3)
    cleared = False
    total = 0.0
    for _ in range(cfg.max_steps):
        # Always shoot toward the boss: aim index = direction bucket from player to boss.
        to_boss = env.boss_pos - env.player_pos
        ang = np.arctan2(to_boss[1], to_boss[0]) % (2 * np.pi)
        aim = int(np.round(ang / (np.pi / 4.0))) % 8 + 1
        obs, reward, terminated, truncated, info = env.step([0, aim])
        total += reward
        if info["cleared"]:
            cleared = True
            break
    assert cleared
    assert total > 0.0
