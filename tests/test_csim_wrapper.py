"""Logging-semantics tests for the vectorized PufferEnv wrapper (CDungeon).

The C env reports per-STEP info (boss_hp_frac, in_room, cleared) like the gym vector backend, and
per-EPISODE aggregates (score, clear_rate) summed at the episode boundary. These guard the exact bug
that conflated the two (reporting boss_hp_frac=0 / cleared=1 every step). Requires pufferlib (the
training stack), so it is skipped on the cheap dev box; the C dynamics themselves are covered
pufferlib-free in test_dungeon.py / test_dungeon_scenarios.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pufferlib")
from rotmg_rl.config import DungeonConfig  # noqa: E402
from rotmg_rl.csim.dungeon import OBS_SIZE, CDungeon  # noqa: E402


def test_metrics_per_step_boss_full_when_undamaged():
    """Passive boss spawned in-room, NO shooting: the boss is never damaged, so the per-step metrics
    report boss_hp_frac ~= 1.0 and cleared = 0 (the bug aggregated per-EPISODE -> boss_hp_frac=0)."""
    cfg = DungeonConfig(
        boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0
    )
    env = CDungeon(cfg, num_envs=64, seed=1, log_interval=100)
    env.reset(seed=1)
    noop = np.zeros((64, 4), np.int32)
    log = None
    for _ in range(100):
        *_, info = env.step(noop)
        if info:
            log = info[0]
    env.close()
    assert log is not None
    assert log["boss_hp_frac"] > 0.99, log  # boss full every step (bug: 0.0)
    assert log["cleared"] == 0.0, log  # never cleared (bug: 1.0)
    assert log["in_room"] > 0.9, log  # spawned in room -> fight active most steps


def test_metrics_cleared_is_a_per_step_rate():
    """With shooting, episodes DO clear fast (per-episode rate ~1), but the per-step `cleared` metric
    is a low rate and boss_hp_frac stays well above 0 -- per-step (numpy) semantics, not per-episode."""
    cfg = DungeonConfig(
        boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0
    )
    env = CDungeon(cfg, num_envs=128, seed=0, log_interval=200)
    env.reset(seed=0)
    rng = np.random.RandomState(0)
    log = None
    for _ in range(200):
        a = rng.randint(0, [9, 32, 2, 2], size=(128, 4)).astype(np.int32)
        *_, info = env.step(a)
        if info:
            log = info[0]
    env.close()
    assert log is not None
    assert 0.0 < log["cleared"] < 0.3, log  # per-step clear rate is small (not the per-episode ~1.0)
    assert log["boss_hp_frac"] > 0.3, log  # boss healthy most steps (not 0)


def test_score_metric_per_episode():
    """`score` is the per-EPISODE mean end-of-episode score (1 if cleared else damage fraction). With a
    passive boss, huge HP, noop actions and a tiny max_steps, every episode truncates at full boss HP
    -> per-episode score == 0, and episodes > 0 (truncations counted)."""
    cfg = DungeonConfig(
        boss_hp_max=300.0,
        player_hp_max=1e9,
        n_snakes=0,
        enable_grenades=False,
        enable_minions=False,
        boss_shoots=False,
        spawn_in_room_prob=1.0,
        max_steps=20,
    )
    env = CDungeon(cfg, num_envs=64, seed=1, log_interval=200)
    env.reset(seed=1)
    noop = np.zeros((64, 4), np.int32)
    log = None
    for _ in range(200):
        *_, info = env.step(noop)
        if info:
            log = info[0]
    env.close()
    assert log is not None
    assert log["episodes"] > 0.0, log  # truncations were counted
    assert log["score"] < 0.01, log  # boss never damaged -> end-of-episode score ~0 (bug: per-step dilute)


def test_score_metric_clears_reach_one():
    """When the boss dies fast (tiny HP, shooting toward it), episodes clear and the per-episode
    `score` approaches 1.0 -- distinct from the per-step `cleared` rate."""
    cfg = DungeonConfig(
        boss_hp_max=20.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0
    )
    env = CDungeon(cfg, num_envs=128, seed=0, log_interval=300)
    env.reset(seed=0)
    a = np.zeros((128, 4), np.int32)
    a[:, 1] = 16  # aim
    a[:, 2] = 1  # always shoot -> the boss in the room dies quickly
    log = None
    for _ in range(300):
        *_, info = env.step(a)
        if info:
            log = info[0]
    env.close()
    assert log is not None
    assert log["episodes"] > 0.0, log
    assert log["score"] > 0.8, log  # most episodes clear -> per-episode score near 1


def test_full_config_runs_via_wrapper():
    """The stochastic training config (snakes + minions + grenades) runs in the PufferEnv wrapper
    without crashing and yields finite, in-range observations + rewards."""
    n = 8
    env = CDungeon(DungeonConfig(), num_envs=n, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (n, OBS_SIZE)
    rng = np.random.RandomState(0)
    for _ in range(300):
        actions = np.stack([rng.randint(0, [9, 32, 2, 2]) for _ in range(n)]).astype(np.int32)
        obs, rew, term, trunc, _ = env.step(actions)
        assert np.isfinite(obs).all()
        assert obs.min() >= -1.0001 and obs.max() <= 1.0001
        assert np.isfinite(rew).all()
    env.close()
