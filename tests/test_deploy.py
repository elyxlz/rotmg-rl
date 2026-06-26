"""M5: the deploy bridge reconstructs bullets faithfully and produces valid observations."""

import numpy as np

from rotmg_rl.deploy.realm_state import (
    ActionIntent,
    EnemyShootEvent,
    RealmState,
    action_to_intent,
    realm_to_observation,
    reconstruct_bullets,
)
from rotmg_rl.sim.snakepit import DIRS, SnakePitEnv


def _realm(**over):
    base = dict(
        arena_size=40.0,
        player_pos=np.array([20.0, 34.0], np.float32),
        player_hp=80.0,
        player_hp_max=100.0,
        player_mp=0.0,
        player_mp_max=1.0,
        ability_ready=False,
        boss_pos=np.array([20.0, 8.0], np.float32),
        boss_hp=150.0,
        boss_hp_max=250.0,
        now=10.0,
    )
    base.update(over)
    return RealmState(**base)


def test_reconstruct_matches_linear_motion():
    # A burst fired at t=5 from (20,8); at now=10 (age=5) each bullet is origin + vel*age.
    speed, age = 0.7, 5.0
    ev = EnemyShootEvent(
        origin=np.array([20.0, 8.0], np.float32),
        base_angle=0.0,
        count=3,
        arc_gap=np.pi / 6,
        speed=speed,
        spawn_time=5.0,
        lifetime=100.0,
    )
    bullets = reconstruct_bullets([ev], now=10.0, arena_size=40.0)
    assert bullets.shape == (3, 4)
    for i in range(3):
        angle = 0.0 + (i - 1) * (np.pi / 6)
        vx, vy = np.cos(angle) * speed, np.sin(angle) * speed
        assert np.allclose(bullets[i, 2:4], [vx, vy], atol=1e-5)
        assert np.allclose(bullets[i, :2], [20.0 + vx * age, 8.0 + vy * age], atol=1e-4)


def test_reconstruct_culls_expired_and_out_of_arena():
    ev_old = EnemyShootEvent(np.array([20.0, 8.0], np.float32), 0.0, 1, 0.0, 0.7, spawn_time=0.0, lifetime=3.0)
    assert reconstruct_bullets([ev_old], now=10.0, arena_size=40.0).shape[0] == 0  # expired
    ev_gone = EnemyShootEvent(np.array([1.0, 1.0], np.float32), np.pi, 1, 0.0, 5.0, spawn_time=0.0, lifetime=100.0)
    assert reconstruct_bullets([ev_gone], now=5.0, arena_size=40.0).shape[0] == 0  # flew off-arena


def test_realm_observation_in_env_space():
    env = SnakePitEnv()
    ev = EnemyShootEvent(np.array([20.0, 8.0], np.float32), 1.2, 8, np.pi / 4, 0.7, spawn_time=8.0, lifetime=60.0)
    obs = realm_to_observation(_realm(enemy_shoots=[ev]))
    assert env.observation_space.contains(obs)


def test_action_to_intent_mapping():
    stay_noshoot = action_to_intent([0, 0])
    assert isinstance(stay_noshoot, ActionIntent)
    assert np.array_equal(stay_noshoot.move, np.zeros(2, np.float32))
    assert stay_noshoot.shoot is False

    moving_shooting = action_to_intent([3, 5])
    assert np.allclose(moving_shooting.move, DIRS[2])
    assert moving_shooting.shoot is True
    assert np.allclose(moving_shooting.aim, DIRS[4])
