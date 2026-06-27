"""Spec-derived scenario tests for the C Snake Pit env (pufferlib/ocean/dungeon/dungeon.h, the single dynamics source).

These are NOT output-pinning golden tests (which would be circular -- they'd just re-assert whatever
the code emits). Each expected value is HAND-COMPUTED from the betterSkillys formulas the env mirrors:

  - DamageWithDefense clamp: dealt = max(raw * damage_floor, raw - defense)
  - HealthRegen (Player.cs HandleRegen): hp/s = 1 + 0.36*VIT (=1.54/tick at VIT 40), added while hp<max
  - the BulletNova spell: spell_num bullets, point-blank, each clamped by the boss defense
  - clear = boss dead (boss_hp<=0 & phase>0) terminates; reaching max_steps without that truncates

A `test_golden_trajectory` drift tripwire sits on top: a fixed seed + fixed action schedule whose
aggregate signals are committed, so any unintended change to the dynamics trips it.

Runs with numpy + the compiled C binding only (no pufferlib/torch), via the single-env wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rotmg_rl.csim.binding")
from rotmg_rl.config import DungeonConfig  # noqa: E402
from rotmg_rl.csim.single import OBS_SIZE, CDungeonSingle  # noqa: E402

BOSS_TILE = (16, 73)  # _nearest_walkable(Stheno) on the real map (matches the C env's BOSS_X/Y)


def _defended(raw: float, defense: float, floor: float = 0.1) -> float:
    return max(raw * floor, raw - defense)


def test_obs_shape_and_scalar_range():
    env = CDungeonSingle(DungeonConfig(n_snakes=10), seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    scalars = obs[-8:]  # the 8 trailing scalars: hp, mp, spell_ready, boss_visible, ...
    assert scalars.min() >= -1.0 and scalars.max() <= 1.0
    # full obs is normalized to [-1, 1] by construction
    assert np.isfinite(obs).all() and obs.min() >= -1.0001 and obs.max() <= 1.0001
    env.close()


def test_damage_clamp_player_blade():
    """A 100-dmg boss blade vs the T7 Wizard's DEF 8 deals max(100*0.1, 100-8) = max(10, 92) = 92
    (a SINGLE blade would take HP 810 -> 718). P1 fires a 3-blade volley point-blank; with regen off
    and the blade cooldown raised so only one volley lands, the three clamped hits take 810 -> 534."""
    per_blade = _defended(100.0, 8.0)
    assert per_blade == 92.0
    cfg = DungeonConfig(
        boss_hp_max=1e9,
        player_hp_max=810.0,
        hp_regen=0.0,
        n_snakes=0,
        boss_wander_speed=0.0,
        enable_grenades=False,
        ebullet_dmg=100.0,
        blade_cd=1000,
        invuln_ticks=0,
        opening_invuln_ticks=0,
    )
    env = CDungeonSingle(cfg, seed=2)
    env.reset(seed=2)
    env.put(player_x=BOSS_TILE[0] + 1.5, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    for _ in range(25):  # let the single point-blank volley land (no regen, no further volleys)
        env.step([0, 0, 0, 0])
    assert env.get()["player_hp"] == pytest.approx(810.0 - 3 * per_blade)  # 534.0
    env.close()


def test_hp_regen_after_idle_ticks():
    """HealthRegen at VIT 40: (1 + 0.36*40)/s = 15.4/s = 1.54/tick, added flat while hp < max. After
    the player is hurt (one 3-blade volley -> 534) and then idles T ticks far from any threat, HP rises
    by exactly hp_regen * T."""
    cfg = DungeonConfig(
        boss_hp_max=1e9,
        player_hp_max=810.0,
        hp_regen=1.54,
        n_snakes=0,
        boss_wander_speed=0.0,
        enable_grenades=False,
        ebullet_dmg=100.0,
        blade_cd=1000,
        invuln_ticks=0,
        opening_invuln_ticks=0,
    )
    env = CDungeonSingle(cfg, seed=2)
    env.reset(seed=2)
    env.put(player_x=BOSS_TILE[0] + 1.5, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    env.step([0, 0, 0, 0])  # take the volley
    env.put(player_x=110.5, player_y=21.5, fight_active=0, phase=0)  # teleport to the entrance, end the fight
    hp_before = env.get()["player_hp"]
    T = 10
    for _ in range(T):
        env.step([0, 0, 0, 0])
    hp_after = env.get()["player_hp"]
    assert hp_after - hp_before == pytest.approx(1.54 * T, abs=1e-2)  # 15.4 (hp_regen stored float32)
    env.close()


def test_spell_nova_total_vs_boss():
    """The 360-degree BulletNova spell: 20 bullets, point-blank on the boss, each clamped by the boss
    DEF 19. With a fixed [D,D] damage the whole nova deals 20 * max(D*0.1, D-19); at D=185 that is
    20 * 166 = 3320, taking the boss 7500 -> 4180."""
    D = 185.0
    per_bullet = _defended(D, 19.0)
    assert per_bullet == 166.0
    cfg = DungeonConfig(
        boss_hp_max=7500.0,
        player_hp_max=1e9,
        n_snakes=0,
        boss_wander_speed=0.0,
        boss_shoots=False,
        enable_grenades=False,
        spell_dmg_lo=D,
        spell_dmg_hi=D,
        invuln_ticks=0,
        opening_invuln_ticks=0,
    )
    env = CDungeonSingle(cfg, seed=1)
    env.reset(seed=1)
    env.put(player_x=BOSS_TILE[0] + 0.5, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    env.step([0, 0, 0, 1])  # cast the nova point-blank
    env.step([0, 0, 0, 0])  # let the bullets advance into the boss and resolve
    assert env.get()["boss_hp"] == pytest.approx(7500.0 - 20 * per_bullet)  # 4180.0
    env.close()


def test_full_clear_terminates():
    """Killing the boss (boss_hp <= 0 in a fight) terminates the episode with cleared=True."""
    cfg = DungeonConfig(
        boss_hp_max=50.0,
        player_hp_max=1e9,
        n_snakes=0,
        boss_wander_speed=0.0,
        boss_shoots=False,
        enable_grenades=False,
        invuln_ticks=0,
        opening_invuln_ticks=0,
    )
    env = CDungeonSingle(cfg, seed=1)
    env.reset(seed=1)
    env.put(player_x=BOSS_TILE[0] + 1.0, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    for _ in range(200):
        _, _, terminated, truncated, info = env.step([0, 16, 1, 1])  # shoot + cast at the boss
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["cleared"] and info["boss_hp_frac"] == 0.0  # boss dead -> hp fraction 0
    env.close()


def test_timeout_truncates():
    """Reaching max_steps without a clear or a death truncates (terminated=False, truncated=True),
    and the episode is not flagged cleared."""
    max_steps = 15
    cfg = DungeonConfig(
        boss_hp_max=300.0,
        player_hp_max=1e9,
        n_snakes=0,
        boss_wander_speed=0.0,
        boss_shoots=False,
        enable_grenades=False,
        max_steps=max_steps,
        spawn_in_room_prob=1.0,
    )
    env = CDungeonSingle(cfg, seed=1)
    env.reset(seed=1)
    ended = None
    for _ in range(max_steps + 5):
        _, _, terminated, truncated, info = env.step([0, 0, 0, 0])  # noop -> boss never damaged
        if terminated or truncated:
            ended = (terminated, truncated, info)
            break
    assert ended is not None
    terminated, truncated, info = ended
    assert truncated and not terminated
    assert not info["cleared"]
    env.close()


# --- drift tripwire ---------------------------------------------------------------------------------
# A fixed seed + deterministic action schedule, 200 steps, with the boss HP high enough that it never
# clears and the player HP high enough that it never dies (so the episode runs the full window). The
# committed aggregates were captured from the C env; an unintended dynamics change moves them. These
# are a TRIPWIRE, not a spec: regenerate them deliberately if the dynamics intentionally change.
GOLDEN_SEED = 2024
GOLDEN_STEPS = 200
GOLDEN = {"total_reward": 0.5704, "obs_checksum": 51500.75, "player_hp": 999991.0, "boss_hp": 6445.59}


def _golden_actions(n: int) -> np.ndarray:
    a = np.zeros((n, 4), np.int32)
    for t in range(n):
        a[t, 0] = (t % 8) + 1  # cycle the 8 move directions
        a[t, 1] = (t * 5) % 32  # sweep the aim
        a[t, 2] = t % 2  # shoot every other tick
        a[t, 3] = 1 if t % 20 == 0 else 0  # cast occasionally
    return a


def test_golden_trajectory_drift_tripwire():
    cfg = DungeonConfig(boss_hp_max=7500.0, player_hp_max=1e6, n_snakes=30, spawn_in_room_prob=1.0, spawn_in_room_radius=8.0, max_steps=10_000)
    env = CDungeonSingle(cfg, seed=GOLDEN_SEED)
    obs = env.reset(seed=GOLDEN_SEED)
    actions = _golden_actions(GOLDEN_STEPS)
    total, checksum = 0.0, 0.0
    for t in range(GOLDEN_STEPS):
        obs, reward, terminated, truncated, _ = env.step(actions[t])
        total += reward
        checksum += float(obs.sum())
        assert not (terminated or truncated), f"golden episode ended early at step {t}"
    state = env.get()
    env.close()
    # tolerances absorb cross-arch transcendental rounding while still catching real dynamics drift
    assert total == pytest.approx(GOLDEN["total_reward"], abs=0.05)
    assert checksum == pytest.approx(GOLDEN["obs_checksum"], abs=100.0)
    assert state["player_hp"] == pytest.approx(GOLDEN["player_hp"], abs=50.0)
    assert state["boss_hp"] == pytest.approx(GOLDEN["boss_hp"], abs=300.0)
