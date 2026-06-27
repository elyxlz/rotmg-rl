"""Behavioral tests for the C Snake Pit env (pufferlib/ocean/dungeon/dungeon.h) via the single-env wrapper.

Qualitative "does the env behave" coverage (local vision, exploration, snakes take damage, the boss's
P1 point-blank-only blades). The exact spec numbers + the drift tripwire live in
tests/test_dungeon_scenarios.py. Numpy + the compiled C binding only (no pufferlib/torch).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rotmg_rl.csim.binding")
from rotmg_rl.config import N_AIM, DungeonConfig  # noqa: E402
from rotmg_rl.csim.single import OBS_SIZE, CDungeonSingle  # noqa: E402

BOSS_TILE = (16, 73)


def _aim_at(src, dst) -> int:
    ang = np.arctan2(dst[1] - src[1], dst[0] - src[0]) % (2 * np.pi)
    return int(round(ang / (2 * np.pi / N_AIM))) % N_AIM


def test_obs_shape_snakes_and_fog():
    env = CDungeonSingle(DungeonConfig(n_snakes=40), seed=0)
    obs = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    rs = env.get()
    assert rs["snakes"].shape[0] > 0  # snakes populate the dungeon
    # fog of war: at spawn only a disk is discovered (partial), and the boss is unseen.
    discovered = int(np.asarray(rs["discovered"]).sum())
    assert 0 < discovered < rs["discovered"].shape[0]
    assert rs["boss_seen"] == 0.0
    env.close()


def test_random_steps_stay_in_space():
    env = CDungeonSingle(DungeonConfig(), seed=1)
    env.reset(seed=1)
    rng = np.random.RandomState(1)
    for _ in range(60):
        obs, r, term, trunc, _ = env.step(rng.randint([9, N_AIM, 2, 2]).astype(np.int32))
        assert np.isfinite(obs).all() and obs.min() >= -1.0001 and obs.max() <= 1.0001
        assert np.isfinite(r)
        if term or trunc:
            break
    env.close()


def test_exploration_grows_and_is_rewarded():
    env = CDungeonSingle(DungeonConfig(spawn_in_room_prob=0.0), seed=2)
    env.reset(seed=2)
    before = int(np.asarray(env.get()["discovered"]).sum())
    total = 0.0
    for a in range(1, 9):  # walk a bit in each direction
        for _ in range(10):
            _, r, *_ = env.step([a, 0, 0, 0])
            total += r
    assert int(np.asarray(env.get()["discovered"]).sum()) > before  # new tiles discovered
    assert total > 0  # exploration reward outweighs the step cost
    env.close()


def test_snakes_take_damage():
    """Standing on a snake and unloading staff + nova kills nearby snakes (the alive count drops)."""
    env = CDungeonSingle(DungeonConfig(n_snakes=40, spell_dmg_lo=185.0, spell_dmg_hi=185.0), seed=3)
    env.reset(seed=3)
    snakes = env.get()["snakes"]
    n0 = snakes.shape[0]
    env.put(player_x=float(snakes[0, 0]), player_y=float(snakes[0, 1]))
    for _ in range(8):
        env.step([0, 16, 1, 1])  # shoot + cast point-blank
    assert env.get()["snakes"].shape[0] < n0
    env.close()


def test_p1_blades_are_point_blank_only():
    """Fidelity: in P1 the boss fires its 3-blade shot ONLY within the point-blank acquire radius (2).
    At range it fires no blades (only the Confused grenade); point-blank, the blades hit the player."""
    # at range (8 tiles), grenades off: no blades reach, so the player takes no damage.
    cfg = DungeonConfig(
        n_snakes=0, invuln_ticks=1, boss_wander_speed=0.0, opening_invuln_ticks=0, enable_grenades=False, player_hp_max=1e9, hp_regen=0.0
    )
    env = CDungeonSingle(cfg, seed=5)
    env.reset(seed=5)
    env.put(player_x=BOSS_TILE[0] + 8.0, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    saw_blade = False
    for _ in range(40):
        env.step([0, 0, 0, 0])
        saw_blade = saw_blade or env.get()["enemy_bullets"].shape[0] > 0
    assert not saw_blade  # no blades at range in P1
    assert env.get()["player_hp"] == 1e9  # and so no blade damage

    # point-blank (within radius 2): the boss acquires and the blades land (player HP drops).
    env.put(player_x=BOSS_TILE[0] + 1.5, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    for _ in range(40):
        env.step([0, 0, 0, 0])
    assert env.get()["player_hp"] < 1e9  # point-blank blades acquired and hit
    env.close()


def test_boss_dies_to_spell():
    """The Wizard's nova + staff kills a low-HP boss from a few tiles away -> cleared."""
    env = CDungeonSingle(DungeonConfig(boss_hp_max=600.0, invuln_ticks=1, opening_invuln_ticks=0, n_snakes=0, boss_shoots=False), seed=4)
    env.reset(seed=4)
    env.put(player_x=BOSS_TILE[0] + 4.0, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    cleared = False
    for _ in range(4000):
        boss = env.get()
        aim = _aim_at((boss["px"], boss["py"]), (boss["boss_x"], boss["boss_y"]))
        _, _, term, trunc, info = env.step([0, aim, 1, 1])  # shoot + cast at the boss
        if term or trunc:
            cleared = bool(info["cleared"])
            break
    assert cleared
    env.close()
