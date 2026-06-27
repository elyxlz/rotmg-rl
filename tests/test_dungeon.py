"""Faithful dungeon env (v3): local vision, exploration, snakes, Wizard staff/spell, boss."""

import numpy as np

from rotmg_rl.sim.dungeon import GRID, MM, N_AIM, NUM_CH, NUM_MM_CH, NUM_SCALARS, DungeonConfig, DungeonEnv


def _aim_at(src, dst) -> int:
    ang = np.arctan2(dst[1] - src[1], dst[0] - src[0]) % (2 * np.pi)
    return int(round(ang / (2 * np.pi / N_AIM))) % N_AIM


def test_obs_shapes_and_space():
    env = DungeonEnv()
    obs, _ = env.reset(seed=0)
    assert obs["grid"].shape == (NUM_CH, GRID, GRID)
    assert obs["minimap"].shape == (NUM_MM_CH, MM, MM)
    assert obs["scalars"].shape == (NUM_SCALARS,)
    assert env.observation_space.contains(obs)
    assert (env.snakes[:, 2] > 0).sum() > 0  # snakes populate the dungeon
    # fog of war: at spawn only a disk is discovered (partial), and the boss is unseen -> no boss dot.
    assert 0 < int(env.discovered.sum()) < env.map.walkable.size
    assert not env.boss_seen and obs["minimap"][NUM_MM_CH - 1].sum() == 0.0


def test_random_steps_stay_in_space():
    env = DungeonEnv()
    env.reset(seed=1)
    for _ in range(60):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        if term or trunc:
            break


def test_exploration_grows_and_is_rewarded():
    env = DungeonEnv()
    env.reset(seed=2)
    before = int(env.visited.sum())
    total = 0.0
    for a in range(1, 9):  # walk in each direction a bit
        for _ in range(10):
            _, r, *_ = env.step([a, 0, 0, 0])
            total += r
    assert int(env.visited.sum()) > before  # new tiles visited
    assert total > 0  # exploration reward outweighs step cost


def test_staff_damages_a_snake():
    env = DungeonEnv()
    env.reset(seed=3)
    i = int(np.argmax(env.snakes[:, 2] > 0))
    env.player_pos = (env.snakes[i, :2] + np.array([2.0, 0.0], np.float32)).astype(np.float32)
    hp0 = env.snakes[i, 2]
    for _ in range(6):
        env.step([0, _aim_at(env.player_pos, env.snakes[i, :2]), 1, 0])
    assert env.snakes[i, 2] < hp0  # the snake took staff damage


def test_p1_blades_are_point_blank_only():
    """Fidelity: in P1 the boss fires its 3-blade shot ONLY when the player is within the point-blank
    acquire radius (2); at range it fires no blades (but still throws the Confused grenade)."""
    cfg = DungeonConfig(n_snakes=0, invuln_ticks=1, boss_wander_speed=0.0)
    env = DungeonEnv(cfg)
    env.reset(seed=5)
    env.fight_active, env.phase = True, 1
    # at range (8 tiles): no blades fire, but the grenade (range 11) lands on the standing player
    env.player_pos = (env.boss_pos + np.array([8.0, 0.0])).astype(np.float32)
    saw_grenade = False
    for _ in range(80):
        env.step([0, 0, 0, 0])
        saw_grenade = saw_grenade or env.grenades.shape[0] > 0
        assert env.enemy_bullets.shape[0] == 0  # no blades at range in P1
    assert saw_grenade  # the Confused grenade is the P1 ranged threat
    assert env.confused_timer > 0 or env.player_hp < cfg.player_hp_max  # grenade landed -> confused/damage
    # point-blank (within radius 2): the boss now acquires and fires blades
    env.player_pos = (env.boss_pos + np.array([1.5, 0.0])).astype(np.float32)
    fired = False
    for _ in range(40):
        env.step([0, 0, 0, 0])
        fired = fired or env.enemy_bullets.shape[0] > 0
    assert fired  # point-blank blades acquire and fire


def test_boss_dies_to_spell():
    env = DungeonEnv(DungeonConfig(boss_hp_max=600.0, invuln_ticks=1, n_snakes=0))
    env.reset(seed=4)
    env.player_pos = (env.boss_pos + np.array([4.0, 0.0], np.float32)).astype(np.float32)
    cleared = False
    for _ in range(env.cfg.max_steps):
        aim = _aim_at(env.player_pos, env.boss_pos)
        _, _, term, trunc, info = env.step([0, aim, 1, 1])  # shoot + cast at the boss
        if info["cleared"]:
            cleared = True
            break
        if term or trunc:
            break
    assert cleared
