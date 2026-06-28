"""Spec-derived scenario tests for the C Snake Pit env (_pufferlib/ocean/dungeon/dungeon.h, the single dynamics source).

These are NOT output-pinning golden tests (which would be circular -- they'd just re-assert whatever
the code emits). Each expected value is HAND-COMPUTED from the betterSkillys formulas the env mirrors:

  - DamageWithDefense clamp: dealt = max(raw * damage_floor, raw - defense)
  - HealthRegen (Player.cs HandleRegen): hp/s = 1 + 0.36*VIT (=1.54/tick at VIT 40), added while hp<max
  - the BulletNova spell: spell_num bullets, point-blank, each clamped by the boss defense
  - clear = boss dead (boss_hp<=0 & phase>0) terminates; reaching max_steps without that truncates

A `test_golden_trajectory` drift tripwire sits on top: a fixed seed + fixed action schedule whose
aggregate signals are committed, so any unintended change to the dynamics trips it.

Runs with numpy + the compiled C binding only (no pufferlib / torch), via the single-env wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rotmg_rl.csim.binding")
from rotmg_rl.config import GRID, DungeonConfig  # noqa: E402
from rotmg_rl.csim.single import OBS_SIZE, CDungeonSingle  # noqa: E402
from rotmg_rl.schedule import N_SNAKES_MAX, difficulty_config  # noqa: E402
from rotmg_rl.sim.snakepit_map import _nearest_walkable, authored_snakes, geodesic_field, load_jm  # noqa: E402

BOSS_TILE = (16, 73)  # _nearest_walkable(Stheno) on the real map (matches the C env's BOSS_X/Y)
ENTRANCE_TILE = (110, 21)  # _nearest_walkable(Portal of Cowardice) (matches the C env's ENTRANCE_X/Y)


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
    """A 100-dmg boss blade vs the real Wizard's DEF 25 deals max(100*0.1, 100-25) = max(10, 75) = 75
    (a SINGLE blade would take HP 670 -> 595). P1 fires a 3-blade volley point-blank; with regen off
    and the blade cooldown raised so only one volley lands, the three clamped hits take 670 -> 445."""
    per_blade = _defended(100.0, 25.0)
    assert per_blade == 75.0
    cfg = DungeonConfig(
        boss_hp_max=1e9,
        player_hp_max=670.0,
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
    assert env.get()["player_hp"] == pytest.approx(670.0 - 3 * per_blade)  # 445.0
    env.close()


def test_hp_regen_after_idle_ticks():
    """HealthRegen at VIT 40: (1 + 0.36*40)/s = 15.4/s = 1.54/tick, added flat while hp < max. After
    the player is hurt (one 3-blade volley -> 585 at DEF 25) and then idles T ticks far from any threat,
    HP rises by exactly hp_regen * T."""
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


def _clear_after_wait(wait_steps: int, max_steps: int = 200, rew_speed: float = 0.2) -> tuple[float, int]:
    """Drive a deterministic point-blank clear after `wait_steps` idle (noop) ticks, returning the
    total episode reward and the env step count at the clear. Everything but the fast-clear bonus is
    held flat: no per-step reward (rew_step/survive/explore = 0), no boss-damage reward (so the random
    nova roll can't leak in), no snakes/grenades, the boss can't shoot, and a fixed nova damage. The
    only thing that varies with wait_steps is the terminal rew_speed bonus -- isolating it."""
    cfg = DungeonConfig(
        boss_hp_max=100.0,  # one fixed-damage nova (20 * defended(185,19) = 3320) overkills -> clears
        player_hp_max=1e9,
        n_snakes=0,
        boss_wander_speed=0.0,
        boss_shoots=False,
        enable_grenades=False,
        spell_dmg_lo=185.0,
        spell_dmg_hi=185.0,  # fixed nova damage so the boss-damage step is identical in both runs
        rew_boss_dmg=0.0,
        rew_explore=0.0,
        rew_step=0.0,
        rew_survive=0.0,
        rew_clear=1.0,
        rew_speed=rew_speed,
        max_steps=max_steps,
        invuln_ticks=0,
        opening_invuln_ticks=0,
    )
    env = CDungeonSingle(cfg, seed=1)
    env.reset(seed=1)
    env.put(player_x=BOSS_TILE[0] + 0.5, player_y=BOSS_TILE[1] + 0.5, fight_active=1, phase=1)
    total, steps = 0.0, 0
    for _ in range(wait_steps):  # idle: stationary, boss unharmed -> every per-step signal is 0
        _, r, terminated, truncated, _ = env.step([0, 0, 0, 0])
        total += r
        steps += 1
        assert not (terminated or truncated)
    cleared = False
    for action in ([0, 0, 0, 1], [0, 0, 0, 0]):  # cast the nova, then let it resolve into the boss
        _, r, terminated, truncated, info = env.step(action)
        total += r
        steps += 1  # the env auto-resets steps on clear, so count what we drove (= env->steps at clear)
        cleared = info["cleared"]
        if terminated or truncated:
            break
    assert terminated and cleared
    env.close()
    return total, steps


def test_fast_clear_scores_higher_than_slow_clear():
    """The terminal fast-clear bonus rewards clearing with more time left: reward += rew_speed *
    (max_steps - steps)/max_steps on a win. Two otherwise-identical clears that differ only in how
    many idle ticks preceded the kill must differ in total reward by exactly that bonus delta,
    rew_speed * (steps_slow - steps_fast)/max_steps (the per-step terms are all zeroed here)."""
    max_steps, rew_speed = 200, 0.2
    total_fast, steps_fast = _clear_after_wait(0, max_steps=max_steps, rew_speed=rew_speed)
    total_slow, steps_slow = _clear_after_wait(40, max_steps=max_steps, rew_speed=rew_speed)
    assert steps_slow > steps_fast
    assert total_fast > total_slow  # clearing sooner is worth more
    expected_delta = rew_speed * (steps_slow - steps_fast) / max_steps
    assert (total_fast - total_slow) == pytest.approx(expected_delta, abs=1e-5)


def _swarm_fight_cfg(**overrides) -> DungeonConfig:
    """A clean point-blank boss fight that isolates the player-bullet-vs-boss path: no snakes, no boss
    fire, no grenades, no regen, the boss can't phase or die (huge HP), and a fixed staff damage so the
    only variable is whether the protective swarm is interposed. Overrides tweak enable_swarm etc."""
    base = {
        "boss_hp_max": 1e9,
        "player_hp_max": 1e9,
        "hp_regen": 0.0,
        "n_snakes": 0,
        "boss_wander_speed": 0.0,
        "boss_shoots": False,
        "enable_grenades": False,
        "staff_dmg_lo": 170.0,
        "staff_dmg_hi": 170.0,  # fixed staff damage so each landed bullet is identical
        "invuln_ticks": 0,
        "opening_invuln_ticks": 0,
    }
    base.update(overrides)
    return DungeonConfig(**base)


def _fire_at_boss(env, steps: int) -> float:
    """Hold the staff on the boss (aim -x, the boss sits at -x of the player) for `steps` ticks and
    return how much boss HP was removed over the window."""
    before = env.get()["boss_hp"]
    for _ in range(steps):
        env.step([0, 16, 1, 0])  # aim index 16 = angle pi = straight -x at the boss, shoot, no cast
    return before - env.get()["boss_hp"]


def test_protective_swarm_body_blocks_boss():
    """The defining Snake-Pit-C mechanic: Stheno's replenishing swarm body-blocks the player's bullets.

    Geometry (hand-derived): player point-blank at boss+3.0 tiles on +x, aiming -x at the boss. The
    swarm interposes on the player->boss line at boss + min(SWARM_INTERPOSE_DIST=2, d-0.5)=2.0 tiles,
    i.e. one tile in front of the player's muzzle. A staff bullet (speed 1.8) advances 19.5->17.7 in
    one tick and lands 0.8 tiles from the interpose point (< snake_radius+staff_radius = 1.0), so the
    swarm consumes it BEFORE the boss collision runs -> the boss takes ~zero damage while the wall is
    up. Clear the wall and the identical fire reaches the boss. The replenishing 1000-HP swarm is what
    the 98%-sim policy never faced, so it transferred to nothing on the real (walled) Stheno."""
    px, py = BOSS_TILE[0] + 3.5, BOSS_TILE[1] + 0.5  # boss at 16.5,73.5 -> player at 19.5 (d = 3.0)

    env = CDungeonSingle(_swarm_fight_cfg(enable_swarm=True, swarm_max=5), seed=3)
    env.reset(seed=3)
    env.put(player_x=px, player_y=py, fight_active=1, phase=1)
    env.step([0, 0, 0, 0])  # one fight tick -> Reproduce spawns the swarm and it interposes
    assert env.get()["swarm_count"] == 5  # ~swarm_max members maintained around the boss

    shielded_loss = _fire_at_boss(env, 20)
    assert shielded_loss == 0.0  # every staff bullet is eaten by the interposed wall

    env.put(clear_swarm=1)  # tear the wall down (and hold off replenishment)
    assert env.get()["swarm_count"] == 0
    cleared_loss = _fire_at_boss(env, 20)
    # with the lane open the identical fire reaches the boss: many clamped hits land (defended(170,19)
    # = 151 per bullet, ~8 bullets over the window), so the boss takes hundreds of HP, not zero.
    assert cleared_loss > 500.0


def test_protective_swarm_replenishes():
    """Reproduce tops the swarm back up to swarm_max on the swarm_cd cadence: kill the wall and, within
    swarm_cd ticks, it is fully rebuilt around the boss (the wall the policy can never simply outlast)."""
    px, py = BOSS_TILE[0] + 3.5, BOSS_TILE[1] + 0.5
    env = CDungeonSingle(_swarm_fight_cfg(enable_swarm=True, swarm_max=5, swarm_cd=15), seed=4)
    env.reset(seed=4)
    env.put(player_x=px, player_y=py, fight_active=1, phase=1)
    env.step([0, 0, 0, 0])
    assert env.get()["swarm_count"] == 5
    env.put(kill_swarm=1)  # wipe the live members but leave the Reproduce timer running
    assert env.get()["swarm_count"] == 0
    for _ in range(16):  # > swarm_cd (15): the next Reproduce tick must rebuild the wall
        env.step([0, 0, 0, 0])
    assert env.get()["swarm_count"] == 5  # replenished back to the cap, never starved
    env.close()


# --- navigation fidelity (collision footprint + geodesic potential) ---------------------------------


def test_footprint_blocks_concave_corner_cut():
    """The 0.5-tile player footprint (isValidPosition) cannot cut a concave corner the dimensionless
    point model cut. On the real map, tile (CX,CY) is floor, its east neighbor is a WALL, and its
    north neighbor is open -- a concave corner. A point at (CX+0.5,CY+0.5) stepping NE samples only
    tile centers, so walkable_at lets its x advance toward the wall; the 0.5 footprint's east edge
    reaches into the wall tile, so the x-advance is rejected and the player slides north along the
    wall instead. This is the cornering the live server enforces and the point sim never trained."""
    cx, cy = 97, 55
    m = load_jm()
    assert m.walkable[cy, cx] and not m.walkable[cy, cx + 1]  # the concave corner: floor with a wall to the east
    assert m.walkable[cy + 1, cx] and m.walkable[cy, cx - 1]  # north + west open (so only the corner is tight)
    cfg = DungeonConfig(player_speed=0.3, player_radius=0.5, n_snakes=0, boss_wander_speed=0.0, enable_grenades=False)
    env = CDungeonSingle(cfg, seed=0)
    env.reset(seed=0)
    env.put(player_x=cx + 0.5, player_y=cy + 0.5, fight_active=0, phase=0)
    env.step([2, 0, 0, 0])  # move_idx 2 = NE (g_move_dx[1]=cos45, g_move_dy[1]=sin45)
    s = env.get()
    # x stays put: the footprint blocks the eastward advance the point model (walkable_at on the floor
    # tile center) would have allowed. y advances freely -> the player slides north, never cutting in.
    assert s["px"] == pytest.approx(cx + 0.5, abs=1e-4)
    assert s["py"] > cy + 0.5 + 0.2
    env.close()


def _descend_geodesic(geo: np.ndarray, walkable: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    """Follow steepest geodesic descent (8-connected) from start to target. The geodesic field has no
    local minimum except the target, so the descent always arrives -- and every step is, by
    construction, strictly closer, which is exactly the property the approach reward needs."""
    h, w = walkable.shape
    x, y = start
    path = [(x, y)]
    while (x, y) != target:
        best, best_val = None, geo[y, x]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx] and geo[ny, nx] < best_val:
                    best, best_val = (nx, ny), geo[ny, nx]
        assert best is not None, f"geodesic descent stuck at {(x, y)} (a local minimum off the boss)"
        x, y = best
        path.append((x, y))
    return path


def test_geodesic_field_decreases_along_entrance_to_boss_path():
    """The privileged approach potential (baked into MAP_GEODESIC from this same function) decreases
    monotonically along the real entrance->boss route, so closing GEODESIC distance -- including the
    south-then-west detour that INCREASES euclidean distance -- always earns positive reward. The
    field's value at the entrance exceeds the straight-line distance, proving the detour is real."""
    m = load_jm()
    bx, by = _nearest_walkable(m.walkable, 16, 73)
    ex, ey = _nearest_walkable(m.walkable, 110, 21)
    geo = geodesic_field(m.walkable, (bx, by))
    assert geo[by, bx] == 0.0  # zero at the boss
    euclid = ((ex - bx) ** 2 + (ey - by) ** 2) ** 0.5
    assert np.isfinite(geo[ey, ex]) and geo[ey, ex] > euclid + 10.0  # the detour is ~28 tiles longer than straight-line
    path = _descend_geodesic(geo, m.walkable, (ex, ey), (bx, by))
    vals = [float(geo[y, x]) for x, y in path]
    assert all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))  # strictly closing every step


# --- authored map fidelity (real enemy positions + real entrance spawn + real walls) ----------------


def test_authored_enemies_spawn_at_real_positions():
    """At d=1 the env spawns every enemy at its AUTHORED .jm tile/type -- the exact real layout, not a
    uniform random scatter. With the full roster active and no density jitter, the set of spawned snake
    tiles is precisely the authored set (every cluster, including the entrance chokepoint, reproduced)."""
    env = CDungeonSingle(DungeonConfig(n_snakes=N_SNAKES_MAX, n_snakes_jitter=0), seed=7)
    env.reset(seed=7)
    spawned = {(int(np.floor(x)), int(np.floor(y))) for x, y in env.get()["snakes"]}
    env.close()
    authored = {(x, y) for x, y, _ in authored_snakes()}
    assert spawned == authored  # every authored tile occupied, nothing extra -> the real map exactly


def test_authored_chokepoint_present_at_d1():
    """The lethal entrance chokepoint (Fire Pythons + a Greater Pit Viper around (28-31, 41-44)) is
    actually spawned at d=1 -- the cluster the random-scatter sim never reproduced and the live policy
    dies in."""
    env = CDungeonSingle(DungeonConfig(n_snakes=N_SNAKES_MAX, n_snakes_jitter=0), seed=11)
    env.reset(seed=11)
    spawned = {(int(np.floor(x)), int(np.floor(y))) for x, y in env.get()["snakes"]}
    env.close()
    assert {(30, 43), (30, 44), (26, 46), (31, 43), (28, 41)} <= spawned


def test_authored_fraction_scales_with_difficulty():
    """The curriculum ramps the FRACTION of authored enemies active: low d spawns ~round(N_AUTHORED*d)
    of them (a subset of the authored tiles), d=1 the whole roster -- so easy early, the full real map
    at the end. n_snakes is the count lever; with jitter off the spawn count matches it exactly."""
    authored = {(x, y) for x, y, _ in authored_snakes()}
    low = difficulty_config(0.2)["n_snakes"]
    assert 0 < low < N_SNAKES_MAX
    env = CDungeonSingle(DungeonConfig(n_snakes=low, n_snakes_jitter=0), seed=3)
    env.reset(seed=3)
    spawned = {(int(np.floor(x)), int(np.floor(y))) for x, y in env.get()["snakes"]}
    env.close()
    assert len(spawned) == low  # exactly the scheduled count
    assert spawned <= authored  # always a subset of the real authored tiles, never random tiles


def test_player_spawns_at_real_entrance():
    """In the navigate-in regime the player starts at the REAL portal entrance (~110,21), so it learns
    THIS entrance's route to the boss -- not a random far tile. With both random + in-room spawn off,
    every reset lands on the entrance tile regardless of seed."""
    env = CDungeonSingle(DungeonConfig(n_snakes=0, spawn_in_room_prob=0.0, random_spawn_prob=0.0), seed=0)
    for s in range(5):
        env.reset(seed=s)
        st = env.get()
        assert (st["px"], st["py"]) == pytest.approx((ENTRANCE_TILE[0] + 0.5, ENTRANCE_TILE[1] + 0.5))
    env.close()


def test_env_wall_window_matches_real_jm():
    """Definitive walls-in-env check: the env's egocentric wall channel (CH_WALL) at the entrance equals
    the real .jm walkable grid tile-for-tile. The C env navigates the actual authored walls, not an open
    arena -- so the policy trains against the real corridors."""
    m = load_jm()
    env = CDungeonSingle(DungeonConfig(n_snakes=0, spawn_in_room_prob=0.0, random_spawn_prob=0.0), seed=0)
    obs = env.reset(seed=0)
    st = env.get()
    px, py = int(st["px"]), int(st["py"])
    assert (px, py) == ENTRANCE_TILE  # reset put us on the real entrance tile
    wall = obs[: GRID * GRID].reshape(GRID, GRID)  # CH_WALL is channel 0, centered on the player tile
    half = GRID // 2
    mismatches = 0
    for row in range(GRID):
        for col in range(GRID):
            wx, wy = px + col - half, py + row - half
            if 0 <= wx < m.width and 0 <= wy < m.height:
                expected = 0.0 if m.walkable[wy, wx] else 1.0  # 1 = wall in the obs channel
                mismatches += wall[row, col] != expected
    env.close()
    assert mismatches == 0


def test_entrance_wall_blocks_westward_shortcut():
    """The entrance room is walled to the west (the .jm wall band at x94-101, y21), so a player can't
    walk straight toward the SW boss -- it must take the real corridor. The env enforces that wall: a
    step due west from the wall-adjacent floor tile (102,21) is rejected (the footprint hits the wall)."""
    m = load_jm()
    assert m.walkable[21, 102] and not m.walkable[21, 101]  # floor with a wall immediately west
    cfg = DungeonConfig(player_speed=0.773, n_snakes=0, spawn_in_room_prob=0.0, random_spawn_prob=0.0)
    env = CDungeonSingle(cfg, seed=0)
    env.reset(seed=0)
    env.put(player_x=102.5, player_y=21.5, fight_active=0, phase=0)
    env.step([5, 0, 0, 0])  # move_idx 5 -> g_move_dx[4]=cos(pi)=-1: due west into the wall
    assert env.get()["px"] == pytest.approx(102.5, abs=1e-4)  # blocked: the real entrance wall holds
    env.close()


# --- drift tripwire ---------------------------------------------------------------------------------
# A fixed seed + deterministic action schedule, 200 steps, with the boss HP high enough that it never
# clears and the player HP high enough that it never dies (so the episode runs the full window). The
# committed aggregates were captured from the C env; an unintended dynamics change moves them. These
# are a TRIPWIRE, not a spec: regenerate them deliberately if the dynamics intentionally change.
GOLDEN_SEED = 2024
GOLDEN_STEPS = 200
# Regenerated for the authored-map fidelity fix: snake spawning moved from uniform-random tiles to the
# AUTHORED .jm positions/types (spawn_snakes draws a subset of AUTHORED_SNAKES), so the 30 in-room snakes
# now sit on real authored tiles with their real archetypes -- the trajectory legitimately moved.
GOLDEN = {"total_reward": 1.3400, "obs_checksum": 54701.94, "player_hp": 998425.0, "boss_hp": 3293.23}


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
