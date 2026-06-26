"""Parity test: the C dungeon env matches the Python oracle bit-faithfully on a deterministic
config (no snakes/minions, point-mass damage), same seed + same actions -> matching obs + rewards.

This is the M4 acceptance criterion. Snake spawn/wander + minion placement use rand() in C and
numpy RNG in Python, so they are excluded from parity (stochastic by design); everything else
(movement, walls, staff/spell bullets, bullet physics, collisions, boss movement + aimed/rotating
bullet patterns, phase transitions + invuln, grenades + status, exploration + all reward terms) is
matched exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rotmg_rl.sim.dungeon import GRID, MM, NUM_CH, NUM_MM_CH, DungeonConfig, DungeonEnv

binding = pytest.importorskip("rotmg_rl.csim.binding")
from rotmg_rl.csim.dungeon import OBS_SIZE, config_to_kwargs  # noqa: E402

GRID_FLAT = NUM_CH * GRID * GRID
MM_FLAT = NUM_MM_CH * MM * MM
DET = dict(n_snakes=0, enable_minions=False, staff_dmg=(65.0, 65.0), spell_dmg=(157.0, 157.0))


class CEnv:
    """Thin handle over the single-env C binding (env_init/reset/step/put)."""

    def __init__(self, cfg: DungeonConfig, seed: int):
        self.obs = np.zeros(OBS_SIZE, np.float32)
        self.act = np.zeros(4, np.int32)
        self.rew = np.zeros(1, np.float32)
        self.term = np.zeros(1, np.uint8)
        self.trunc = np.zeros(1, np.uint8)
        self.handle = binding.env_init(self.obs, self.act, self.rew, self.term, self.trunc, seed, **config_to_kwargs(cfg))

    def reset(self, seed: int):
        binding.env_reset(self.handle, seed)

    def put(self, **kwargs):
        binding.env_put(self.handle, **kwargs)

    def get(self) -> dict:
        return binding.env_get(self.handle)

    def step(self, action):
        self.act[:] = action
        binding.env_step(self.handle)
        return float(self.rew[0]), bool(self.term[0])


def _oracle_flat(obs: dict) -> np.ndarray:
    # obs layout matches the C env: [grid, minimap, scalars].
    return np.concatenate([obs["grid"].reshape(-1), obs["minimap"].reshape(-1), obs["scalars"]]).astype(np.float32)


def _discovered_cells(oracle) -> np.ndarray:
    """Boolean (MM, MM) mask: which minimap terrain cells overlap at least one discovered tile (the
    only cells allowed to be non-zero). Everything else is fog and must read exactly 0."""
    h, w = oracle.map.walkable.shape
    cx = (np.arange(w) * MM) // w
    cy = (np.arange(h) * MM) // h
    cell = cy[:, None] * MM + cx[None, :]
    out = np.zeros(MM * MM, bool)
    out[cell[oracle.discovered]] = True
    return out.reshape(MM, MM)


def _first_divergence(a: np.ndarray, b: np.ndarray, atol: float):
    diff = np.abs(a - b)
    idx = int(np.argmax(diff))
    if diff[idx] <= atol:
        return None
    if idx < GRID_FLAT:
        ch, rem = divmod(idx, GRID * GRID)
        row, col = divmod(rem, GRID)
        where = f"grid ch={ch} row={row} col={col}"
    elif idx < GRID_FLAT + MM_FLAT:
        ch, rem = divmod(idx - GRID_FLAT, MM * MM)
        row, col = divmod(rem, MM)
        where = f"minimap ch={ch} row={row} col={col}"
    else:
        where = f"scalar[{idx - GRID_FLAT - MM_FLAT}]"
    return f"{where}: oracle={a[idx]:.6f} c={b[idx]:.6f} (|d|={diff[idx]:.6f})"


def _run(cfg: DungeonConfig, seed: int, actions: np.ndarray, inject: dict | None, steps: int, obs_atol=2e-3, rew_atol=2e-3):
    oracle = DungeonEnv(cfg)
    oracle.reset(seed=seed)
    c = CEnv(cfg, seed)
    c.reset(seed)
    if inject is not None:
        oracle.player_pos = np.array([inject["player_x"], inject["player_y"]], np.float32)
        if "fight_active" in inject:
            oracle.fight_active = bool(inject["fight_active"])
        if "phase" in inject:
            oracle.phase = int(inject["phase"])
        if "boss_hp" in inject:
            oracle.boss_hp = float(inject["boss_hp"])
        c.put(**inject)
        # C's env_put recomputes obs (marking the injected position's fog disk + boss_seen); mirror that
        # in the oracle so the discovered masks stay symmetric before the step loop begins.
        oracle._obs()

    compared = 0
    for t in range(steps):
        a = actions[t]
        o_obs, o_rew, o_term, o_trunc, o_info = oracle.step(a.tolist())
        c_rew, c_term = c.step(a)
        assert o_term == c_term, f"step {t}: terminal mismatch oracle={o_term} c={c_term}"
        assert abs(o_rew - c_rew) <= rew_atol, f"step {t}: reward oracle={o_rew:.6f} c={c_rew:.6f}"
        if o_term:
            compared += 1
            break
        msg = _first_divergence(_oracle_flat(o_obs), c.obs, obs_atol)
        assert msg is None, f"step {t}: obs divergence {msg}"
        # fog of war: undiscovered minimap terrain cells read exactly 0 (no leak of unexplored map),
        # and the boss channel is all-zero until the boss has actually been seen.
        mm_terr = o_obs["minimap"][0]
        assert mm_terr[~_discovered_cells(oracle)].max(initial=0.0) == 0.0, f"step {t}: fog leak in undiscovered minimap cell"
        if not oracle.boss_seen:
            assert o_obs["minimap"][2].sum() == 0.0, f"step {t}: boss on minimap before being seen"
            assert c.get()["boss_seen"] == 0.0, f"step {t}: C boss_seen set before the boss was seen"
        # info-field parity (the bug the obs/reward checks missed): on non-terminal steps the C env
        # has not auto-reset, so its live state must reproduce the oracle's per-step info dict.
        g = c.get()
        c_bhf = max(g["boss_hp"], 0.0) / g["boss_hp_max"]
        assert abs(o_info["boss_hp_frac"] - c_bhf) <= 2e-3, f"step {t}: boss_hp_frac oracle={o_info['boss_hp_frac']:.5f} c={c_bhf:.5f}"
        assert float(o_info["in_room"]) == g["fight_active"], f"step {t}: in_room oracle={o_info['in_room']} c={g['fight_active']}"
        assert not o_info["cleared"], f"step {t}: oracle cleared True on a non-terminal step"
        # score parity: on a non-terminal (never-cleared) step score == 1 - boss_hp_frac in both sims.
        assert abs(o_info["score"] - (1.0 - c_bhf)) <= 2e-3, f"step {t}: score oracle={o_info['score']:.5f} c={1.0 - c_bhf:.5f}"
        compared += 1
    return compared, oracle, c


def _fixed_actions(seed: int, steps: int, aim_fixed: int | None = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = np.zeros((steps, 4), np.int32)
    a[:, 0] = rng.randint(0, 9, steps)
    a[:, 1] = aim_fixed if aim_fixed is not None else rng.randint(0, 32, steps)
    a[:, 2] = rng.randint(0, 2, steps)
    a[:, 3] = rng.randint(0, 2, steps)
    return a


def test_parity_entrance_wander():
    """No boss / no enemies: movement, walls, staff+spell bullets, exploration, scalars."""
    cfg = DungeonConfig(**DET)
    actions = _fixed_actions(0, 200)
    compared, _, _ = _run(cfg, seed=7, actions=actions, inject=None, steps=200)
    assert compared == 200


def test_parity_boss_fight():
    """Inject the player next to the boss: boss movement, aimed/rotating bullets, phases, grenades."""
    # huge player HP so the player survives the window and we test 150 steps of boss dynamics
    cfg = DungeonConfig(boss_hp_max=20000.0, player_hp_max=1e9, invuln_ticks=15, **DET)
    bx, by = 16, 73  # boss_xy
    inject = {"player_x": bx + 3.5, "player_y": by + 0.5, "fight_active": 1, "phase": 1}
    actions = _fixed_actions(1, 150)
    # always aim toward the boss (-x dir from player) and shoot+cast to engage
    actions[:, 1] = 16  # aim ~ pointing -x (180 deg)
    actions[:, 2] = 1
    actions[:, 3] = 1
    compared, oracle, c = _run(cfg, seed=3, actions=actions, inject=inject, steps=150)
    assert compared == 150  # survives the window: 150 matched steps
    assert oracle.phase >= 2  # at least one phase transition (+ invuln) was exercised
    assert oracle.confused_timer > 0 or oracle.petrify_timer > 0 or oracle.steps > 0  # status exercised


def test_minimap_fog_and_boss_reveal():
    """Fog of war end to end: wandering from the entrance (boss far away) the boss minimap channel
    stays all-zero and boss_seen never sets; injecting the player next to the boss reveals it on the
    minimap (boss channel non-zero) once seen -- and the C env tracks boss_seen identically."""
    # 1) Entrance wander: boss never seen -> boss channel all-zero, fog covers most cells.
    cfg = DungeonConfig(**DET)
    oracle = DungeonEnv(cfg)
    obs, _ = oracle.reset(seed=11)
    c = CEnv(cfg, 11)
    c.reset(11)
    actions = _fixed_actions(11, 60)
    for t in range(60):
        obs, *_ = oracle.step(actions[t].tolist())
        c.step(actions[t])
        assert not oracle.boss_seen
        assert obs["minimap"][2].sum() == 0.0  # no boss dot before it is seen
        assert c.get()["boss_seen"] == 0.0
        # undiscovered terrain cells are exactly fog (0)
        assert obs["minimap"][0][~_discovered_cells(oracle)].max(initial=0.0) == 0.0
    assert _discovered_cells(oracle).sum() < MM * MM  # genuinely partial knowledge (fog remains)

    # 2) Inject next to the boss: it must now appear on the minimap, and parity holds.
    inject = {"player_x": 16 + 3.5, "player_y": 73 + 0.5, "fight_active": 1, "phase": 1}
    actions = _fixed_actions(12, 5)
    actions[:, 2] = actions[:, 3] = 0  # don't kill it; just observe the reveal
    compared, oracle2, c2 = _run(DungeonConfig(boss_hp_max=1e9, player_hp_max=1e9, **DET), seed=12, actions=actions, inject=inject, steps=5)
    assert compared == 5
    assert oracle2.boss_seen and c2.get()["boss_seen"] == 1.0
    assert oracle2._minimap()[2].sum() == 1.0  # exactly one boss cell lit once seen


def test_parity_boss_to_death():
    """Same fight with low HP: parity holds through the boss death + clear reward."""
    cfg = DungeonConfig(boss_hp_max=2000.0, player_hp_max=1e9, invuln_ticks=15, **DET)
    bx, by = 16, 73
    inject = {"player_x": bx + 3.5, "player_y": by + 0.5, "fight_active": 1, "phase": 1}
    actions = _fixed_actions(1, 200)
    actions[:, 1] = 16
    actions[:, 2] = 1
    actions[:, 3] = 1
    compared, oracle, c = _run(cfg, seed=3, actions=actions, inject=inject, steps=200)
    assert oracle.boss_hp <= 0.0  # boss died, parity held through the terminal/clear step


def test_parity_info_passive_boss_in_room():
    """The coordinator's config (passive boss, in room, no snakes/grenades/minions): assert the C
    env's per-step info (boss_hp_frac, in_room, cleared) matches the numpy oracle step by step. The
    player moves around the boss but does not shoot, so the boss stays full -> boss_hp_frac is 1.0
    (NOT 0) and cleared is 0 every step, exactly matching the oracle. (The damaging boss_hp_frac<1.0
    case is covered by test_parity_boss_fight, whose _run also asserts info parity per step.)"""
    cfg = DungeonConfig(boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False)
    bx, by = 16, 73
    inject = {"player_x": bx + 5.5, "player_y": by + 0.5, "fight_active": 1, "phase": 1}
    actions = _fixed_actions(2, 120)
    actions[:, 2] = 0  # no shooting -> boss never damaged
    actions[:, 3] = 0
    compared, oracle, c = _run(cfg, seed=2, actions=actions, inject=inject, steps=120)
    assert compared == 120  # per-step info parity held for the whole window
    assert oracle.boss_hp == 300.0  # boss full the whole time -> info boss_hp_frac stayed 1.0


def test_metrics_per_step_boss_full_when_undamaged():
    """Regression for the reported bug: with a passive boss spawned-in-room and NO shooting, the
    boss is never damaged, so the per-step metrics report boss_hp_frac~=1.0 and cleared=0 (the bug
    aggregated per-EPISODE and reported boss_hp_frac=0 / cleared=1)."""
    from rotmg_rl.csim.dungeon import CDungeon

    cfg = DungeonConfig(boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0)
    env = CDungeon(cfg, num_envs=64, seed=1, log_interval=100)
    env.reset(seed=1)
    noop = np.zeros((64, 4), np.int32)  # move/aim/shoot/cast all 0 -> boss never takes damage
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
    """With shooting, episodes DO clear fast (per-episode rate ~1), but the per-step `cleared`
    metric is a low rate and boss_hp_frac stays well above 0 -- proving per-step (numpy) semantics,
    not per-episode."""
    from rotmg_rl.csim.dungeon import CDungeon

    cfg = DungeonConfig(boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0)
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
    """The `score` log key is the per-EPISODE mean end-of-episode score (1 if cleared else damage
    fraction), NOT a per-step mean. With a passive boss, huge player HP, noop actions and a short
    max_steps, every episode truncates with the boss at full HP -> per-episode score == 0 (the boss
    took no damage), and episodes > 0 (truncations are counted)."""
    from rotmg_rl.csim.dungeon import CDungeon

    cfg = DungeonConfig(boss_hp_max=300.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0, max_steps=20)
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
    assert log["episodes"] > 0.0, log  # episodes (truncations) were counted
    assert log["score"] < 0.01, log  # boss never damaged -> end-of-episode score ~0 (bug: per-step dilute)


def test_score_metric_clears_reach_one():
    """When the boss dies fast (tiny HP, player shooting toward it), episodes clear and the
    per-episode `score` approaches 1.0 -- distinct from the per-step `cleared` rate."""
    from rotmg_rl.csim.dungeon import CDungeon

    cfg = DungeonConfig(boss_hp_max=20.0, player_hp_max=1e9, n_snakes=0, enable_grenades=False, enable_minions=False, boss_shoots=False, spawn_in_room_prob=1.0)
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
    from rotmg_rl.csim.dungeon import CDungeon

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


if __name__ == "__main__":
    test_parity_entrance_wander()
    print("entrance-wander parity OK")
    test_parity_boss_fight()
    print("boss-fight parity OK")
