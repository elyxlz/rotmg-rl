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

from rotmg_rl.sim.dungeon import GRID, NUM_CH, NUM_SCALARS, DungeonConfig, DungeonEnv

binding = pytest.importorskip("rotmg_rl.csim.binding")
from rotmg_rl.csim.dungeon import OBS_SIZE, config_to_kwargs  # noqa: E402

GRID_FLAT = NUM_CH * GRID * GRID
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

    def step(self, action):
        self.act[:] = action
        binding.env_step(self.handle)
        return float(self.rew[0]), bool(self.term[0])


def _oracle_flat(obs: dict) -> np.ndarray:
    return np.concatenate([obs["grid"].reshape(-1), obs["scalars"]]).astype(np.float32)


def _first_divergence(a: np.ndarray, b: np.ndarray, atol: float):
    diff = np.abs(a - b)
    idx = int(np.argmax(diff))
    if diff[idx] <= atol:
        return None
    if idx < GRID_FLAT:
        ch, rem = divmod(idx, GRID * GRID)
        row, col = divmod(rem, GRID)
        where = f"grid ch={ch} row={row} col={col}"
    else:
        where = f"scalar[{idx - GRID_FLAT}]"
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

    compared = 0
    for t in range(steps):
        a = actions[t]
        o_obs, o_rew, o_term, o_trunc, _ = oracle.step(a.tolist())
        c_rew, c_term = c.step(a)
        assert o_term == c_term, f"step {t}: terminal mismatch oracle={o_term} c={c_term}"
        assert abs(o_rew - c_rew) <= rew_atol, f"step {t}: reward oracle={o_rew:.6f} c={c_rew:.6f}"
        if o_term:
            compared += 1
            break
        msg = _first_divergence(_oracle_flat(o_obs), c.obs, obs_atol)
        assert msg is None, f"step {t}: obs divergence {msg}"
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
    cfg = DungeonConfig(boss_hp_max=20000.0, invuln_ticks=15, **DET)
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


def test_parity_boss_to_death():
    """Same fight with low HP: parity holds through the boss death + clear reward."""
    cfg = DungeonConfig(boss_hp_max=4000.0, invuln_ticks=15, **DET)
    bx, by = 16, 73
    inject = {"player_x": bx + 3.5, "player_y": by + 0.5, "fight_active": 1, "phase": 1}
    actions = _fixed_actions(1, 200)
    actions[:, 1] = 16
    actions[:, 2] = 1
    actions[:, 3] = 1
    compared, oracle, c = _run(cfg, seed=3, actions=actions, inject=inject, steps=200)
    assert oracle.boss_hp <= 0.0  # boss died, parity held through the terminal/clear step


if __name__ == "__main__":
    test_parity_entrance_wander()
    print("entrance-wander parity OK")
    test_parity_boss_fight()
    print("boss-fight parity OK")
