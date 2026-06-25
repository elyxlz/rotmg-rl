"""Shared, world-agnostic observation schema.

This is the sim-to-real contract. The policy only ever sees the tensor produced here.
Two adapters build the same `GameState` and call `build_observation`:
  - the sim, from ground truth
  - the real game, from network packets (deploy time)

If both produce the same `GameState`, the policy cannot tell which world it is in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Egocentric grid: GRID x GRID cells centered on the player, each CELL world-units wide.
GRID = 15
CELL = 2.0  # world units per cell -> the agent sees a (GRID*CELL) square around itself

# Grid channels.
CH_WALL = 0
CH_ENEMY = 1
CH_ENEMY_BULLET = 2
CH_ENEMY_BULLET_VX = 3
CH_ENEMY_BULLET_VY = 4
CH_PLAYER_BULLET = 5
NUM_CHANNELS = 6

# Scalar features.
SCALARS = ("hp_frac", "mp_frac", "ability_ready", "boss_hp_frac", "dir_boss_x", "dir_boss_y")
NUM_SCALARS = len(SCALARS)

GRID_SHAPE = (NUM_CHANNELS, GRID, GRID)


@dataclass
class GameState:
    """World-agnostic snapshot. Built identically by the sim and the real-game adapter."""

    arena_size: float
    player_pos: np.ndarray  # (2,)
    player_hp: float
    player_hp_max: float
    player_mp: float
    player_mp_max: float
    ability_ready: bool
    boss_pos: np.ndarray  # (2,)
    boss_hp: float
    boss_hp_max: float
    enemy_bullets: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # x,y,vx,vy
    player_bullets: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # x,y,vx,vy


def _scatter(grid: np.ndarray, channel: int, rel: np.ndarray, value: np.ndarray | float = 1.0) -> None:
    """Place points (in world units relative to the player) onto a grid channel."""
    if rel.shape[0] == 0:
        return
    half = GRID // 2
    cells = np.floor(rel / CELL).astype(np.int64) + half
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < GRID) & (cells[:, 1] >= 0) & (cells[:, 1] < GRID)
    cx, cy = cells[inside, 0], cells[inside, 1]
    if isinstance(value, np.ndarray):
        grid[channel, cy, cx] = value[inside]
    else:
        grid[channel, cy, cx] = value


def build_observation(state: GameState) -> dict[str, np.ndarray]:
    """Return {'grid': (C,GRID,GRID) float32, 'scalars': (NUM_SCALARS,) float32}."""
    grid = np.zeros(GRID_SHAPE, np.float32)
    p = state.player_pos
    half = GRID // 2
    reach = half * CELL

    # Walls: cells whose world center falls outside the arena.
    ys, xs = np.mgrid[0:GRID, 0:GRID]
    world_x = p[0] + (xs - half) * CELL
    world_y = p[1] + (ys - half) * CELL
    outside = (world_x < 0) | (world_x > state.arena_size) | (world_y < 0) | (world_y > state.arena_size)
    grid[CH_WALL] = outside.astype(np.float32)

    # Boss occupancy.
    rel_boss = (state.boss_pos - p).reshape(1, 2)
    _scatter(grid, CH_ENEMY, rel_boss, 1.0)

    # Enemy bullets: presence + normalized velocity components.
    if state.enemy_bullets.shape[0] > 0:
        rel = state.enemy_bullets[:, :2] - p
        _scatter(grid, CH_ENEMY_BULLET, rel, 1.0)
        speed = np.linalg.norm(state.enemy_bullets[:, 2:4], axis=1, keepdims=True) + 1e-6
        unit = state.enemy_bullets[:, 2:4] / speed
        _scatter(grid, CH_ENEMY_BULLET_VX, rel, unit[:, 0])
        _scatter(grid, CH_ENEMY_BULLET_VY, rel, unit[:, 1])

    # Player bullets: presence.
    if state.player_bullets.shape[0] > 0:
        _scatter(grid, CH_PLAYER_BULLET, state.player_bullets[:, :2] - p, 1.0)

    to_boss = state.boss_pos - p
    dist = float(np.linalg.norm(to_boss)) + 1e-6
    scalars = np.array(
        [
            state.player_hp / max(state.player_hp_max, 1e-6),
            state.player_mp / max(state.player_mp_max, 1e-6),
            1.0 if state.ability_ready else 0.0,
            state.boss_hp / max(state.boss_hp_max, 1e-6),
            to_boss[0] / dist,
            to_boss[1] / dist,
        ],
        np.float32,
    )
    return {"grid": grid, "scalars": scalars}
