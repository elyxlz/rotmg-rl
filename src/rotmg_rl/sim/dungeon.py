"""Whole-dungeon navigation over the real Snake Pit map (M1 navigation half).

The agent spawns at the dungeon entrance and must reach the boss room. Movement is 8-directional
on the real walkable grid (walls block). A BFS geodesic distance field from the boss provides a
dense, wall-aware navigation reward so corridor-following is learnable. The boss fight layers on
top once the agent reaches the boss room (added next).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rotmg_rl.sim.snakepit_map import DungeonMap, find_objects, load_jm

GRID = 15  # egocentric local wall view (tiles), odd so the player is centered
_ANGLES = np.arange(8, dtype=np.float32) * (np.pi / 4.0)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1).astype(np.float32)  # (8,2)
NUM_SCALARS = 4  # dir_to_boss(2), geodist_norm, hp_frac


@dataclass
class DungeonConfig:
    player_speed: float = 0.6  # tiles per tick
    player_radius: float = 0.4
    max_steps: int = 3000
    reach_radius: float = 6.0  # tiles from boss = "reached boss room"
    rew_progress: float = 0.1  # per tile of geodesic progress toward the boss
    rew_reach: float = 50.0
    rew_step: float = -0.002


def _nearest_walkable(walkable: np.ndarray, x: int, y: int) -> tuple[int, int]:
    if walkable[y, x]:
        return x, y
    h, w = walkable.shape
    best, bestd = (x, y), 1e9
    ys, xs = np.where(walkable)
    d = (xs - x) ** 2 + (ys - y) ** 2
    i = int(np.argmin(d))
    return int(xs[i]), int(ys[i])


def geodesic_field(walkable: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """BFS distance (in tiles) from goal over walkable cells; inf where unreachable."""
    h, w = walkable.shape
    dist = np.full((h, w), np.inf, np.float32)
    gx, gy = goal
    dq = deque([(gx, gy)])
    dist[gy, gx] = 0.0
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while dq:
        x, y = dq.popleft()
        for dx, dy in nbrs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx] and not np.isfinite(dist[ny, nx]):
                dist[ny, nx] = dist[y, x] + 1.0
                dq.append((nx, ny))
    return dist


class DungeonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: DungeonConfig | None = None, dmap: DungeonMap | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or DungeonConfig()
        self.map = dmap or load_jm()
        self.render_mode = render_mode
        boss = find_objects(self.map, "Stheno")
        portal = find_objects(self.map, "Portal")
        self.boss_xy = _nearest_walkable(self.map.walkable, *(boss[0] if boss else (16, 73)))
        self.entrance_xy = _nearest_walkable(self.map.walkable, *(portal[0] if portal else (110, 21)))
        self.geodist = geodesic_field(self.map.walkable, self.boss_xy)
        finite = self.geodist[np.isfinite(self.geodist)]
        self.max_geodist = float(finite.max()) if finite.size else 1.0
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(0.0, 1.0, (1, GRID, GRID), np.float32),
                "scalars": gym.spaces.Box(-1.0, 1.0, (NUM_SCALARS,), np.float32),
            }
        )
        self.action_space = gym.spaces.Discrete(9)  # 0 stay, 1-8 move dir
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.player_pos = np.array([self.entrance_xy[0] + 0.5, self.entrance_xy[1] + 0.5], np.float32)
        self.prev_geodist = self._geodist_at(self.player_pos)
        return self._obs(), {}

    def step(self, action: int):
        c = self.cfg
        a = int(action)
        if a > 0:
            step_vec = DIRS[a - 1] * c.player_speed
            self._try_move(step_vec)
        gd = self._geodist_at(self.player_pos)
        reward = (self.prev_geodist - gd) * c.rew_progress + c.rew_step
        self.prev_geodist = gd
        self.steps += 1
        dist_to_boss = float(np.hypot(*(self.player_pos - np.array(self.boss_xy) - 0.5)))
        reached = dist_to_boss <= c.reach_radius
        terminated = reached
        if reached:
            reward += c.rew_reach
        truncated = (not terminated) and self.steps >= c.max_steps
        info = {"reached": reached, "geodist": gd, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    # --- helpers -----------------------------------------------------------

    def _walkable_at(self, pos: np.ndarray) -> bool:
        x, y = int(pos[0]), int(pos[1])
        h, w = self.map.walkable.shape
        return 0 <= x < w and 0 <= y < h and bool(self.map.walkable[y, x])

    def _try_move(self, step_vec: np.ndarray) -> None:
        # Axis-separated so the agent slides along walls instead of sticking.
        for axis in (0, 1):
            cand = self.player_pos.copy()
            cand[axis] += step_vec[axis]
            if self._walkable_at(cand):
                self.player_pos = cand

    def _geodist_at(self, pos: np.ndarray) -> float:
        x, y = int(pos[0]), int(pos[1])
        h, w = self.geodist.shape
        if 0 <= x < w and 0 <= y < h and np.isfinite(self.geodist[y, x]):
            return float(self.geodist[y, x])
        return self.max_geodist

    def _obs(self) -> dict[str, np.ndarray]:
        half = GRID // 2
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        grid = np.zeros((1, GRID, GRID), np.float32)
        h, w = self.map.walkable.shape
        for gy in range(GRID):
            for gx in range(GRID):
                wx, wy = px + gx - half, py + gy - half
                if 0 <= wx < w and 0 <= wy < h and self.map.walkable[wy, wx]:
                    grid[0, gy, gx] = 1.0
        to_boss = np.array(self.boss_xy, np.float32) + 0.5 - self.player_pos
        dist = float(np.linalg.norm(to_boss)) + 1e-6
        scalars = np.array(
            [to_boss[0] / dist, to_boss[1] / dist, self._geodist_at(self.player_pos) / self.max_geodist, 1.0],
            np.float32,
        )
        return {"grid": grid, "scalars": scalars}
