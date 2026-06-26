"""Whole-dungeon Snake Pit env over the real map: navigate entrance -> boss room, then fight the
faithful 3-phase Stheno (spec: docs/snakepit-spec.md, extracted from betterSkillys source).

Stage 1 (here): navigation (BFS geodesic reward) + core boss fight (HP-gated 3 phases with
invuln transitions, aimed + rotating bullet spreads, projectiles, player shooting, completion).
Grenades/status-effects and minions layer on next. Completion = Stheno dead = dungeon cleared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rotmg_rl.sim.snakepit_map import DungeonMap, find_objects, load_jm

GRID = 15  # egocentric local view (tiles), player-centered
_ANGLES = np.arange(8, dtype=np.float32) * (np.pi / 4.0)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1).astype(np.float32)  # (8,2)

# grid channels
CH_WALL, CH_BOSS, CH_EBULLET, CH_EBVX, CH_EBVY, CH_PBULLET = range(6)
NUM_CH = 6
NUM_SCALARS = 8  # dir_to_obj(2), geodist_norm, hp_frac, boss_hp_frac, phase_norm, in_room, aim_ready


@dataclass
class DungeonConfig:
    # navigation
    player_speed: float = 0.6
    player_radius: float = 0.4
    max_steps: int = 4000
    boss_room_radius: float = 10.0  # within this of boss = in the boss room (fight active)
    activation_range: float = 20.0
    # player combat
    player_hp_max: float = 200.0
    shoot_cooldown: int = 2
    player_bullet_speed: float = 1.2
    player_bullet_dmg: float = 40.0
    player_bullet_radius: float = 0.5
    player_bullet_life: int = 25
    # boss (ScaleHP2(20) in source; sim uses a feasible single-player value)
    boss_hp_max: float = 2500.0
    boss_radius: float = 2.0
    boss_speed: float = 0.12
    invuln_ticks: int = 15
    # projectiles (tiles/tick, lifetime ticks ~ source Speed/10, LifetimeMS/100)
    ebullet_speed: float = 0.7
    ebullet_life: int = 15
    ebullet_dmg: float = 30.0
    ebullet_radius: float = 0.4
    max_bullets: int = 4096
    # rewards
    rew_progress: float = 0.1
    rew_reach: float = 20.0
    rew_boss_dmg: float = 0.01  # per HP
    rew_survive: float = 0.002
    rew_damage_taken: float = 0.02  # per HP
    rew_clear: float = 100.0
    rew_death: float = 20.0
    rew_step: float = -0.002


def _nearest_walkable(walkable, x, y):
    if walkable[y, x]:
        return x, y
    ys, xs = np.where(walkable)
    i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[i]), int(ys[i])


def geodesic_field(walkable, goal):
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
                "grid": gym.spaces.Box(-1.0, 1.0, (NUM_CH, GRID, GRID), np.float32),
                "scalars": gym.spaces.Box(-1.0, 1.0, (NUM_SCALARS,), np.float32),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([9, 9])  # move(0=stay,1-8), aim(0=none,1-8)
        self._rng = np.random.default_rng()

    # --- core loop ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        self.steps = 0
        self.player_pos = np.array([self.entrance_xy[0] + 0.5, self.entrance_xy[1] + 0.5], np.float32)
        self.player_hp = c.player_hp_max
        self.prev_geodist = self._geodist_at(self.player_pos)
        self.shoot_timer = 0
        self.boss_pos = np.array([self.boss_xy[0] + 0.5, self.boss_xy[1] + 0.5], np.float32)
        self.boss_hp = c.boss_hp_max
        self.phase = 0  # 0 none(nav), 1/2/3 fight phases, -1 transition
        self.fight_active = False
        self.invuln_timer = 0
        self.phase_shoot_timers: dict = {}
        self.rotate_angle = 0.0
        self.enemy_bullets = np.zeros((0, 5), np.float32)  # x,y,vx,vy,life
        self.player_bullets = np.zeros((0, 5), np.float32)
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        move_idx, aim_idx = int(action[0]), int(action[1])
        reward = c.rew_step
        in_room = float(np.linalg.norm(self.player_pos - self.boss_pos)) <= c.boss_room_radius

        # player movement (wall-sliding)
        if move_idx > 0:
            self._try_move(DIRS[move_idx - 1] * c.player_speed)

        # navigation reward until the boss room
        gd = self._geodist_at(self.player_pos)
        if not self.fight_active:
            reward += (self.prev_geodist - gd) * c.rew_progress
        self.prev_geodist = gd

        # activate fight when close enough
        if not self.fight_active and np.linalg.norm(self.player_pos - self.boss_pos) <= c.activation_range:
            self.fight_active = True
            self.phase = 1
            reward += c.rew_reach

        # player shooting
        self.shoot_timer = max(0, self.shoot_timer - 1)
        if aim_idx > 0 and self.shoot_timer == 0:
            vel = DIRS[aim_idx - 1] * c.player_bullet_speed
            self.player_bullets = self._append(self.player_bullets, [[*self.player_pos, vel[0], vel[1], c.player_bullet_life]])
            self.shoot_timer = c.shoot_cooldown

        if self.fight_active:
            reward += self._boss_tick()

        # advance bullets
        self.player_bullets = self._advance(self.player_bullets)
        self.enemy_bullets = self._advance(self.enemy_bullets)

        # player bullets vs boss (only when vulnerable)
        if self.player_bullets.shape[0] and self.invuln_timer == 0 and self.phase > 0:
            hit = np.linalg.norm(self.player_bullets[:, :2] - self.boss_pos, axis=1) < (c.boss_radius + c.player_bullet_radius)
            n = int(hit.sum())
            if n:
                dmg = n * c.player_bullet_dmg
                self.boss_hp -= dmg
                reward += dmg * c.rew_boss_dmg
                self.player_bullets = self.player_bullets[~hit]

        # enemy bullets vs player
        if self.enemy_bullets.shape[0]:
            hit = np.linalg.norm(self.enemy_bullets[:, :2] - self.player_pos, axis=1) < (c.player_radius + c.ebullet_radius)
            n = int(hit.sum())
            if n:
                self.player_hp -= n * c.ebullet_dmg
                reward -= n * c.ebullet_dmg * c.rew_damage_taken
                self.enemy_bullets = self.enemy_bullets[~hit]

        if self.fight_active:
            reward += c.rew_survive
        self.steps += 1

        terminated, cleared = False, False
        if self.boss_hp <= 0.0 and self.phase > 0:
            terminated, cleared = True, True
            reward += c.rew_clear
        elif self.player_hp <= 0.0:
            terminated = True
            reward -= c.rew_death
        truncated = (not terminated) and self.steps >= c.max_steps
        info = {"cleared": cleared, "in_room": in_room, "phase": self.phase, "boss_hp_frac": max(self.boss_hp, 0) / c.boss_hp_max, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    # --- boss behaviour (faithful core; grenades/status/minions next) ------

    def _boss_tick(self) -> float:
        c = self.cfg
        # HP-gated phase transitions with invulnerability
        if self.invuln_timer > 0:
            self.invuln_timer -= 1
        else:
            frac = self.boss_hp / c.boss_hp_max
            if self.phase == 1 and frac <= 0.66:
                self.phase, self.invuln_timer = 2, c.invuln_ticks
            elif self.phase == 2 and frac <= 0.33:
                self.phase, self.invuln_timer = 3, c.invuln_ticks
        # drift toward player (Wander/approach)
        to_p = self.player_pos - self.boss_pos
        d = np.linalg.norm(to_p) + 1e-6
        self.boss_pos = self.boss_pos + (to_p / d) * c.boss_speed
        if self.invuln_timer > 0:
            return 0.0
        # phase shoots (aimed spreads; rotating in phases 2-3)
        if self.phase == 1:
            self._aimed_shoot("p1", count=3, spread_deg=15, cooldown=15)
        elif self.phase == 2:
            self._rotating_shoot("p2", count=4, step_deg=15, cooldown=3)
        elif self.phase == 3:
            self._aimed_shoot("p3a", count=3, spread_deg=15, cooldown=15)
            self._rotating_shoot("p3b", count=4, step_deg=15, cooldown=5)
        return 0.0

    def _aimed_shoot(self, key, count, spread_deg, cooldown):
        t = self.phase_shoot_timers.get(key, 0)
        if t > 0:
            self.phase_shoot_timers[key] = t - 1
            return
        self.phase_shoot_timers[key] = cooldown
        base = np.arctan2(*(self.player_pos - self.boss_pos)[::-1])
        gap = np.radians(spread_deg)
        self._spawn_burst(base, count, gap)

    def _rotating_shoot(self, key, count, step_deg, cooldown):
        t = self.phase_shoot_timers.get(key, 0)
        if t > 0:
            self.phase_shoot_timers[key] = t - 1
            return
        self.phase_shoot_timers[key] = cooldown
        self.rotate_angle += np.radians(step_deg)
        gap = 2 * np.pi / count
        self._spawn_burst(self.rotate_angle, count, gap)

    def _spawn_burst(self, base_angle, count, gap):
        c = self.cfg
        idxs = np.arange(count) - (count - 1) / 2.0
        angles = base_angle + idxs * gap
        vel = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32) * c.ebullet_speed
        pos = np.tile(self.boss_pos, (count, 1)).astype(np.float32)
        life = np.full((count, 1), c.ebullet_life, np.float32)
        self.enemy_bullets = self._append(self.enemy_bullets, np.concatenate([pos, vel, life], 1))

    # --- helpers -----------------------------------------------------------

    def _walkable_at(self, pos) -> bool:
        x, y = int(pos[0]), int(pos[1])
        h, w = self.map.walkable.shape
        return 0 <= x < w and 0 <= y < h and bool(self.map.walkable[y, x])

    def _try_move(self, step_vec):
        for axis in (0, 1):
            cand = self.player_pos.copy()
            cand[axis] += step_vec[axis]
            if self._walkable_at(cand):
                self.player_pos = cand

    def _geodist_at(self, pos) -> float:
        x, y = int(pos[0]), int(pos[1])
        h, w = self.geodist.shape
        if 0 <= x < w and 0 <= y < h and np.isfinite(self.geodist[y, x]):
            return float(self.geodist[y, x])
        return self.max_geodist

    def _advance(self, bullets):
        if bullets.shape[0] == 0:
            return bullets
        bullets = bullets.copy()
        bullets[:, :2] += bullets[:, 2:4]
        bullets[:, 4] -= 1.0
        h, w = self.map.walkable.shape
        ix = np.clip(bullets[:, 0].astype(int), 0, w - 1)
        iy = np.clip(bullets[:, 1].astype(int), 0, h - 1)
        alive = (bullets[:, 4] > 0) & self.map.walkable[iy, ix]
        return bullets[alive]

    def _append(self, arr, new):
        new = np.asarray(new, np.float32)
        out = np.concatenate([arr, new], 0) if arr.shape[0] else new
        return out[-self.cfg.max_bullets :] if out.shape[0] > self.cfg.max_bullets else out

    def _obs(self):
        c = self.cfg
        half = GRID // 2
        p = self.player_pos
        grid = np.zeros((NUM_CH, GRID, GRID), np.float32)
        h, w = self.map.walkable.shape
        ys, xs = np.mgrid[0:GRID, 0:GRID]
        wx = (p[0] + xs - half).astype(int)
        wy = (p[1] + ys - half).astype(int)
        inb = (wx >= 0) & (wx < w) & (wy >= 0) & (wy < h)
        grid[CH_WALL][inb] = (~self.map.walkable[np.clip(wy, 0, h - 1), np.clip(wx, 0, w - 1)])[inb].astype(np.float32)
        self._scatter(grid, CH_BOSS, (self.boss_pos - p).reshape(1, 2), 1.0) if self.fight_active else None
        if self.enemy_bullets.shape[0]:
            rel = self.enemy_bullets[:, :2] - p
            self._scatter(grid, CH_EBULLET, rel, 1.0)
            spd = np.linalg.norm(self.enemy_bullets[:, 2:4], axis=1, keepdims=True) + 1e-6
            u = self.enemy_bullets[:, 2:4] / spd
            self._scatter(grid, CH_EBVX, rel, u[:, 0])
            self._scatter(grid, CH_EBVY, rel, u[:, 1])
        if self.player_bullets.shape[0]:
            self._scatter(grid, CH_PBULLET, self.player_bullets[:, :2] - p, 1.0)
        to_obj = (self.boss_pos - p) if self.fight_active else (np.array(self.boss_xy, np.float32) + 0.5 - p)
        dist = float(np.linalg.norm(to_obj)) + 1e-6
        scalars = np.array(
            [
                to_obj[0] / dist,
                to_obj[1] / dist,
                self._geodist_at(p) / self.max_geodist,
                self.player_hp / c.player_hp_max,
                max(self.boss_hp, 0) / c.boss_hp_max,
                self.phase / 3.0,
                1.0 if self.fight_active else 0.0,
                1.0 if self.shoot_timer == 0 else 0.0,
            ],
            np.float32,
        )
        return {"grid": grid, "scalars": scalars}

    def _scatter(self, grid, ch, rel, value):
        half = GRID // 2
        cells = np.floor(rel).astype(int) + half
        inb = (cells[:, 0] >= 0) & (cells[:, 0] < GRID) & (cells[:, 1] >= 0) & (cells[:, 1] < GRID)
        cx, cy = cells[inb, 0], cells[inb, 1]
        grid[ch, cy, cx] = value[inb] if isinstance(value, np.ndarray) else value
