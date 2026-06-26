"""Whole-dungeon Snake Pit env over the real map: navigate entrance -> boss room, then fight the
faithful 3-phase Stheno as a WIZARD (staff + spell). Calibrated to the betterSkillys source
(docs/snakepit-spec.md). Tick dt = 0.1s; projectile tiles/tick ~= XML Speed / 10.

Modeled: real map navigation (BFS geodesic reward), Wizard (HP/MP, 2-shot staff, Spell nuke
burst, MP regen), Stheno 3-phase fight (HP gates, aimed + rotating spreads). NEXT fidelity
layers (noted in spec): boss grenades + Confused/Petrify, Stheno Swarm minions, path enemies,
debug renderer. Completion = Stheno dead = dungeon cleared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rotmg_rl.sim.snakepit_map import DungeonMap, find_objects, load_jm

GRID = 15
_ANGLES = np.arange(8, dtype=np.float32) * (np.pi / 4.0)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1).astype(np.float32)  # (8,2)

CH_WALL, CH_BOSS, CH_EBULLET, CH_EBVX, CH_EBVY, CH_PBULLET = range(6)
NUM_CH = 6
NUM_SCALARS = 9  # dir(2), geodist, hp, mp, boss_hp, phase, in_room, spell_ready
# bullet columns: x, y, vx, vy, life, dmg
BX, BY, BVX, BVY, BLIFE, BDMG = range(6)


@dataclass
class DungeonConfig:
    # navigation
    player_speed: float = 0.5  # tiles/tick (~5 tiles/s)
    player_radius: float = 0.4
    max_steps: int = 4000
    boss_room_radius: float = 10.0
    activation_range: float = 20.0
    spawn_in_room_prob: float = 0.0  # curriculum: prob of spawning near the boss (skip navigation)
    # Wizard (max-level stats, calibrated)
    player_hp_max: float = 670.0
    player_mp_max: float = 385.0
    mp_regen: float = 0.5  # per tick (~5/s)
    # staff: 2 parallel shots, dmg 45-85, speed 180->1.8 t/tick, life 475ms->~8 ticks, ~5 shots/s
    staff_cooldown: int = 2
    staff_num: int = 2
    staff_dmg: tuple[float, float] = (45.0, 85.0)
    staff_speed: float = 1.8
    staff_life: int = 8
    staff_radius: float = 0.5
    staff_offset: float = 0.5  # parallel-shot lateral offset (tiles)
    # spell nuke: burst of ~20 shots in an arc, dmg 110-205, speed 160->1.6, life 1000ms->10
    spell_cost: float = 100.0
    spell_cooldown: int = 5
    spell_num: int = 20
    spell_arc_deg: float = 70.0
    spell_dmg: tuple[float, float] = (110.0, 205.0)
    spell_speed: float = 1.6
    spell_life: int = 10
    # boss
    boss_hp_max: float = 7500.0  # ScaleHP2(20) in source; this is the base
    boss_radius: float = 2.0
    boss_speed: float = 0.12
    invuln_ticks: int = 15
    ebullet_speed: float = 0.7  # proj0 70->0.7
    ebullet_life: int = 15
    ebullet_dmg: float = 40.0
    ebullet_radius: float = 0.4
    max_bullets: int = 8192
    # rewards
    rew_progress: float = 0.1
    rew_reach: float = 20.0
    rew_boss_dmg: float = 0.004  # per HP (boss has 7500)
    rew_survive: float = 0.002
    rew_damage_taken: float = 0.01
    rew_clear: float = 150.0
    rew_death: float = 30.0
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
        self.action_space = gym.spaces.MultiDiscrete([9, 9, 2])  # move, aim(staff), cast spell
        self._rng = np.random.default_rng()

    # --- core loop ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        self.steps = 0
        if self._rng.random() < c.spawn_in_room_prob:
            # curriculum: spawn near the boss on a walkable tile (practice the fight)
            bx, by = self.boss_xy
            ang = self._rng.uniform(0, 2 * np.pi)
            sx, sy = int(bx + 6 * np.cos(ang)), int(by + 6 * np.sin(ang))
            sx, sy = _nearest_walkable(self.map.walkable, np.clip(sx, 1, self.map.width - 2), np.clip(sy, 1, self.map.height - 2))
            self.player_pos = np.array([sx + 0.5, sy + 0.5], np.float32)
        else:
            self.player_pos = np.array([self.entrance_xy[0] + 0.5, self.entrance_xy[1] + 0.5], np.float32)
        self.player_hp = c.player_hp_max
        self.player_mp = c.player_mp_max
        self.prev_geodist = self._geodist_at(self.player_pos)
        self.staff_timer = 0
        self.spell_timer = 0
        self.boss_pos = np.array([self.boss_xy[0] + 0.5, self.boss_xy[1] + 0.5], np.float32)
        self.boss_hp = c.boss_hp_max
        self.phase = 0
        self.fight_active = False
        self.invuln_timer = 0
        self.shoot_timers: dict = {}
        self.rotate_angle = 0.0
        self.enemy_bullets = np.zeros((0, 6), np.float32)
        self.player_bullets = np.zeros((0, 6), np.float32)
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        move_idx, aim_idx, cast = int(action[0]), int(action[1]), int(action[2])
        reward = c.rew_step

        if move_idx > 0:
            self._try_move(DIRS[move_idx - 1] * c.player_speed)

        gd = self._geodist_at(self.player_pos)
        if not self.fight_active:
            reward += (self.prev_geodist - gd) * c.rew_progress
        self.prev_geodist = gd

        if not self.fight_active and np.linalg.norm(self.player_pos - self.boss_pos) <= c.activation_range:
            self.fight_active = True
            self.phase = 1
            reward += c.rew_reach

        # Wizard staff: 2 parallel shots toward aim
        self.staff_timer = max(0, self.staff_timer - 1)
        self.spell_timer = max(0, self.spell_timer - 1)
        self.player_mp = min(c.player_mp_max, self.player_mp + c.mp_regen)
        if aim_idx > 0 and self.staff_timer == 0:
            self._fire_staff(DIRS[aim_idx - 1])
            self.staff_timer = c.staff_cooldown
        # Spell nuke (toward aim, or toward boss if no aim) if MP available
        if cast == 1 and self.spell_timer == 0 and self.player_mp >= c.spell_cost:
            direction = DIRS[aim_idx - 1] if aim_idx > 0 else self._unit(self.boss_pos - self.player_pos)
            self._cast_spell(direction)
            self.player_mp -= c.spell_cost
            self.spell_timer = c.spell_cooldown

        if self.fight_active:
            self._boss_tick()

        self.player_bullets = self._advance(self.player_bullets)
        self.enemy_bullets = self._advance(self.enemy_bullets)

        if self.player_bullets.shape[0] and self.invuln_timer == 0 and self.phase > 0:
            hit = np.linalg.norm(self.player_bullets[:, :2] - self.boss_pos, axis=1) < (c.boss_radius + c.staff_radius)
            if hit.any():
                dmg = float(self.player_bullets[hit, BDMG].sum())
                self.boss_hp -= dmg
                reward += dmg * c.rew_boss_dmg
                self.player_bullets = self.player_bullets[~hit]

        if self.enemy_bullets.shape[0]:
            hit = np.linalg.norm(self.enemy_bullets[:, :2] - self.player_pos, axis=1) < (c.player_radius + c.ebullet_radius)
            if hit.any():
                dmg = float(self.enemy_bullets[hit, BDMG].sum())
                self.player_hp -= dmg
                reward -= dmg * c.rew_damage_taken
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
        info = {"cleared": cleared, "in_room": float(self.fight_active), "phase": self.phase, "boss_hp_frac": max(self.boss_hp, 0) / c.boss_hp_max, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    # --- Wizard ------------------------------------------------------------

    def _fire_staff(self, direction):
        c = self.cfg
        perp = np.array([-direction[1], direction[0]], np.float32) * c.staff_offset
        vel = direction * c.staff_speed
        rows = []
        for i in range(c.staff_num):
            off = perp * (i - (c.staff_num - 1) / 2.0)
            dmg = self._rng.uniform(*c.staff_dmg)
            rows.append([self.player_pos[0] + off[0], self.player_pos[1] + off[1], vel[0], vel[1], c.staff_life, dmg])
        self.player_bullets = self._append(self.player_bullets, rows)

    def _cast_spell(self, direction):
        c = self.cfg
        base = np.arctan2(direction[1], direction[0])
        arc = np.radians(c.spell_arc_deg)
        angles = base + np.linspace(-arc / 2, arc / 2, c.spell_num)
        vel = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32) * c.spell_speed
        dmg = self._rng.uniform(*c.spell_dmg, size=(c.spell_num, 1)).astype(np.float32)
        pos = np.tile(self.player_pos, (c.spell_num, 1)).astype(np.float32)
        life = np.full((c.spell_num, 1), c.spell_life, np.float32)
        self.player_bullets = self._append(self.player_bullets, np.concatenate([pos, vel, life, dmg], 1))

    # --- boss (core; grenades/minions/status next) -------------------------

    def _boss_tick(self):
        c = self.cfg
        if self.invuln_timer > 0:
            self.invuln_timer -= 1
        else:
            frac = self.boss_hp / c.boss_hp_max
            if self.phase == 1 and frac <= 0.66:
                self.phase, self.invuln_timer = 2, c.invuln_ticks
            elif self.phase == 2 and frac <= 0.33:
                self.phase, self.invuln_timer = 3, c.invuln_ticks
        to_p = self.player_pos - self.boss_pos
        self.boss_pos = self.boss_pos + self._unit(to_p) * c.boss_speed
        if self.invuln_timer > 0:
            return
        if self.phase == 1:
            self._aimed_shoot("p1", 3, 15, 15)
        elif self.phase == 2:
            self._rotating_shoot("p2", 4, 15, 3)
        elif self.phase == 3:
            self._aimed_shoot("p3a", 3, 15, 15)
            self._rotating_shoot("p3b", 4, 15, 5)

    def _aimed_shoot(self, key, count, spread_deg, cooldown):
        if self.shoot_timers.get(key, 0) > 0:
            self.shoot_timers[key] -= 1
            return
        self.shoot_timers[key] = cooldown
        base = np.arctan2(*(self.player_pos - self.boss_pos)[::-1])
        self._spawn_burst(base, count, np.radians(spread_deg))

    def _rotating_shoot(self, key, count, step_deg, cooldown):
        if self.shoot_timers.get(key, 0) > 0:
            self.shoot_timers[key] -= 1
            return
        self.shoot_timers[key] = cooldown
        self.rotate_angle += np.radians(step_deg)
        self._spawn_burst(self.rotate_angle, count, 2 * np.pi / count)

    def _spawn_burst(self, base_angle, count, gap):
        c = self.cfg
        idxs = np.arange(count) - (count - 1) / 2.0
        angles = base_angle + idxs * gap
        vel = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32) * c.ebullet_speed
        pos = np.tile(self.boss_pos, (count, 1)).astype(np.float32)
        life = np.full((count, 1), c.ebullet_life, np.float32)
        dmg = np.full((count, 1), c.ebullet_dmg, np.float32)
        self.enemy_bullets = self._append(self.enemy_bullets, np.concatenate([pos, vel, life, dmg], 1))

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-6)

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
        bullets[:, :2] += bullets[:, BVX:BVY + 1]
        bullets[:, BLIFE] -= 1.0
        h, w = self.map.walkable.shape
        ix = np.clip(bullets[:, BX].astype(int), 0, w - 1)
        iy = np.clip(bullets[:, BY].astype(int), 0, h - 1)
        return bullets[(bullets[:, BLIFE] > 0) & self.map.walkable[iy, ix]]

    def _append(self, arr, new):
        new = np.asarray(new, np.float32).reshape(-1, 6)
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
        if self.fight_active:
            self._scatter(grid, CH_BOSS, (self.boss_pos - p).reshape(1, 2), 1.0)
        if self.enemy_bullets.shape[0]:
            rel = self.enemy_bullets[:, :2] - p
            self._scatter(grid, CH_EBULLET, rel, 1.0)
            u = self.enemy_bullets[:, BVX:BVY + 1] / (np.linalg.norm(self.enemy_bullets[:, BVX:BVY + 1], axis=1, keepdims=True) + 1e-6)
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
                self.player_mp / c.player_mp_max,
                max(self.boss_hp, 0) / c.boss_hp_max,
                self.phase / 3.0,
                1.0 if self.fight_active else 0.0,
                1.0 if (self.player_mp >= c.spell_cost and self.spell_timer == 0) else 0.0,
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

    # --- debug render (not faithful; just to follow along) -----------------

    def render(self, px_per_tile: int = 5):
        if self.render_mode != "rgb_array":
            return None
        h, w = self.map.walkable.shape
        img = np.where(self.map.walkable[..., None], np.array([40, 40, 48], np.uint8), np.array([12, 12, 16], np.uint8))
        img = np.repeat(np.repeat(img, px_per_tile, 0), px_per_tile, 1)

        def dot(pos, color, r=2):
            cx, cy = int(pos[0] * px_per_tile), int(pos[1] * px_per_tile)
            y0, y1 = max(0, cy - r), min(img.shape[0], cy + r + 1)
            x0, x1 = max(0, cx - r), min(img.shape[1], cx + r + 1)
            if y0 < y1 and x0 < x1:
                img[y0:y1, x0:x1] = color

        for b in self.enemy_bullets:
            dot(b[:2], (255, 140, 0), 1)
        for b in self.player_bullets:
            dot(b[:2], (90, 200, 255), 1)
        if self.fight_active:
            dot(self.boss_pos, (220, 40, 40), 4)
        dot(self.player_pos, (60, 230, 90), 3)
        return img
