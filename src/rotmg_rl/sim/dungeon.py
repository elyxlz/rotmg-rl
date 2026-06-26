"""Faithful Snake Pit env (v3) over the real map, per docs/real-game-analysis.md.

Faithful to the real game: LOCAL vision only (VISIBILITY_RADIUS 15 -> 31x31 window, no global map
knowledge), continuous-ish mouse-aim (32 fine directions), exploration-based progression (no
path-to-boss breadcrumb), the dungeon populated with the real snakes you fight/dodge through, and
the Wizard (staff + Spell). Calibrated: dt=100ms, tiles/tick = Speed/100, cooldown ticks = ms/100.

Boss grenades + Confused/Petrify status + Stheno Swarm minions are the next fidelity layer.
Completion = Stheno dead = dungeon cleared.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rotmg_rl.sim.snakepit_map import DungeonMap, find_objects, load_jm

VIS_RADIUS = 15  # source VISIBILITY_RADIUS
GRID = 2 * VIS_RADIUS + 1  # 31x31 egocentric window
MOVE_DIRS = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1).astype(np.float32)
N_AIM = 32
AIM_DIRS = np.stack([np.cos(np.arange(N_AIM) * 2 * np.pi / N_AIM), np.sin(np.arange(N_AIM) * 2 * np.pi / N_AIM)], 1).astype(np.float32)

CH_WALL, CH_ENEMY, CH_EBULLET, CH_EBVX, CH_EBVY, CH_PBULLET = range(6)
NUM_CH = 6
NUM_SCALARS = 4  # hp, mp, spell_ready, boss_visible
BX, BY, BVX, BVY, BLIFE, BDMG = range(6)  # bullet columns
EX, EY, EHP, ETIMER = range(4)  # enemy columns


@dataclass
class DungeonConfig:
    player_speed: float = 0.55
    player_radius: float = 0.4
    max_steps: int = 4000
    activation_range: float = 20.0
    # Wizard
    player_hp_max: float = 670.0
    player_mp_max: float = 385.0
    mp_regen: float = 0.5
    staff_cooldown: int = 2
    staff_num: int = 2
    staff_dmg: tuple[float, float] = (45.0, 85.0)
    staff_speed: float = 1.8
    staff_life: int = 8
    staff_radius: float = 0.5
    staff_offset: float = 0.5
    spell_cost: float = 100.0
    spell_cooldown: int = 5
    spell_num: int = 20
    spell_arc_deg: float = 70.0
    spell_dmg: tuple[float, float] = (110.0, 205.0)
    spell_speed: float = 1.6
    spell_life: int = 10
    # snakes (Pit Snake/Viper HP 5, Speed 60->0.6 t/tick, Wander+Shoot ~1s)
    n_snakes: int = 40
    snake_hp: float = 5.0
    snake_speed: float = 0.2
    snake_shoot_range: float = 8.0
    snake_cooldown: int = 10
    snake_bullet_speed: float = 0.6
    snake_bullet_life: int = 12
    snake_bullet_dmg: float = 20.0
    snake_radius: float = 0.5
    # boss
    boss_hp_max: float = 7500.0
    boss_radius: float = 2.0
    boss_speed: float = 0.12
    invuln_ticks: int = 15
    ebullet_speed: float = 0.7
    ebullet_life: int = 15
    ebullet_dmg: float = 40.0
    ebullet_radius: float = 0.4
    max_bullets: int = 8192
    # rewards (exploration-based, no global pathfinding)
    rew_explore: float = 0.05  # per newly-visited tile
    rew_kill: float = 0.5  # per snake killed
    rew_boss_dmg: float = 0.004
    rew_reach: float = 20.0
    rew_survive: float = 0.001
    rew_damage_taken: float = 0.01
    rew_clear: float = 200.0
    rew_death: float = 30.0
    rew_step: float = -0.002


def _nearest_walkable(walkable, x, y):
    if walkable[y, x]:
        return x, y
    ys, xs = np.where(walkable)
    i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[i]), int(ys[i])


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
        self._walk_xs, self._walk_ys = np.where(self.map.walkable)
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(-1.0, 1.0, (NUM_CH, GRID, GRID), np.float32),
                "scalars": gym.spaces.Box(-1.0, 1.0, (NUM_SCALARS,), np.float32),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([9, N_AIM, 2, 2])  # move, aim(mouse), shoot, cast
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
        self.player_mp = c.player_mp_max
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
        self.visited = np.zeros(self.map.walkable.shape, bool)
        self._spawn_snakes()
        return self._obs(), {}

    def _spawn_snakes(self):
        c = self.cfg
        # snakes on random walkable tiles not too close to the entrance
        idx = self._rng.choice(len(self._walk_xs), size=min(c.n_snakes, len(self._walk_xs)), replace=False)
        xs, ys = self._walk_xs[idx], self._walk_ys[idx]
        keep = (np.abs(xs - self.entrance_xy[0]) + np.abs(ys - self.entrance_xy[1])) > 6
        xs, ys = xs[keep], ys[keep]
        self.snakes = np.zeros((len(xs), 4), np.float32)
        self.snakes[:, EX] = xs + 0.5
        self.snakes[:, EY] = ys + 0.5
        self.snakes[:, EHP] = c.snake_hp
        self.snakes[:, ETIMER] = self._rng.integers(0, c.snake_cooldown, size=len(xs))

    def step(self, action):
        c = self.cfg
        move_idx, aim_idx, shoot, cast = int(action[0]), int(action[1]), int(action[2]), int(action[3])
        reward = c.rew_step

        if move_idx > 0:
            self._try_move(MOVE_DIRS[move_idx - 1] * c.player_speed)

        # exploration reward: newly visited tile
        tx, ty = int(self.player_pos[0]), int(self.player_pos[1])
        if 0 <= ty < self.visited.shape[0] and 0 <= tx < self.visited.shape[1] and not self.visited[ty, tx]:
            self.visited[ty, tx] = True
            reward += c.rew_explore

        # Wizard: staff toward aim, spell toward aim
        self.staff_timer = max(0, self.staff_timer - 1)
        self.spell_timer = max(0, self.spell_timer - 1)
        self.player_mp = min(c.player_mp_max, self.player_mp + c.mp_regen)
        aim = AIM_DIRS[aim_idx]
        if shoot == 1 and self.staff_timer == 0:
            self._fire_staff(aim)
            self.staff_timer = c.staff_cooldown
        if cast == 1 and self.spell_timer == 0 and self.player_mp >= c.spell_cost:
            self._cast_spell(aim)
            self.player_mp -= c.spell_cost
            self.spell_timer = c.spell_cooldown

        if not self.fight_active and np.linalg.norm(self.player_pos - self.boss_pos) <= c.activation_range:
            self.fight_active = True
            self.phase = 1
            reward += c.rew_reach

        self._snakes_tick()
        if self.fight_active:
            self._boss_tick()

        self.player_bullets = self._advance(self.player_bullets)
        self.enemy_bullets = self._advance(self.enemy_bullets)
        reward += self._resolve_collisions()

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
        info = {"cleared": cleared, "in_room": float(self.fight_active), "snakes": int((self.snakes[:, EHP] > 0).sum()), "boss_hp_frac": max(self.boss_hp, 0) / c.boss_hp_max, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    def _resolve_collisions(self) -> float:
        c = self.cfg
        reward = 0.0
        # player bullets vs snakes
        if self.player_bullets.shape[0] and self.snakes.shape[0]:
            alive = self.snakes[:, EHP] > 0
            for i in np.where(alive)[0]:
                hit = np.linalg.norm(self.player_bullets[:, :2] - self.snakes[i, :2], axis=1) < (c.snake_radius + c.staff_radius)
                if hit.any():
                    self.snakes[i, EHP] -= float(self.player_bullets[hit, BDMG].sum())
                    self.player_bullets = self.player_bullets[~hit]
                    if self.snakes[i, EHP] <= 0:
                        reward += c.rew_kill
        # player bullets vs boss
        if self.player_bullets.shape[0] and self.invuln_timer == 0 and self.phase > 0:
            hit = np.linalg.norm(self.player_bullets[:, :2] - self.boss_pos, axis=1) < (c.boss_radius + c.staff_radius)
            if hit.any():
                dmg = float(self.player_bullets[hit, BDMG].sum())
                self.boss_hp -= dmg
                reward += dmg * c.rew_boss_dmg
                self.player_bullets = self.player_bullets[~hit]
        # enemy bullets vs player
        if self.enemy_bullets.shape[0]:
            hit = np.linalg.norm(self.enemy_bullets[:, :2] - self.player_pos, axis=1) < (c.player_radius + c.ebullet_radius)
            if hit.any():
                dmg = float(self.enemy_bullets[hit, BDMG].sum())
                self.player_hp -= dmg
                reward -= dmg * c.rew_damage_taken
                self.enemy_bullets = self.enemy_bullets[~hit]
        return reward

    # --- snakes ------------------------------------------------------------

    def _snakes_tick(self):
        c = self.cfg
        alive = self.snakes[:, EHP] > 0
        if not alive.any():
            return
        idx = np.where(alive)[0]
        # wander: small random drift, clamped to walkable
        drift = self._rng.normal(0, c.snake_speed, size=(len(idx), 2)).astype(np.float32)
        for k, i in enumerate(idx):
            cand = self.snakes[i, :2] + drift[k]
            if self._walkable_at(cand):
                self.snakes[i, :2] = cand
            self.snakes[i, ETIMER] -= 1
            d = self.snakes[i, :2] - self.player_pos
            if self.snakes[i, ETIMER] <= 0 and np.linalg.norm(d) <= c.snake_shoot_range:
                self.snakes[i, ETIMER] = c.snake_cooldown
                ang = np.arctan2(self.player_pos[1] - self.snakes[i, 1], self.player_pos[0] - self.snakes[i, 0])
                vel = np.array([np.cos(ang), np.sin(ang)], np.float32) * c.snake_bullet_speed
                self.enemy_bullets = self._append(self.enemy_bullets, [[self.snakes[i, 0], self.snakes[i, 1], vel[0], vel[1], c.snake_bullet_life, c.snake_bullet_dmg]])

    # --- Wizard ------------------------------------------------------------

    def _fire_staff(self, direction):
        c = self.cfg
        perp = np.array([-direction[1], direction[0]], np.float32) * c.staff_offset
        vel = direction * c.staff_speed
        rows = []
        for i in range(c.staff_num):
            off = perp * (i - (c.staff_num - 1) / 2.0)
            rows.append([self.player_pos[0] + off[0], self.player_pos[1] + off[1], vel[0], vel[1], c.staff_life, self._rng.uniform(*c.staff_dmg)])
        self.player_bullets = self._append(self.player_bullets, rows)

    def _cast_spell(self, direction):
        c = self.cfg
        base = np.arctan2(direction[1], direction[0])
        angles = base + np.linspace(-np.radians(c.spell_arc_deg) / 2, np.radians(c.spell_arc_deg) / 2, c.spell_num)
        vel = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32) * c.spell_speed
        dmg = self._rng.uniform(*c.spell_dmg, size=(c.spell_num, 1)).astype(np.float32)
        pos = np.tile(self.player_pos, (c.spell_num, 1)).astype(np.float32)
        life = np.full((c.spell_num, 1), c.spell_life, np.float32)
        self.player_bullets = self._append(self.player_bullets, np.concatenate([pos, vel, life, dmg], 1))

    # --- boss --------------------------------------------------------------

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
        self.boss_pos = self.boss_pos + self._unit(self.player_pos - self.boss_pos) * c.boss_speed
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
        angles = base_angle + (np.arange(count) - (count - 1) / 2.0) * gap
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
        alive = self.snakes[self.snakes[:, EHP] > 0]
        if alive.shape[0]:
            self._scatter(grid, CH_ENEMY, alive[:, :2] - p, 0.6)
        boss_visible = self.fight_active and np.linalg.norm(self.boss_pos - p) <= VIS_RADIUS
        if boss_visible:
            self._scatter(grid, CH_ENEMY, (self.boss_pos - p).reshape(1, 2), 1.0)
        if self.enemy_bullets.shape[0]:
            rel = self.enemy_bullets[:, :2] - p
            self._scatter(grid, CH_EBULLET, rel, 1.0)
            u = self.enemy_bullets[:, BVX:BVY + 1] / (np.linalg.norm(self.enemy_bullets[:, BVX:BVY + 1], axis=1, keepdims=True) + 1e-6)
            self._scatter(grid, CH_EBVX, rel, u[:, 0])
            self._scatter(grid, CH_EBVY, rel, u[:, 1])
        if self.player_bullets.shape[0]:
            self._scatter(grid, CH_PBULLET, self.player_bullets[:, :2] - p, 1.0)
        scalars = np.array(
            [
                self.player_hp / c.player_hp_max,
                self.player_mp / c.player_mp_max,
                1.0 if (self.player_mp >= c.spell_cost and self.spell_timer == 0) else 0.0,
                1.0 if boss_visible else 0.0,
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

    def render(self, px_per_tile: int = 5):
        if self.render_mode != "rgb_array":
            return None
        img = np.where(self.map.walkable[..., None], np.array([40, 40, 48], np.uint8), np.array([12, 12, 16], np.uint8))
        img = np.repeat(np.repeat(img, px_per_tile, 0), px_per_tile, 1)

        def dot(pos, color, r=2):
            cx, cy = int(pos[0] * px_per_tile), int(pos[1] * px_per_tile)
            y0, y1 = max(0, cy - r), min(img.shape[0], cy + r + 1)
            x0, x1 = max(0, cx - r), min(img.shape[1], cx + r + 1)
            if y0 < y1 and x0 < x1:
                img[y0:y1, x0:x1] = color

        for s in self.snakes[self.snakes[:, EHP] > 0]:
            dot(s[:2], (180, 120, 40), 2)
        for b in self.enemy_bullets:
            dot(b[:2], (255, 140, 0), 1)
        for b in self.player_bullets:
            dot(b[:2], (90, 200, 255), 1)
        if self.fight_active:
            dot(self.boss_pos, (220, 40, 40), 4)
        dot(self.player_pos, (60, 230, 90), 3)
        return img
