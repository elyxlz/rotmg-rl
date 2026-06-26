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

CH_WALL, CH_ENEMY, CH_EBULLET, CH_EBVX, CH_EBVY, CH_PBULLET, CH_GRENADE = range(7)
NUM_CH = 7
NUM_SCALARS = 6  # hp, mp, spell_ready, boss_visible, confused, petrified
BX, BY, BVX, BVY, BLIFE, BDMG = range(6)  # bullet columns
EX, EY, EHP, ETIMER = range(4)  # enemy columns
GX, GY, GFUSE, GRAD, GDMG, GSTATUS = range(6)  # grenade columns (status: 0 confused, 1 petrify)


@dataclass
class DungeonConfig:
    player_speed: float = 0.55
    player_radius: float = 0.4
    max_steps: int = 4000
    activation_range: float = 20.0
    spawn_in_room_prob: float = 0.0  # curriculum: prob of spawning near the boss (practice the fight)
    random_spawn_prob: float = 0.0  # spawn at a random walkable tile anywhere (coverage, less overfitting)
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
    boss_shoots: bool = True  # curriculum stage 0 sets False: passive target, learn aim+kill first
    invuln_ticks: int = 15
    ebullet_speed: float = 0.7
    ebullet_life: int = 15
    ebullet_dmg: float = 40.0
    ebullet_radius: float = 0.4
    max_bullets: int = 8192
    # grenades (telegraphed AoE -> status): source r3.5 dmg150 Confused (P1/P2), r1.5 dmg75 Petrify (P3)
    grenade_fuse: int = 10  # ticks telegraphed before detonation (~1s)
    grenade_cd_p1: int = 15
    grenade_cd_p2: int = 10
    grenade_radius_confuse: float = 3.5
    grenade_dmg_confuse: float = 150.0
    grenade_radius_petrify: float = 1.5
    grenade_dmg_petrify: float = 75.0
    confused_ticks: int = 10
    petrify_ticks: int = 10
    # Stheno Swarm minions (Reproduce up to 5 every 1.5s in P1)
    minion_max: int = 5
    minion_cd: int = 15
    minion_hp: float = 30.0
    enable_grenades: bool = True  # curriculum can disable for early stages
    enable_minions: bool = True
    # rewards (exploration-based, no global pathfinding). Scaled to PufferLib's roughly -1..1 rule:
    # per-step signals tiny, a full clean clear totals ~1-2.5 total episode reward.
    rew_explore: float = 0.01  # per newly-visited tile
    rew_kill: float = 0.1  # per snake killed
    rew_boss_dmg: float = 1.0  # dominant signal, applied normalized by boss_hp_max (full boss ~= 1.0)
    rew_reach: float = 0.3
    rew_survive: float = 0.0  # NO reward for existing (paid the agent to flee -> cleared fell)
    rew_damage_taken: float = 0.5  # applied normalized by player_hp_max (full HP lost == 0.5)
    rew_clear: float = 1.0
    rew_death: float = 0.5  # small: don't make it terrified to engage the boss
    rew_step: float = -0.001  # net-negative existence: must make progress (kill the boss)


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
        roll = self._rng.random()
        if roll < c.random_spawn_prob:  # random restart anywhere in the dungeon (coverage)
            i = int(self._rng.integers(len(self._walk_xs)))
            self.player_pos = np.array([self._walk_xs[i] + 0.5, self._walk_ys[i] + 0.5], np.float32)
        elif roll < c.random_spawn_prob + c.spawn_in_room_prob:  # near the boss (fight practice)
            bx, by = self.boss_xy
            ang = self._rng.uniform(0, 2 * np.pi)
            sx, sy = _nearest_walkable(self.map.walkable, int(np.clip(bx + 6 * np.cos(ang), 1, self.map.width - 2)), int(np.clip(by + 6 * np.sin(ang), 1, self.map.height - 2)))
            self.player_pos = np.array([sx + 0.5, sy + 0.5], np.float32)
        else:
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
        self.grenades = np.zeros((0, 6), np.float32)
        self.confused_timer = 0
        self.petrify_timer = 0
        self.minion_timer = 0
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

        if move_idx > 0 and self.petrify_timer == 0:
            mv = MOVE_DIRS[move_idx - 1] * c.player_speed
            if self.confused_timer > 0:
                mv = -mv  # Confused reverses movement
            self._try_move(mv)
        self.confused_timer = max(0, self.confused_timer - 1)
        self.petrify_timer = max(0, self.petrify_timer - 1)

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
        reward += self._grenades_tick()

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
                reward += (dmg / c.boss_hp_max) * c.rew_boss_dmg
                self.player_bullets = self.player_bullets[~hit]
        # enemy bullets vs player
        if self.enemy_bullets.shape[0]:
            hit = np.linalg.norm(self.enemy_bullets[:, :2] - self.player_pos, axis=1) < (c.player_radius + c.ebullet_radius)
            if hit.any():
                dmg = float(self.enemy_bullets[hit, BDMG].sum())
                self.player_hp -= dmg
                reward -= (dmg / c.player_hp_max) * c.rew_damage_taken
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
            if c.boss_shoots:
                self._aimed_shoot("p1", 3, 15, 15)
            self._spawn_minions()
            self._throw_grenade("g1", c.grenade_cd_p1, c.grenade_radius_confuse, c.grenade_dmg_confuse, 0)
        elif self.phase == 2:
            if c.boss_shoots:
                self._rotating_shoot("p2", 4, 15, 3)
            self._throw_grenade("g2", c.grenade_cd_p2, c.grenade_radius_confuse, c.grenade_dmg_confuse, 0)
        elif self.phase == 3:
            if c.boss_shoots:
                self._aimed_shoot("p3a", 3, 15, 15)
                self._rotating_shoot("p3b", 4, 15, 5)
            self._throw_grenade("g3", c.grenade_cd_p1, c.grenade_radius_petrify, c.grenade_dmg_petrify, 1)

    def _spawn_minions(self):
        c = self.cfg
        if not c.enable_minions:
            return
        if self.minion_timer > 0:
            self.minion_timer -= 1
            return
        self.minion_timer = c.minion_cd
        if int((self.snakes[:, EHP] > 0).sum()) >= self.cfg.n_snakes + c.minion_max:
            return
        ang = self._rng.uniform(0, 2 * np.pi, size=c.minion_max)
        pos = self.boss_pos + np.stack([np.cos(ang), np.sin(ang)], 1) * 3.0
        new = np.zeros((c.minion_max, 4), np.float32)
        new[:, :2] = pos
        new[:, EHP] = c.minion_hp
        self.snakes = np.concatenate([self.snakes, new], 0)

    def _throw_grenade(self, key, cooldown, radius, dmg, status):
        if not self.cfg.enable_grenades:
            return
        if self.shoot_timers.get(key, 0) > 0:
            self.shoot_timers[key] -= 1
            return
        self.shoot_timers[key] = cooldown
        # telegraphed at the player's current location
        g = [[self.player_pos[0], self.player_pos[1], self.cfg.grenade_fuse, radius, dmg, status]]
        self.grenades = np.concatenate([self.grenades, np.asarray(g, np.float32)], 0) if self.grenades.shape[0] else np.asarray(g, np.float32)

    def _grenades_tick(self) -> float:
        if self.grenades.shape[0] == 0:
            return 0.0
        c = self.cfg
        reward = 0.0
        self.grenades[:, GFUSE] -= 1.0
        detonated = self.grenades[:, GFUSE] <= 0
        for g in self.grenades[detonated]:
            if np.linalg.norm(self.player_pos - g[:2]) <= g[GRAD]:
                self.player_hp -= g[GDMG]
                reward -= (g[GDMG] / c.player_hp_max) * c.rew_damage_taken
                if int(g[GSTATUS]) == 0:
                    self.confused_timer = c.confused_ticks
                else:
                    self.petrify_timer = c.petrify_ticks
        self.grenades = self.grenades[~detonated]
        return reward

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
        if self.grenades.shape[0]:
            # telegraph: intensity rises as the fuse runs out, so the agent learns to flee
            urgency = np.clip(1.0 - self.grenades[:, GFUSE] / max(c.grenade_fuse, 1), 0.2, 1.0)
            self._scatter(grid, CH_GRENADE, self.grenades[:, :2] - p, urgency.astype(np.float32))
        scalars = np.array(
            [
                self.player_hp / c.player_hp_max,
                self.player_mp / c.player_mp_max,
                1.0 if (self.player_mp >= c.spell_cost and self.spell_timer == 0) else 0.0,
                1.0 if boss_visible else 0.0,
                1.0 if self.confused_timer > 0 else 0.0,
                1.0 if self.petrify_timer > 0 else 0.0,
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

    def render(self, size: int = 480, pov_radius: float = 16.0):
        """Player POV (camera follows the player, like the real ROTMG client) + minimap top-right."""
        if self.render_mode != "rgb_array":
            return None
        h, w = self.map.walkable.shape
        ppt = size / (2 * pov_radius)  # screen px per world tile
        center = size / 2.0
        p = self.player_pos

        yy, xx = np.mgrid[0:size, 0:size]
        tx = np.floor(p[0] + (xx - center) / ppt).astype(int)
        ty = np.floor(p[1] + (yy - center) / ppt).astype(int)
        inb = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        floor = np.zeros((size, size), bool)
        floor[inb] = self.map.walkable[np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)][inb]
        img = np.where(floor[..., None], np.array([58, 50, 42], np.uint8), np.array([16, 14, 20], np.uint8))

        def dot(world, color, r):
            sx, sy = int((world[0] - p[0]) * ppt + center), int((world[1] - p[1]) * ppt + center)
            y0, y1, x0, x1 = max(0, sy - r), min(size, sy + r + 1), max(0, sx - r), min(size, sx + r + 1)
            if y0 < y1 and x0 < x1:
                img[y0:y1, x0:x1] = color

        for g in self.grenades:
            dot(g[:2], (200, 60, 200), max(2, int(g[GRAD] * ppt)))
        for s in self.snakes[self.snakes[:, EHP] > 0]:
            dot(s[:2], (200, 150, 50), max(2, int(0.6 * ppt)))
        for b in self.enemy_bullets:
            dot(b[:2], (255, 140, 0), max(1, int(0.3 * ppt)))
        for b in self.player_bullets:
            dot(b[:2], (120, 210, 255), max(1, int(0.25 * ppt)))
        if np.linalg.norm(self.boss_pos - p) <= pov_radius + self.cfg.boss_radius:
            dot(self.boss_pos, (230, 50, 50), int(self.cfg.boss_radius * ppt))
        dot(p, (70, 240, 110), max(2, int(0.6 * ppt)))  # player at screen center

        # minimap (whole dungeon) top-right
        mm = 120
        mini = np.where(self.map.walkable[..., None], np.array([70, 64, 58], np.uint8), np.array([24, 22, 28], np.uint8))
        mini = mini[np.clip(np.arange(mm) * h // mm, 0, h - 1)][:, np.clip(np.arange(mm) * w // mm, 0, w - 1)]
        for (wx, wy, col) in [(self.boss_xy[0], self.boss_xy[1], (230, 50, 50)), (int(p[0]), int(p[1]), (70, 240, 110))]:
            mx, my = int(wx * mm / w), int(wy * mm / h)
            mini[max(0, my - 2):my + 3, max(0, mx - 2):mx + 3] = col
        img[6:6 + mm, size - mm - 6:size - 6] = mini

        # HP (green) / MP (blue) bars, bottom-left
        c = self.cfg
        for j, (frac, col) in enumerate([(self.player_hp / c.player_hp_max, (60, 220, 60)), (self.player_mp / c.player_mp_max, (60, 120, 255))]):
            yb = size - 18 + j * 8
            img[yb:yb + 6, 8:8 + int(max(0, frac) * 160)] = col
        return img
