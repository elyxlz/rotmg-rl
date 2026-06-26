"""Snake Pit: numpy reference simulator (correctness-first; the C port comes later).

A small square arena with one boss (Stheno) that fires bullet patterns at the player.
The player moves in 8 directions and shoots in 8 directions. Goal: kill the boss without
dying. Reward is densely shaped so a cold-start policy gets gradient before its first clear.

This env is the ground-truth adapter: its `game_state()` builds the same `GameState` the
real-game packet adapter will, so `build_observation` yields an identical tensor in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from rotmg_rl.observation import GRID_SHAPE, NUM_SCALARS, GameState, build_observation

# 8 unit directions, index 0..7, evenly spaced starting East going counter-clockwise.
_ANGLES = np.arange(8, dtype=np.float32) * (np.pi / 4.0)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1).astype(np.float32)  # (8,2)


@dataclass
class SnakePitConfig:
    arena_size: float = 40.0
    dt: float = 1.0
    max_steps: int = 1200
    player_speed: float = 0.9
    player_hp_max: float = 100.0
    player_radius: float = 0.6
    # DPS tuned so the full boss is killable within max_steps with margin for dodging:
    # ~half the episode spent shooting -> ~(600*0.5/cooldown)*dmg*hit_rate damage capacity.
    shoot_cooldown: int = 2
    player_bullet_speed: float = 1.6
    player_bullet_dmg: float = 2.0
    player_bullet_radius: float = 0.5
    boss_hp_max: float = 250.0  # full Snake Pit boss; feasible (not 600, which exceeds the horizon)
    boss_radius: float = 2.0
    boss_speed: float = 0.25
    boss_fire_interval: int = 18
    boss_burst: int = 12  # bullets per radial burst
    enemy_bullet_speed: float = 0.7
    enemy_bullet_dmg: float = 8.0
    enemy_bullet_radius: float = 0.5
    max_bullets: int = 4096
    # Reward shaping.
    rew_boss_dmg: float = 1.0  # per HP of boss damage
    rew_survive: float = 0.002  # per tick alive
    rew_damage_taken: float = 0.05  # per HP lost
    rew_clear: float = 50.0
    rew_death: float = 10.0
    # Domain randomization (per-episode), for sim-to-real robustness (M4). Multiplicative
    # ranges around the base values, plus additive obs noise and spawn jitter.
    randomize: bool = False
    dr_bullet_speed: tuple[float, float] = (0.9, 1.1)
    dr_boss_speed: tuple[float, float] = (0.85, 1.15)
    dr_player_speed: tuple[float, float] = (0.92, 1.08)
    dr_fire_interval: tuple[int, int] = (-1, 1)  # additive to boss_fire_interval, clamped >= 8
    dr_spawn_jitter: float = 0.08  # fraction of arena_size
    dr_obs_noise: float = 0.005  # gaussian std added to the observation tensor when randomizing


class SnakePitEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: SnakePitConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = config or SnakePitConfig()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Dict(
            {
                # Velocity channels (CH_ENEMY_BULLET_VX/VY) are signed unit components.
                "grid": gym.spaces.Box(-1.0, 1.0, GRID_SHAPE, np.float32),
                "scalars": gym.spaces.Box(-1.0, 1.0, (NUM_SCALARS,), np.float32),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([9, 9])  # move(0=stay,1-8), aim(0=none,1-8)
        self._rng = np.random.default_rng()

    # --- core loop ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        self.steps = 0
        self.shoot_timer = 0
        # Per-episode effective dynamics (randomized for M4 robustness, else the base values).
        rng = self._rng
        if c.randomize:
            self.eff_bullet_speed = c.enemy_bullet_speed * rng.uniform(*c.dr_bullet_speed)
            self.eff_boss_speed = c.boss_speed * rng.uniform(*c.dr_boss_speed)
            self.eff_player_speed = c.player_speed * rng.uniform(*c.dr_player_speed)
            self.eff_fire_interval = max(8, c.boss_fire_interval + int(rng.integers(c.dr_fire_interval[0], c.dr_fire_interval[1] + 1)))
            self.eff_obs_noise = c.dr_obs_noise
            jit = c.dr_spawn_jitter * c.arena_size
            px = c.arena_size * 0.5 + rng.uniform(-jit, jit)
            bx = c.arena_size * 0.5 + rng.uniform(-jit, jit)
        else:
            self.eff_bullet_speed = c.enemy_bullet_speed
            self.eff_boss_speed = c.boss_speed
            self.eff_player_speed = c.player_speed
            self.eff_fire_interval = c.boss_fire_interval
            self.eff_obs_noise = 0.0
            px = bx = c.arena_size * 0.5
        self.boss_timer = self.eff_fire_interval
        self.player_pos = np.array([px, c.arena_size * 0.85], np.float32)
        self.player_hp = c.player_hp_max
        self.boss_pos = np.array([bx, c.arena_size * 0.2], np.float32)
        self.boss_hp = c.boss_hp_max
        self.enemy_bullets = np.zeros((0, 4), np.float32)
        self.player_bullets = np.zeros((0, 4), np.float32)
        # Per-step log of bursts fired (the sim's analog of EnemyShoot packets), for captures.
        self.shoot_log: list[tuple] = []
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        move_idx, aim_idx = int(action[0]), int(action[1])
        reward = 0.0

        # Player movement.
        if move_idx > 0:
            self.player_pos = self.player_pos + DIRS[move_idx - 1] * self.eff_player_speed * c.dt
            self.player_pos = np.clip(self.player_pos, 0.0, c.arena_size)

        # Player shooting.
        self.shoot_timer = max(0, self.shoot_timer - 1)
        if aim_idx > 0 and self.shoot_timer == 0:
            vel = DIRS[aim_idx - 1] * c.player_bullet_speed
            bullet = np.array([[self.player_pos[0], self.player_pos[1], vel[0], vel[1]]], np.float32)
            self.player_bullets = self._append(self.player_bullets, bullet)
            self.shoot_timer = c.shoot_cooldown

        # Boss behaviour: drift toward player, fire radial bursts.
        to_player = self.player_pos - self.boss_pos
        d = np.linalg.norm(to_player) + 1e-6
        self.boss_pos = np.clip(self.boss_pos + (to_player / d) * self.eff_boss_speed * c.dt, 0.0, c.arena_size)
        self.boss_timer -= 1
        if self.boss_timer <= 0:
            self.boss_timer = self.eff_fire_interval
            self.enemy_bullets = self._append(self.enemy_bullets, self._radial_burst())

        # Advance bullets.
        self.player_bullets = self._advance(self.player_bullets)
        self.enemy_bullets = self._advance(self.enemy_bullets)

        # Player bullets vs boss.
        if self.player_bullets.shape[0] > 0:
            hit = np.linalg.norm(self.player_bullets[:, :2] - self.boss_pos, axis=1) < (c.boss_radius + c.player_bullet_radius)
            n_hit = int(hit.sum())
            if n_hit:
                self.boss_hp -= n_hit * c.player_bullet_dmg
                reward += n_hit * c.player_bullet_dmg * c.rew_boss_dmg
                self.player_bullets = self.player_bullets[~hit]

        # Enemy bullets vs player.
        if self.enemy_bullets.shape[0] > 0:
            hit = np.linalg.norm(self.enemy_bullets[:, :2] - self.player_pos, axis=1) < (c.player_radius + c.enemy_bullet_radius)
            n_hit = int(hit.sum())
            if n_hit:
                self.player_hp -= n_hit * c.enemy_bullet_dmg
                reward -= n_hit * c.enemy_bullet_dmg * c.rew_damage_taken
                self.enemy_bullets = self.enemy_bullets[~hit]

        reward += c.rew_survive
        self.steps += 1

        terminated = False
        cleared = False
        if self.boss_hp <= 0.0:
            terminated, cleared = True, True
            reward += c.rew_clear
        elif self.player_hp <= 0.0:
            terminated = True
            reward -= c.rew_death
        truncated = (not terminated) and self.steps >= c.max_steps

        info = {"cleared": cleared, "boss_hp_frac": max(self.boss_hp, 0.0) / c.boss_hp_max, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    # --- helpers -----------------------------------------------------------

    def _radial_burst(self) -> np.ndarray:
        c = self.cfg
        gap = 2 * np.pi / c.boss_burst
        phase = float(self._rng.uniform(0.0, gap))  # random per-burst phase
        angles = np.arange(c.boss_burst, dtype=np.float32) * gap + phase
        vel = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32) * self.eff_bullet_speed
        pos = np.tile(self.boss_pos, (c.boss_burst, 1)).astype(np.float32)
        # Log the burst as an EnemyShoot-style event (base_angle centers the arc for reconstruction).
        base_angle = phase + (c.boss_burst - 1) / 2.0 * gap
        lifetime = (c.arena_size * 1.5) / max(self.eff_bullet_speed, 1e-6)
        self.shoot_log.append(
            (self.boss_pos.copy(), base_angle, c.boss_burst, gap, float(self.eff_bullet_speed), self.steps, lifetime)
        )
        return np.concatenate([pos, vel], axis=1)

    def _advance(self, bullets: np.ndarray) -> np.ndarray:
        if bullets.shape[0] == 0:
            return bullets
        bullets = bullets.copy()
        bullets[:, :2] += bullets[:, 2:4] * self.cfg.dt
        inside = (
            (bullets[:, 0] >= 0) & (bullets[:, 0] <= self.cfg.arena_size) & (bullets[:, 1] >= 0) & (bullets[:, 1] <= self.cfg.arena_size)
        )
        return bullets[inside]

    def _append(self, bullets: np.ndarray, new: np.ndarray) -> np.ndarray:
        out = np.concatenate([bullets, new], axis=0) if bullets.shape[0] else new
        if out.shape[0] > self.cfg.max_bullets:
            out = out[-self.cfg.max_bullets :]
        return out

    def game_state(self) -> GameState:
        c = self.cfg
        return GameState(
            arena_size=c.arena_size,
            player_pos=self.player_pos,
            player_hp=self.player_hp,
            player_hp_max=c.player_hp_max,
            player_mp=0.0,
            player_mp_max=1.0,
            ability_ready=False,
            boss_pos=self.boss_pos,
            boss_hp=max(self.boss_hp, 0.0),
            boss_hp_max=c.boss_hp_max,
            enemy_bullets=self.enemy_bullets,
            player_bullets=self.player_bullets,
        )

    def _obs(self) -> dict[str, np.ndarray]:
        obs = build_observation(self.game_state())
        if self.eff_obs_noise > 0.0:
            for key in obs:
                obs[key] = np.clip(obs[key] + self._rng.normal(0.0, self.eff_obs_noise, obs[key].shape).astype(np.float32), -1.0, 1.0)
        return obs

    # --- rendering (headless rgb_array, for video logging) -----------------

    def render(self, px: int = 480):
        if self.render_mode != "rgb_array":
            return None
        c = self.cfg
        frame = np.full((px, px, 3), 18, np.uint8)  # dark background
        scale = px / c.arena_size

        def draw(pos: np.ndarray, radius: float, color: tuple[int, int, int]) -> None:
            cx, cy = int(pos[0] * scale), int(pos[1] * scale)
            r = max(1, int(radius * scale))
            y0, y1 = max(0, cy - r), min(px, cy + r + 1)
            x0, x1 = max(0, cx - r), min(px, cx + r + 1)
            if y0 >= y1 or x0 >= x1:
                return
            ys, xs = np.mgrid[y0:y1, x0:x1]
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
            frame[y0:y1, x0:x1][mask] = color

        for b in self.enemy_bullets:
            draw(b[:2], c.enemy_bullet_radius, (255, 140, 0))
        for b in self.player_bullets:
            draw(b[:2], c.player_bullet_radius, (80, 200, 255))
        draw(self.boss_pos, c.boss_radius, (220, 40, 40))
        draw(self.player_pos, c.player_radius, (60, 230, 90))
        return frame
