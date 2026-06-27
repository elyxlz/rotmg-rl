"""Thin single-env handle over the C dungeon binding (rotmg_rl.csim.binding).

The C env (pufferlib/ocean/dungeon/dungeon.h) is the single source of the Snake Pit dynamics. This wraps ONE env
instance for evaluation + POV rendering using only numpy (no pufferlib / torch), so it runs in any
venv that has the compiled binding. `step` returns the flat [grid, minimap, scalars] obs plus the
gym-style (reward, terminated, truncated, info); `render_state` is a read-only snapshot of the live C
entity buffers for the POV renderer (never a re-simulation). `config_to_kwargs` + `OBS_SIZE` live here
(not in csim.dungeon) so this path stays free of the heavy pufferlib import.
"""

from __future__ import annotations

import numpy as np

from rotmg_rl.config import GRID, MM, NUM_CH, NUM_MM_CH, NUM_SCALARS, DungeonConfig
from rotmg_rl.csim import binding

OBS_SIZE = NUM_CH * GRID * GRID + NUM_MM_CH * MM * MM + NUM_SCALARS


def config_to_kwargs(cfg: DungeonConfig) -> dict:
    """Flatten a DungeonConfig into the scalar kwargs the C my_init expects (tuples -> lo/hi)."""
    return dict(
        player_speed=cfg.player_speed,
        player_radius=cfg.player_radius,
        max_steps=cfg.max_steps,
        activation_range=cfg.activation_range,
        spawn_in_room_prob=cfg.spawn_in_room_prob,
        spawn_in_room_radius=cfg.spawn_in_room_radius,
        random_spawn_prob=cfg.random_spawn_prob,
        player_hp_max=cfg.player_hp_max,
        player_mp_max=cfg.player_mp_max,
        player_defense=cfg.player_defense,
        damage_floor=cfg.damage_floor,
        mp_regen=cfg.mp_regen,
        hp_regen=cfg.hp_regen,
        staff_cooldown=cfg.staff_cooldown,
        staff_num=cfg.staff_num,
        staff_dmg_lo=cfg.staff_dmg[0],
        staff_dmg_hi=cfg.staff_dmg[1],
        staff_speed=cfg.staff_speed,
        staff_life=cfg.staff_life,
        staff_radius=cfg.staff_radius,
        staff_offset=cfg.staff_offset,
        spell_cost=cfg.spell_cost,
        spell_cooldown=cfg.spell_cooldown,
        spell_num=cfg.spell_num,
        spell_dmg_lo=cfg.spell_dmg[0],
        spell_dmg_hi=cfg.spell_dmg[1],
        spell_speed=cfg.spell_speed,
        spell_life=cfg.spell_life,
        n_snakes=cfg.n_snakes,
        n_snakes_jitter=cfg.n_snakes_jitter,
        snake_speed=cfg.snake_speed,
        snake_radius=cfg.snake_radius,
        boss_hp_max=cfg.boss_hp_max,
        boss_radius=cfg.boss_radius,
        boss_defense=cfg.boss_defense,
        boss_wander_speed=cfg.boss_wander_speed,
        boss_return_speed=cfg.boss_return_speed,
        boss_shoots=int(cfg.boss_shoots),
        opening_invuln_ticks=cfg.opening_invuln_ticks,
        invuln_ticks=cfg.invuln_ticks,
        blade_cd=cfg.blade_cd,
        blade_radius_p1=cfg.blade_radius_p1,
        blade_radius_p3=cfg.blade_radius_p3,
        ebullet_speed=cfg.ebullet_speed,
        ebullet_life=cfg.ebullet_life,
        ebullet_dmg=cfg.ebullet_dmg,
        ebullet_radius=cfg.ebullet_radius,
        max_bullets=cfg.max_bullets,
        grenade_fuse=cfg.grenade_fuse,
        grenade_cd_p1=cfg.grenade_cd_p1,
        grenade_cd_p2=cfg.grenade_cd_p2,
        grenade_cd_p3_diag=cfg.grenade_cd_p3_diag,
        grenade_range_confuse=cfg.grenade_range_confuse,
        grenade_petrify_dist=cfg.grenade_petrify_dist,
        grenade_radius_confuse=cfg.grenade_radius_confuse,
        grenade_dmg_confuse=cfg.grenade_dmg_confuse,
        grenade_radius_petrify=cfg.grenade_radius_petrify,
        grenade_dmg_petrify=cfg.grenade_dmg_petrify,
        confused_ticks=cfg.confused_ticks,
        petrify_ticks=cfg.petrify_ticks,
        minion_max=cfg.minion_max,
        minion_cd=cfg.minion_cd,
        minion_hp=cfg.minion_hp,
        enable_grenades=int(cfg.enable_grenades),
        enable_minions=int(cfg.enable_minions),
        rew_explore=cfg.rew_explore,
        rew_kill=cfg.rew_kill,
        rew_boss_dmg=cfg.rew_boss_dmg,
        rew_reach=cfg.rew_reach,
        rew_survive=cfg.rew_survive,
        rew_damage_taken=cfg.rew_damage_taken,
        rew_clear=cfg.rew_clear,
        rew_death=cfg.rew_death,
        rew_step=cfg.rew_step,
        rew_approach=cfg.rew_approach,
    )


class CDungeonSingle:
    """One C dungeon env behind a gym-ish surface (numpy-only). The obs buffer is shared with C and
    refilled in place every reset/step. The C env auto-resets in place on episode end; `step` reports
    the just-ended episode's outcome from the env's outcome latch (which survives that auto-reset)."""

    def __init__(self, config: DungeonConfig | None = None, seed: int = 0):
        self.cfg = config or DungeonConfig()
        self.obs = np.zeros(OBS_SIZE, np.float32)
        self._act = np.zeros(4, np.int32)
        self._rew = np.zeros(1, np.float32)
        self._term = np.zeros(1, np.uint8)
        self._trunc = np.zeros(1, np.uint8)
        self.handle = binding.env_init(self.obs, self._act, self._rew, self._term, self._trunc, seed, **config_to_kwargs(self.cfg))
        self._closed = False

    def reset(self, seed: int = 0) -> np.ndarray:
        binding.env_reset(self.handle, seed)
        return self.obs

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._act[:] = action
        binding.env_step(self.handle)
        g = binding.env_get(self.handle)
        terminated = bool(self._term[0])
        ep_done = bool(g["ep_done"])
        truncated = ep_done and not terminated
        cleared = ep_done and bool(g["ep_cleared"])
        boss_hp_frac = g["ep_boss_hp_frac"] if ep_done else max(g["boss_hp"], 0.0) / g["boss_hp_max"]
        info = {"cleared": cleared, "ep_done": ep_done, "boss_hp_frac": float(boss_hp_frac), "steps": int(g["steps"])}
        return self.obs, float(self._rew[0]), terminated, truncated, info

    def render_state(self) -> dict:
        """Read-only snapshot of the live C entity buffers (player/snakes/bullets/grenades/boss/fog)."""
        return binding.env_get(self.handle)

    def get(self) -> dict:
        return binding.env_get(self.handle)

    def put(self, **kwargs) -> None:
        """Inject deterministic state (player pos / fight_active / phase / boss_hp) + refresh obs."""
        binding.env_put(self.handle, **kwargs)

    def close(self) -> None:
        if not self._closed:
            binding.env_close(self.handle)
            self._closed = True
