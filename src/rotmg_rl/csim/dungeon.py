"""PufferEnv wrapper around the C dungeon env (rotmg_rl.csim.binding).

Native Ocean env: flat float32 Box observation laid out as [grid (NUM_CH*GRID*GRID), scalars
(NUM_SCALARS)] then a MultiDiscrete([9, N_AIM, 2, 2]) action, matching the Python oracle. Use
`CDungeonPolicy` (csim.policy) which slices the flat obs back into grid + scalars.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pufferlib

from rotmg_rl.csim import binding
from rotmg_rl.sim.dungeon import GRID, N_AIM, NUM_CH, NUM_SCALARS, DungeonConfig

OBS_SIZE = NUM_CH * GRID * GRID + NUM_SCALARS


def config_to_kwargs(cfg: DungeonConfig) -> dict:
    """Flatten a DungeonConfig into the scalar kwargs the C my_init expects (tuples -> lo/hi)."""
    return dict(
        player_speed=cfg.player_speed,
        player_radius=cfg.player_radius,
        max_steps=cfg.max_steps,
        activation_range=cfg.activation_range,
        spawn_in_room_prob=cfg.spawn_in_room_prob,
        random_spawn_prob=cfg.random_spawn_prob,
        player_hp_max=cfg.player_hp_max,
        player_mp_max=cfg.player_mp_max,
        mp_regen=cfg.mp_regen,
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
        spell_arc_deg=cfg.spell_arc_deg,
        spell_dmg_lo=cfg.spell_dmg[0],
        spell_dmg_hi=cfg.spell_dmg[1],
        spell_speed=cfg.spell_speed,
        spell_life=cfg.spell_life,
        n_snakes=cfg.n_snakes,
        snake_hp=cfg.snake_hp,
        snake_speed=cfg.snake_speed,
        snake_shoot_range=cfg.snake_shoot_range,
        snake_cooldown=cfg.snake_cooldown,
        snake_bullet_speed=cfg.snake_bullet_speed,
        snake_bullet_life=cfg.snake_bullet_life,
        snake_bullet_dmg=cfg.snake_bullet_dmg,
        snake_radius=cfg.snake_radius,
        boss_hp_max=cfg.boss_hp_max,
        boss_radius=cfg.boss_radius,
        boss_speed=cfg.boss_speed,
        boss_shoots=int(cfg.boss_shoots),
        invuln_ticks=cfg.invuln_ticks,
        ebullet_speed=cfg.ebullet_speed,
        ebullet_life=cfg.ebullet_life,
        ebullet_dmg=cfg.ebullet_dmg,
        ebullet_radius=cfg.ebullet_radius,
        max_bullets=cfg.max_bullets,
        grenade_fuse=cfg.grenade_fuse,
        grenade_cd_p1=cfg.grenade_cd_p1,
        grenade_cd_p2=cfg.grenade_cd_p2,
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
    )


class CDungeon(pufferlib.PufferEnv):
    def __init__(self, config: DungeonConfig | None = None, num_envs: int = 1, render_mode=None, log_interval: int = 128, buf=None, seed: int = 0):
        self.cfg = config or DungeonConfig()
        self.single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, N_AIM, 2, 2])
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards, self.terminals, self.truncations, num_envs, seed, **config_to_kwargs(self.cfg))

    def reset(self, seed: int = 0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        pass

    def close(self):
        binding.vec_close(self.c_envs)
