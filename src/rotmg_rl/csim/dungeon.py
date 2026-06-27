"""PufferEnv wrapper around the C dungeon env (rotmg_rl.csim.binding).

Native Ocean env: flat float32 Box observation laid out as [grid (NUM_CH*GRID*GRID), scalars
(NUM_SCALARS)] then a MultiDiscrete([9, N_AIM, 2, 2]) action, matching the Python oracle. Use
`CDungeonPolicy` (csim.policy) which slices the flat obs back into grid + scalars.
"""

from __future__ import annotations

from dataclasses import asdict

import gymnasium
import numpy as np
import pufferlib

from rotmg_rl.config import N_AIM, DungeonConfig
from rotmg_rl.csim import binding
from rotmg_rl.csim.single import OBS_SIZE

__all__ = ["OBS_SIZE", "CDungeon"]


class CDungeon(pufferlib.PufferEnv):
    def __init__(self, config: DungeonConfig | None = None, num_envs: int = 1, render_mode=None, log_interval: int = 128, buf=None, seed: int = 0):
        self.cfg = config or DungeonConfig()
        self.single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, N_AIM, 2, 2])
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards, self.terminals, self.truncations, num_envs, seed, **asdict(self.cfg))

    @property
    def agents_per_batch(self) -> int:
        # PuffeRL reads this (plural); the native PufferEnv backend only provides the singular alias.
        return self.num_agents

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
