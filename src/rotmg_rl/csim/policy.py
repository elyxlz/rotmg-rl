"""CNN-LSTM policy for the C dungeon env (flat float32 obs).

Mirrors rotmg_rl.puffer_policy.DungeonPolicy but reads the native flat Box obs (no emulation):
the first NUM_CH*GRID*GRID floats are the grid, the last NUM_SCALARS are the scalars.
"""

from __future__ import annotations

import pufferlib.pytorch
import pufferlib.spaces
import torch
from torch import nn

from rotmg_rl.sim.dungeon import GRID, MM, NUM_CH, NUM_MM_CH, NUM_SCALARS

_li = pufferlib.pytorch.layer_init
_GRID_FLAT = NUM_CH * GRID * GRID
_MM_FLAT = NUM_MM_CH * MM * MM


class CDungeonPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        assert isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        self.is_multidiscrete = True
        self.is_continuous = False
        self.action_nvec = tuple(env.single_action_space.nvec)
        self.cnn = nn.Sequential(
            _li(nn.Conv2d(NUM_CH, 32, 3, padding=1)),
            nn.GELU(),
            _li(nn.Conv2d(32, 32, 3, padding=1)),
            nn.GELU(),
            nn.Flatten(),
        )
        self.grid_fc = nn.Sequential(_li(nn.Linear(32 * GRID * GRID, 256)), nn.GELU())
        # global fog-of-war minimap: a shallow CNN giving the policy navigation context (where it is,
        # where the boss is once seen) that the local 31x31 window cannot carry.
        self.mm_cnn = nn.Sequential(
            _li(nn.Conv2d(NUM_MM_CH, 16, 3, padding=1)),
            nn.GELU(),
            _li(nn.Conv2d(16, 16, 3, padding=1)),
            nn.GELU(),
            nn.Flatten(),
        )
        self.mm_fc = nn.Sequential(_li(nn.Linear(16 * MM * MM, 128)), nn.GELU())
        self.scalar_fc = nn.Sequential(_li(nn.Linear(NUM_SCALARS, 64)), nn.GELU())
        self.fuse = nn.Sequential(_li(nn.Linear(256 + 128 + 64, hidden_size)), nn.GELU())
        self.decoder = _li(nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01)
        self.value = _li(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(self, observations, state=None):
        b = observations.shape[0]
        obs = observations.float()
        grid = obs[:, :_GRID_FLAT].view(b, NUM_CH, GRID, GRID)
        minimap = obs[:, _GRID_FLAT : _GRID_FLAT + _MM_FLAT].view(b, NUM_MM_CH, MM, MM)
        scalars = obs[:, _GRID_FLAT + _MM_FLAT :]
        g = self.grid_fc(self.cnn(grid))
        m = self.mm_fc(self.mm_cnn(minimap))
        s = self.scalar_fc(scalars)
        return self.fuse(torch.cat([g, m, s], dim=1))

    def decode_actions(self, hidden):
        logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        return logits, self.value(hidden)

    def forward_eval(self, observations, state=None):
        return self.decode_actions(self.encode_observations(observations, state=state))

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state=state)
