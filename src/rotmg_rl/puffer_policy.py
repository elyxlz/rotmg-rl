"""CNN-LSTM policy for the whole-dungeon env, in PufferLib's encode/decode convention.

The default PufferLib policy flattens the obs (losing the spatial grid). This one nativizes the
emulated Dict obs back to {grid (6x15x15), scalars (8)}, runs a CNN over the grid + MLP over the
scalars, and decodes the MultiDiscrete [move, aim] action + value. Wrap with PufferLib's LSTM.
"""

from __future__ import annotations

import pufferlib.pytorch
import pufferlib.spaces
import torch
from torch import nn

from rotmg_rl.sim.dungeon import GRID, NUM_CH, NUM_SCALARS

_li = pufferlib.pytorch.layer_init


class DungeonPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        assert isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        # flags the PufferLib LSTM wrapper / action sampler reads
        self.is_multidiscrete = True
        self.is_continuous = False
        self.is_dict_obs = True
        self.action_nvec = tuple(env.single_action_space.nvec)
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.cnn = nn.Sequential(
            _li(nn.Conv2d(NUM_CH, 32, 3, padding=1)),
            nn.GELU(),
            _li(nn.Conv2d(32, 32, 3, padding=1)),
            nn.GELU(),
            nn.Flatten(),
        )
        self.grid_fc = nn.Sequential(_li(nn.Linear(32 * GRID * GRID, 256)), nn.GELU())
        self.scalar_fc = nn.Sequential(_li(nn.Linear(NUM_SCALARS, 64)), nn.GELU())
        self.fuse = nn.Sequential(_li(nn.Linear(256 + 64, hidden_size)), nn.GELU())
        self.decoder = _li(nn.Linear(hidden_size, sum(self.action_nvec)), std=0.01)
        self.value = _li(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(self, observations, state=None):
        b = observations.shape[0]
        obs = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        grid = obs["grid"].view(b, NUM_CH, GRID, GRID).float()
        scalars = obs["scalars"].view(b, NUM_SCALARS).float()
        g = self.grid_fc(self.cnn(grid))
        s = self.scalar_fc(scalars)
        return self.fuse(torch.cat([g, s], dim=1))

    def decode_actions(self, hidden):
        logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        return logits, self.value(hidden)

    def forward_eval(self, observations, state=None):
        return self.decode_actions(self.encode_observations(observations, state=state))

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state=state)
