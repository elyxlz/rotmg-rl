"""Recurrent policy: CNN over the egocentric grid + MLP over scalars -> fuse -> LSTM -> heads.

Observation is the flat (1356,) vector PufferLib produces from our Dict space: the first 1350
values are the (6,15,15) grid, the last 6 are the scalars (verified on box). MultiDiscrete
action [move(9), aim(9)] -> two categorical heads with summed logprob/entropy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rotmg_rl.observation import GRID_SHAPE, NUM_SCALARS

GRID_FLAT = int(np.prod(GRID_SHAPE))  # 1350
MOVE_N = 9
AIM_N = 9


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        c, h, w = GRID_SHAPE
        self.hidden = hidden
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.grid_fc = nn.Sequential(layer_init(nn.Linear(32 * h * w, 256)), nn.ReLU())
        self.scalar_fc = nn.Sequential(layer_init(nn.Linear(NUM_SCALARS, 32)), nn.ReLU())
        self.fuse = nn.Sequential(layer_init(nn.Linear(256 + 32, hidden)), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor_move = layer_init(nn.Linear(hidden, MOVE_N), std=0.01)
        self.actor_aim = layer_init(nn.Linear(hidden, AIM_N), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        grid = obs[:, :GRID_FLAT].view(b, *GRID_SHAPE)
        scal = obs[:, GRID_FLAT:]
        g = self.grid_fc(self.cnn(grid))
        s = self.scalar_fc(scal)
        return self.fuse(torch.cat([g, s], dim=1))

    def get_states(self, obs, lstm_state, done):
        hidden = self.encode(obs)
        batch = lstm_state[0].shape[1]
        hidden = hidden.reshape(-1, batch, self.hidden)
        done = done.reshape(-1, batch)
        out = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1]),
            )
            out.append(h)
        return torch.flatten(torch.cat(out), 0, 1), lstm_state

    def get_value(self, obs, lstm_state, done):
        hidden, _ = self.get_states(obs, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, obs, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(obs, lstm_state, done)
        dist_move = Categorical(logits=self.actor_move(hidden))
        dist_aim = Categorical(logits=self.actor_aim(hidden))
        if action is None:
            action = torch.stack([dist_move.sample(), dist_aim.sample()], dim=1)
        a_move, a_aim = action[:, 0], action[:, 1]
        logprob = dist_move.log_prob(a_move) + dist_aim.log_prob(a_aim)
        entropy = dist_move.entropy() + dist_aim.entropy()
        return action, logprob, entropy, self.critic(hidden), lstm_state

    @torch.no_grad()
    def act_greedy(self, obs, lstm_state, done):
        hidden, lstm_state = self.get_states(obs, lstm_state, done)
        a_move = torch.argmax(self.actor_move(hidden), dim=1)
        a_aim = torch.argmax(self.actor_aim(hidden), dim=1)
        return torch.stack([a_move, a_aim], dim=1), lstm_state

    def initial_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch, self.hidden, device=device),
            torch.zeros(1, batch, self.hidden, device=device),
        )
