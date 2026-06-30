"""Shared policy reconstruction: rebuild the trained torch policy for eval and deploy.

`build_policy` is the single owner of the network shape (DungeonEncoder + DefaultDecoder + LSTM) and
the obs/action sizes, so the trainer's eval ladder and the deploy runner load identical architectures.
`OBS_SIZE` is derived from the `rotmg_rl.config` constants (the one dynamics source), not re-imported
from any simulator wrapper.
"""

from __future__ import annotations

import pufferlib.models as models  # ty: ignore[unresolved-import]  pufferlib is pip-installed only on the GPU box
import torch  # ty: ignore[unresolved-import]  torch is a GPU-box-only dep, not installed on this CPU dev box

from rotmg_rl.config import GRID, MM, NUM_CH, NUM_MM_CH, NUM_SCALARS

OBS_SIZE = NUM_CH * GRID * GRID + NUM_MM_CH * MM * MM + NUM_SCALARS
ACT_SIZES = [9, 32, 2, 2]  # MultiDiscrete: move, aim, shoot, cast


def build_policy(hidden: int, num_layers: int, device) -> torch.nn.Module:
    """Reconstruct the --slowly torch policy: Policy(DungeonEncoder, DefaultDecoder, LSTM)."""
    encoder = models.DungeonEncoder(OBS_SIZE, hidden)
    decoder = models.DefaultDecoder(ACT_SIZES, hidden)
    network = models.LSTM(hidden, num_layers=num_layers)
    return models.Policy(encoder, decoder, network).to(device)
