"""Load + run the deployed recurrent CDungeonPolicy on a flat real-game obs.

The checkpoint is the recurrent (LSTM) wrapper around CDungeonPolicy; the obs is the flat
[grid, minimap, scalars] Box(9807); the LSTM state is a dict carried across steps (in-place
mutated by forward_eval), reset per episode. Action = one sample per MultiDiscrete head.
"""

from __future__ import annotations

import numpy as np
import torch  # ty: ignore[unresolved-import]  torch is a GPU-box-only dep, not installed on this CPU dev box


class PolicyRunner:
    def __init__(self, checkpoint: str, hidden: int = 256, device: str | None = None) -> None:
        from pufferlib.ocean import torch as ocean_torch  # ty: ignore[unresolved-import]  pufferlib is pip-installed only on the GPU box

        from rotmg_rl.csim.dungeon import CDungeon
        from rotmg_rl.csim.policy import CDungeonPolicy

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # The C env is the policy's native env (flat [grid, minimap, scalars] Box obs + the
        # MultiDiscrete action); use it directly as the driver env.
        driver = CDungeon(num_envs=1)
        policy = CDungeonPolicy(driver, hidden_size=hidden)
        policy = ocean_torch.Recurrent(driver, policy, input_size=hidden, hidden_size=hidden).to(self.device)
        policy.load_state_dict(torch.load(checkpoint, map_location=self.device))
        policy.eval()
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.state = {"lstm_h": None, "lstm_c": None, "hidden": None}

    @torch.no_grad()
    def act(self, flat: np.ndarray, greedy: bool = False) -> dict:
        x = torch.tensor(np.asarray(flat, np.float32), device=self.device).unsqueeze(0)
        logits, _ = self.policy.forward_eval(x, self.state)
        a = [int(lg.argmax(dim=1)) if greedy else int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
        return {"move": a[0], "aim": a[1], "shoot": a[2], "cast": a[3]}
