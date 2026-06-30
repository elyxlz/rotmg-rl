"""Load + run the deployed recurrent dungeon policy (PufferLib 4.0) on a flat real-game obs.

Mirrors `rotmg_rl.eval`: the 4.0 policy is `models.Policy(DungeonEncoder, DefaultDecoder, LSTM)` (built
by `eval.build_policy`); the obs is the flat [grid, minimap, scalars] Box(9807); the LSTM state comes
from `policy.initial_state(...)` and is carried across steps (returned by `forward_eval`), reset per
episode. Action = one sample per MultiDiscrete head.
"""

from __future__ import annotations

import numpy as np
import torch  # ty: ignore[unresolved-import]  torch is a GPU-box-only dep, not installed on this CPU dev box

from rotmg_rl.eval import build_policy


class PolicyRunner:
    def __init__(self, checkpoint: str, hidden: int = 256, num_layers: int = 1, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        policy = build_policy(hidden, num_layers, self.device)
        policy.load_state_dict(torch.load(checkpoint, map_location=self.device))
        policy.eval()
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.state = self.policy.initial_state(1, self.device)

    @torch.no_grad()
    def act(self, flat: np.ndarray, greedy: bool = False) -> dict:
        x = torch.tensor(np.asarray(flat, np.float32), device=self.device).unsqueeze(0)
        logits, _, self.state = self.policy.forward_eval(x, self.state)
        a = [int(lg.argmax(dim=1)) if greedy else int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
        # 4-head action space: move, aim, shoot, cast. The staff and the spell SHARE the single
        # aim head (one mouse) -- the BulletNova is cast along the same direction the staff fires,
        # so to drop the spell on a different target the policy turns the aim between ticks.
        return {"move": a[0], "aim": a[1], "shoot": a[2], "cast": a[3]}
