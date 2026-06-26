"""Policy inference server: the Python side of the headless-client <-> policy bridge.

Loads a trained checkpoint, carries LSTM state across ticks, and maps each incoming RealmState
to an ActionIntent. The live nrelay (Node) client pipes one JSON RealmState per line to stdin
and reads one JSON intent per line from stdout; a `{"reset": true}` line resets the LSTM
between episodes. The policy acts stochastically (deployment-faithful; greedy is brittle).

    echo '{"reset":true}' | uv run --extra train python -m rotmg_rl.deploy.policy_server \
        --checkpoint checkpoints/m3-final.pt
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from rotmg_rl.deploy.realm_state import action_to_intent, realm_state_from_dict, realm_to_observation
from rotmg_rl.policy import Agent


class PolicyRunner:
    def __init__(self, checkpoint: str, device: str | None = None, stochastic: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.agent = Agent().to(self.device)
        self.agent.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.agent.eval()
        self.stochastic = stochastic
        self.reset()

    def reset(self) -> None:
        self.lstm_state = self.agent.initial_state(1, self.device)
        self.done = torch.zeros(1, device=self.device)

    @torch.no_grad()
    def step(self, realm_dict: dict) -> dict:
        rs = realm_state_from_dict(realm_dict)
        obs = realm_to_observation(rs)
        flat = np.concatenate([obs["grid"].ravel(), obs["scalars"]]).astype(np.float32)
        x = torch.tensor(flat, device=self.device).unsqueeze(0)
        act = self.agent.act_sample if self.stochastic else self.agent.act_greedy
        action, self.lstm_state = act(x, self.lstm_state, self.done)
        intent = action_to_intent(action[0].cpu().numpy())
        return {
            "move": [float(intent.move[0]), float(intent.move[1])],
            "shoot": bool(intent.shoot),
            "aim": [float(intent.aim[0]), float(intent.aim[1])],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--greedy", action="store_true", help="use greedy argmax (default: stochastic)")
    args = parser.parse_args()

    runner = PolicyRunner(args.checkpoint, stochastic=not args.greedy)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if msg.get("reset"):
            runner.reset()
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
        else:
            sys.stdout.write(json.dumps(runner.step(msg)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
