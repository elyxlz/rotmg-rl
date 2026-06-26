"""M6: the policy inference server maps RealmState -> action intent and carries LSTM state."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # needs the `train` extra; skipped in the base env

from rotmg_rl.deploy.policy_server import PolicyRunner  # noqa: E402
from rotmg_rl.policy import Agent  # noqa: E402


def _realm_dict(now: float, boss_hp: float = 150.0) -> dict:
    return {
        "arena_size": 40.0,
        "player_pos": [20.0, 33.0],
        "player_hp": 80.0,
        "player_hp_max": 100.0,
        "boss_pos": [20.0, 9.0],
        "boss_hp": boss_hp,
        "boss_hp_max": 250.0,
        "now": now,
        "enemy_shoots": [
            {"origin": [20.0, 9.0], "base_angle": 1.0, "count": 8, "arc_gap": 0.39, "speed": 0.7, "spawn_time": now - 3, "lifetime": 60.0}
        ],
    }


def test_runner_returns_valid_intents_and_keeps_state(tmp_path):
    ckpt = str(tmp_path / "agent.pt")
    torch.save(Agent().state_dict(), ckpt)
    runner = PolicyRunner(ckpt, device="cpu")

    state_before = runner.lstm_state[0].clone()
    for tick in range(5):
        intent = runner.step(_realm_dict(now=float(tick)))
        assert len(intent["move"]) == 2 and len(intent["aim"]) == 2
        assert isinstance(intent["shoot"], bool)
        assert all(np.isfinite(intent["move"])) and all(np.isfinite(intent["aim"]))

    # LSTM state advanced across ticks, and reset() restores the zero state.
    assert not torch.equal(runner.lstm_state[0], state_before)
    runner.reset()
    assert torch.equal(runner.lstm_state[0], state_before)
