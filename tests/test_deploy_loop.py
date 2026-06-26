"""The intent<->action mapping round-trips, and the deploy loop runs end to end."""

import numpy as np
import pytest

from rotmg_rl.deploy.realm_state import action_to_intent, intent_to_action


def test_intent_action_roundtrip_all_actions():
    for move in range(9):
        for aim in range(9):
            assert intent_to_action(action_to_intent([move, aim])) == [move, aim]


def test_deploy_loop_runs_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    from rotmg_rl.policy import Agent
    from rotmg_rl.deploy.policy_server import PolicyRunner
    from rotmg_rl.deploy.loop import run_episode
    from rotmg_rl.sim.snakepit import SnakePitConfig

    ckpt = str(tmp_path / "agent.pt")
    torch.save(Agent().state_dict(), ckpt)
    runner = PolicyRunner(ckpt, device="cpu")
    # Untrained agent won't clear; we only assert the full bridge loop runs and terminates.
    result = run_episode(runner, SnakePitConfig(boss_hp_max=20.0, max_steps=200), seed=0)
    assert isinstance(result, bool)
