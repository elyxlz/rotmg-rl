"""M5/M6: the gap-measurement harness recovers true boss firing params from a capture."""

import numpy as np

from rotmg_rl.deploy.capture import load_capture, save_capture, sim_to_capture
from rotmg_rl.deploy.gap import extract_boss_params, measure_gap
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def _known_env():
    # Survive the whole episode (huge HP, no shooting) so many bursts are captured.
    cfg = SnakePitConfig(
        boss_fire_interval=15,
        boss_burst=10,
        enemy_bullet_speed=0.8,
        boss_hp_max=200.0,
        player_hp_max=100_000.0,
        max_steps=300,
    )
    return SnakePitEnv(cfg), cfg


def test_capture_roundtrip_recovers_params(tmp_path):
    env, cfg = _known_env()
    frames = sim_to_capture(env, actions=lambda e: [0, 0], max_ticks=cfg.max_steps)
    path = str(tmp_path / "cap.jsonl")
    save_capture(frames, path)
    frames2 = load_capture(path)
    assert len(frames2) == len(frames)

    params = extract_boss_params(frames2)
    assert params.n_bursts > 5
    assert params.fire_interval == 15
    assert params.burst_count == 10
    assert np.isclose(params.bullet_speed, 0.8, atol=1e-3)
    assert np.isclose(params.arc_gap, 2 * np.pi / 10, atol=1e-3)
    assert np.isclose(params.arena_size, cfg.arena_size, atol=1e-3)
    assert np.isclose(params.boss_hp_max, 200.0, atol=1e-3)


def test_measure_gap_against_default_and_refit(tmp_path):
    env, cfg = _known_env()
    frames = sim_to_capture(env, actions=lambda e: [0, 0], max_ticks=cfg.max_steps)

    fields, refit = measure_gap(frames, cfg=SnakePitConfig())  # diff vs the DEFAULT sim
    by_name = {f.name: f for f in fields}
    # Default fire interval (18) differs from observed (15) -> a real, nonzero gap is reported.
    assert by_name["boss_fire_interval"].observed == 15
    assert by_name["boss_fire_interval"].rel_error > 0.0
    # Refit config adopts the observed params, ready to feed back into the sim.
    assert refit.boss_fire_interval == 15
    assert refit.boss_burst == 10
    assert np.isclose(refit.enemy_bullet_speed, 0.8, atol=1e-3)
