"""The RealmShark adapter reproduces the real bullet fan (angle + i*angleInc)."""

import numpy as np

from rotmg_rl.deploy.realm_state import reconstruct_bullets
from rotmg_rl.deploy.realmshark import RealEnemyShoot, to_event


def test_adapter_reproduces_real_fan_geometry():
    # Real packet: 3 bullets fanning from angle 0 by pi/6 each -> angles 0, pi/6, 2pi/6.
    p = RealEnemyShoot(x=20.0, y=8.0, angle=0.0, num_shots=3, angle_inc=np.pi / 6, time_ms=1000.0, bullet_type=0)
    ev = to_event(p, ms_per_tick=200.0)
    # spawn_time in ticks: 1000ms / 200ms = 5; reconstruct one tick later (age=1).
    bullets = reconstruct_bullets([ev], now=6.0, arena_size=40.0)
    assert bullets.shape[0] == 3
    speed = 0.7
    for i, expected_angle in enumerate([0.0, np.pi / 6, 2 * np.pi / 6]):
        vx, vy = np.cos(expected_angle) * speed, np.sin(expected_angle) * speed
        # bullets aren't ordered, so match by velocity.
        assert any(np.allclose(b[2:4], [vx, vy], atol=1e-4) for b in bullets)


def test_single_bullet_defaults_to_count_one():
    p = RealEnemyShoot(x=10.0, y=10.0, angle=1.2, num_shots=0, angle_inc=0.0, time_ms=0.0)
    ev = to_event(p)
    assert ev.count == 1
