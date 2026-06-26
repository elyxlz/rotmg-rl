"""Adapter: RealmShark `EnemyShoot` packets -> our capture / EnemyShootEvent.

Real packet fields (RealmShark `EnemyShootPacket`): startingPos (origin), angle (first bullet),
numShots, angleInc (fan step), ownerId, bulletType, damage, time. Bullets fan as
`angle + i*angleInc` for i in 0..numShots-1. Speed/lifetime are NOT in the packet -- they come
from the projectile asset (`bulletType` -> Objects.xml). Supply a bulletType->speed table when
known; otherwise a default is used (and flagged for the gap report to refine).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotmg_rl.deploy.realm_state import EnemyShootEvent

DEFAULT_BULLET_SPEED = 0.7  # placeholder until the bulletType->speed asset table is supplied
DEFAULT_BULLET_LIFETIME = 60.0
MS_PER_TICK = 200.0  # ROTMG server tick ~5/s; refined from real capture timing in the gap pass


@dataclass
class RealEnemyShoot:
    """One RealmShark EnemyShoot packet (the fields we use)."""

    x: float
    y: float
    angle: float
    num_shots: int
    angle_inc: float
    time_ms: float
    bullet_type: int = 0
    owner_id: int = 0


def to_event(p: RealEnemyShoot, speed_table: dict[int, float] | None = None, ms_per_tick: float = MS_PER_TICK) -> EnemyShootEvent:
    count = max(1, p.num_shots)
    # Convert real (first-bullet angle + i*inc) to our centered (base + (i-(n-1)/2)*gap) form,
    # so reconstruct_bullets reproduces the real fan exactly.
    base_angle = p.angle + (count - 1) / 2.0 * p.angle_inc
    speed = (speed_table or {}).get(p.bullet_type, DEFAULT_BULLET_SPEED)
    return EnemyShootEvent(
        origin=np.array([p.x, p.y], np.float32),
        base_angle=float(base_angle),
        count=count,
        arc_gap=float(p.angle_inc),
        speed=float(speed),
        spawn_time=p.time_ms / ms_per_tick,
        lifetime=DEFAULT_BULLET_LIFETIME,
    )
