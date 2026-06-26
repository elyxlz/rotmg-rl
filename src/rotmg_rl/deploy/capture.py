"""Capture format for real (or sim) ROTMG sessions, and a sim->capture recorder.

A capture is a JSONL stream of per-tick frames: player/boss state plus any EnemyShoot bursts
observed that tick. A real capture comes from a RealmShark/nrelay dump (a thin format adapter
is written once a real sample is available); a sim capture is produced here for testing the
gap-measurement harness end to end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from rotmg_rl.deploy.realm_state import EnemyShootEvent
from rotmg_rl.sim.snakepit import SnakePitEnv


@dataclass
class ShootRecord:
    origin: list[float]
    base_angle: float
    count: int
    arc_gap: float
    speed: float
    spawn_tick: int
    lifetime: float

    def to_event(self) -> EnemyShootEvent:
        return EnemyShootEvent(
            origin=np.array(self.origin, np.float32),
            base_angle=self.base_angle,
            count=self.count,
            arc_gap=self.arc_gap,
            speed=self.speed,
            spawn_time=float(self.spawn_tick),
            lifetime=self.lifetime,
        )


@dataclass
class CaptureFrame:
    tick: int
    arena_size: float
    player_pos: list[float]
    player_hp: float
    boss_pos: list[float]
    boss_hp: float
    shoots: list[ShootRecord] = field(default_factory=list)


def save_capture(frames: list[CaptureFrame], path: str) -> None:
    with open(path, "w") as f:
        for fr in frames:
            f.write(json.dumps(asdict(fr)) + "\n")


def load_capture(path: str) -> list[CaptureFrame]:
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            shoots = [ShootRecord(**s) for s in d.pop("shoots")]
            frames.append(CaptureFrame(shoots=shoots, **d))
    return frames


def sim_to_capture(env: SnakePitEnv, actions, max_ticks: int = 1200) -> list[CaptureFrame]:
    """Run the sim under a fixed action callable and record a capture (for harness testing)."""
    obs, _ = env.reset(seed=0)
    frames: list[CaptureFrame] = []
    for tick in range(max_ticks):
        before = len(env.shoot_log)
        obs, _, term, trunc, _ = env.step(actions(env))
        new_shoots = [
            ShootRecord(list(map(float, o)), float(ba), int(cnt), float(g), float(sp), int(st), float(lt))
            for (o, ba, cnt, g, sp, st, lt) in env.shoot_log[before:]
        ]
        frames.append(
            CaptureFrame(
                tick=tick,
                arena_size=env.cfg.arena_size,
                player_pos=list(map(float, env.player_pos)),
                player_hp=float(env.player_hp),
                boss_pos=list(map(float, env.boss_pos)),
                boss_hp=float(max(env.boss_hp, 0.0)),
                shoots=new_shoots,
            )
        )
        if term or trunc:
            break
    return frames
