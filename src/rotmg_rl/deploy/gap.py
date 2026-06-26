"""Measure the sim-to-real gap from a capture and suggest sim-config corrections.

Given a capture (real or sim), estimate the boss's observable firing parameters (fire
interval, burst size, arc gap, bullet speed, boss HP, arena size) and diff them against the
sim's SnakePitConfig. The output drives the design's "measure gap on real data -> fix sim ->
retrain" loop: feed the suggested config back into the sim and re-run the curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotmg_rl.deploy.capture import CaptureFrame
from rotmg_rl.sim.snakepit import SnakePitConfig


@dataclass
class BossParams:
    fire_interval: float
    burst_count: float
    arc_gap: float
    bullet_speed: float
    boss_hp_max: float
    arena_size: float
    n_bursts: int


def extract_boss_params(frames: list[CaptureFrame]) -> BossParams:
    shoots = [(fr.tick, s) for fr in frames for s in fr.shoots]
    spawn_ticks = [s.spawn_tick for _, s in shoots]
    intervals = np.diff(sorted(set(spawn_ticks))) if len(set(spawn_ticks)) > 1 else np.array([float("nan")])
    counts = [s.count for _, s in shoots] or [float("nan")]
    gaps = [s.arc_gap for _, s in shoots] or [float("nan")]
    speeds = [s.speed for _, s in shoots] or [float("nan")]
    return BossParams(
        fire_interval=float(np.median(intervals)),
        burst_count=float(np.median(counts)),
        arc_gap=float(np.median(gaps)),
        bullet_speed=float(np.median(speeds)),
        boss_hp_max=float(max((fr.boss_hp for fr in frames), default=float("nan"))),
        arena_size=float(frames[0].arena_size) if frames else float("nan"),
        n_bursts=len(shoots),
    )


@dataclass
class GapField:
    name: str
    sim: float
    observed: float

    @property
    def rel_error(self) -> float:
        return abs(self.observed - self.sim) / max(abs(self.sim), 1e-6)


def measure_gap(frames: list[CaptureFrame], cfg: SnakePitConfig | None = None) -> tuple[list[GapField], SnakePitConfig]:
    """Return per-parameter sim-vs-observed fields and a SnakePitConfig refit to the capture."""
    cfg = cfg or SnakePitConfig()
    p = extract_boss_params(frames)
    fields = [
        GapField("boss_fire_interval", cfg.boss_fire_interval, p.fire_interval),
        GapField("boss_burst", cfg.boss_burst, p.burst_count),
        GapField("enemy_bullet_speed", cfg.enemy_bullet_speed, p.bullet_speed),
        GapField("boss_hp_max", cfg.boss_hp_max, p.boss_hp_max),
        GapField("arena_size", cfg.arena_size, p.arena_size),
    ]
    refit = SnakePitConfig(
        arena_size=p.arena_size if np.isfinite(p.arena_size) else cfg.arena_size,
        boss_fire_interval=int(round(p.fire_interval)) if np.isfinite(p.fire_interval) else cfg.boss_fire_interval,
        boss_burst=int(round(p.burst_count)) if np.isfinite(p.burst_count) else cfg.boss_burst,
        enemy_bullet_speed=p.bullet_speed if np.isfinite(p.bullet_speed) else cfg.enemy_bullet_speed,
        boss_hp_max=p.boss_hp_max if np.isfinite(p.boss_hp_max) else cfg.boss_hp_max,
    )
    return fields, refit


def format_gap(fields: list[GapField]) -> str:
    lines = ["param                  sim        observed   rel_err"]
    for f in fields:
        lines.append(f"{f.name:<22} {f.sim:<10.3f} {f.observed:<10.3f} {f.rel_error:.1%}")
    return "\n".join(lines)
