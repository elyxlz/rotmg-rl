"""Real-game adapter: turn ROTMG packet data into the shared observation, and policy actions
into protocol intents. This is the deploy half of the sim-to-real bridge.

The real client (a headless nrelay fork) does not receive per-frame bullet positions; the
server sends `EnemyShoot` packets describing a burst (origin, base angle, count, arc gap,
speed, spawn time). We reconstruct live bullet positions by locally simulating those bursts
forward (the same technique vrelay's predictive autonexus uses), then build the exact same
`GameState` -> observation the sim produces, so the policy cannot tell sim from real.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rotmg_rl.observation import GameState, build_observation
from rotmg_rl.sim.snakepit import DIRS


@dataclass
class EnemyShootEvent:
    """One boss burst, as carried by an `EnemyShoot` packet."""

    origin: np.ndarray  # (2,) world position the burst spawned from
    base_angle: float  # radians, angle of the burst's center bullet
    count: int  # bullets in the burst
    arc_gap: float  # radians between adjacent bullets
    speed: float  # world units per tick
    spawn_time: float  # tick the burst fired
    lifetime: float  # ticks the bullets live before despawning


@dataclass
class RealmState:
    """World-agnostic snapshot reconstructed from packets at tick `now`."""

    arena_size: float
    player_pos: np.ndarray
    player_hp: float
    player_hp_max: float
    player_mp: float
    player_mp_max: float
    ability_ready: bool
    boss_pos: np.ndarray
    boss_hp: float
    boss_hp_max: float
    now: float
    enemy_shoots: list[EnemyShootEvent] = field(default_factory=list)
    player_bullets: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))


@dataclass
class ActionIntent:
    """What to send to the client: a move direction and an optional shot direction."""

    move: np.ndarray  # (2,) unit vector or zeros
    shoot: bool
    aim: np.ndarray  # (2,) unit vector or zeros


def realm_state_from_dict(d: dict) -> RealmState:
    """Parse a per-tick state dict (as the headless client sends) into a RealmState."""
    shoots = [
        EnemyShootEvent(
            origin=np.array(s["origin"], np.float32),
            base_angle=float(s["base_angle"]),
            count=int(s["count"]),
            arc_gap=float(s["arc_gap"]),
            speed=float(s["speed"]),
            spawn_time=float(s["spawn_time"]),
            lifetime=float(s["lifetime"]),
        )
        for s in d.get("enemy_shoots", [])
    ]
    pb = d.get("player_bullets")
    return RealmState(
        arena_size=float(d["arena_size"]),
        player_pos=np.array(d["player_pos"], np.float32),
        player_hp=float(d["player_hp"]),
        player_hp_max=float(d["player_hp_max"]),
        player_mp=float(d.get("player_mp", 0.0)),
        player_mp_max=float(d.get("player_mp_max", 1.0)),
        ability_ready=bool(d.get("ability_ready", False)),
        boss_pos=np.array(d["boss_pos"], np.float32),
        boss_hp=float(d["boss_hp"]),
        boss_hp_max=float(d["boss_hp_max"]),
        now=float(d["now"]),
        enemy_shoots=shoots,
        player_bullets=np.array(pb, np.float32) if pb else np.zeros((0, 4), np.float32),
    )


def reconstruct_bullets(events: list[EnemyShootEvent], now: float, arena_size: float) -> np.ndarray:
    """Forward-simulate live bullets from shoot events -> (N,4) array of x,y,vx,vy."""
    rows: list[list[float]] = []
    for e in events:
        age = now - e.spawn_time
        if age < 0.0 or age > e.lifetime:
            continue
        for i in range(e.count):
            angle = e.base_angle + (i - (e.count - 1) / 2.0) * e.arc_gap
            vx, vy = np.cos(angle) * e.speed, np.sin(angle) * e.speed
            x, y = e.origin[0] + vx * age, e.origin[1] + vy * age
            if 0.0 <= x <= arena_size and 0.0 <= y <= arena_size:
                rows.append([x, y, vx, vy])
    return np.array(rows, np.float32) if rows else np.zeros((0, 4), np.float32)


def realm_to_gamestate(rs: RealmState) -> GameState:
    return GameState(
        arena_size=rs.arena_size,
        player_pos=rs.player_pos,
        player_hp=rs.player_hp,
        player_hp_max=rs.player_hp_max,
        player_mp=rs.player_mp,
        player_mp_max=rs.player_mp_max,
        ability_ready=rs.ability_ready,
        boss_pos=rs.boss_pos,
        boss_hp=rs.boss_hp,
        boss_hp_max=rs.boss_hp_max,
        enemy_bullets=reconstruct_bullets(rs.enemy_shoots, rs.now, rs.arena_size),
        player_bullets=rs.player_bullets,
    )


def realm_to_observation(rs: RealmState) -> dict[str, np.ndarray]:
    return build_observation(realm_to_gamestate(rs))


def action_to_intent(action) -> ActionIntent:
    """Map the policy's MultiDiscrete [move(0-8), aim(0-8)] to a protocol intent."""
    move_idx, aim_idx = int(action[0]), int(action[1])
    move = DIRS[move_idx - 1].copy() if move_idx > 0 else np.zeros(2, np.float32)
    if aim_idx > 0:
        return ActionIntent(move=move, shoot=True, aim=DIRS[aim_idx - 1].copy())
    return ActionIntent(move=move, shoot=False, aim=np.zeros(2, np.float32))
