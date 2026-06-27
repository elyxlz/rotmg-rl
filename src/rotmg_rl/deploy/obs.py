"""Reconstruct the trained env observation from REAL betterSkillys game state.

Ground truth: the C env's obs (pufferlib/ocean/dungeon/dungeon.h compute_obs / update_visibility). We mirror it
bit-for-bit so the 95%-clear CDungeonPolicy sees the same 9807-float vector on the real server that
it saw in sim: an egocentric 7x31x31 grid + a 3x32x32 fog-of-war minimap + 8 scalars, flattened
[grid, minimap, scalars]. Constants + normalization come from rotmg_rl.config, never re-hardcoded.
"""

from __future__ import annotations

import numpy as np

from rotmg_rl.config import (
    AIM_DIRS,
    CH_EBULLET,
    CH_EBVX,
    CH_EBVY,
    CH_ENEMY,
    CH_PBULLET,
    CH_WALL,
    GRID,
    MM,
    MM_CH_BOSS,
    MM_CH_PLAYER,
    MM_CH_TERRAIN,
    MOVE_DIRS,
    NUM_CH,
    NUM_MM_CH,
    VIS_RADIUS,
    DungeonConfig,
)

MS_PER_TICK = 100.0  # sim calibration: 1 tick = 100ms, tiles/tick = speed_per_100ms (docs/real-game-analysis)
ACTIVATION_RANGE = 20.0  # DungeonConfig.activation_range: fight latches when player first within this of boss
SPELL_COST = 100.0  # DungeonConfig.spell_cost


class RealObsBuilder:
    """Stateful bridge state owner: the explored walkable map, the fog-of-war discovered mask, the
    boss-seen latch, the fight-active latch, and the live enemy-bullet bursts reconstructed from
    EnemyShoot packets (the server streams bursts, not per-frame bullet positions)."""

    def __init__(self) -> None:
        self.cfg = DungeonConfig()
        self.w = 0
        self.h = 0
        self.walkable: np.ndarray | None = None  # (h, w) bool, True=walkable; unmapped defaults walkable (wall=0)
        self.discovered: np.ndarray | None = None  # (h, w) bool fog-of-war
        self.boss_seen = False
        self.fight_active = False
        self.bursts: list[dict] = []  # active EnemyShoot bursts
        self.last_eb = np.zeros((0, 4), np.float32)  # last reconstructed enemy bullets (x,y,ux,uy), for the recorder

    def set_map(self, w: int, h: int) -> None:
        if self.w == w and self.h == h and self.walkable is not None:
            return
        self.w, self.h = int(w), int(h)
        self.walkable = np.ones((self.h, self.w), bool)
        self.discovered = np.zeros((self.h, self.w), bool)

    def update_tiles(self, tiles) -> None:
        # tiles: iterable of (x, y, walkable_bool) received from Update/GroundTile packets
        for x, y, wk in tiles:
            if 0 <= x < self.w and 0 <= y < self.h:
                self.walkable[int(y), int(x)] = bool(wk)

    def add_shots(self, shots) -> None:
        # each shot: {origin_x, origin_y, angle, count, angle_inc, speed, lifetime, spawn_ms}
        self.bursts.extend(shots)

    def _active_bullets(self, now_ms: float) -> np.ndarray:
        """Forward-simulate each live burst to current world positions. Returns (N,4) = x,y,ux,uy."""
        out: list[tuple[float, float, float, float]] = []
        keep: list[dict] = []
        for b in self.bursts:
            age = (now_ms - b["spawn_ms"]) / MS_PER_TICK
            if age < 0.0:
                keep.append(b)
                continue
            if age > b["lifetime"]:
                continue  # expired -> drop
            keep.append(b)
            for i in range(int(b["count"])):
                a = b["angle"] + i * b["angle_inc"]
                ca, sa = float(np.cos(a)), float(np.sin(a))
                x = b["origin_x"] + ca * b["speed"] * age
                y = b["origin_y"] + sa * b["speed"] * age
                if 0 <= int(x) < self.w and 0 <= int(y) < self.h:
                    out.append((x, y, ca, sa))  # unit velocity = (cos a, sin a)
        self.bursts = keep
        return np.array(out, np.float32).reshape(-1, 4)

    def _scatter(self, grid, ch, rel, value) -> None:
        half = GRID // 2
        cells = np.floor(rel).astype(int) + half
        inb = (cells[:, 0] >= 0) & (cells[:, 0] < GRID) & (cells[:, 1] >= 0) & (cells[:, 1] < GRID)
        cx, cy = cells[inb, 0], cells[inb, 1]
        grid[ch, cy, cx] = value[inb] if isinstance(value, np.ndarray) else value

    def _update_visibility(self, px, py, boss_pos) -> None:
        ipx, ipy = int(px), int(py)
        y0, y1 = max(0, ipy - VIS_RADIUS), min(self.h, ipy + VIS_RADIUS + 1)
        x0, x1 = max(0, ipx - VIS_RADIUS), min(self.w, ipx + VIS_RADIUS + 1)
        dy = np.arange(y0, y1) - ipy
        dx = np.arange(x0, x1) - ipx
        disk = (dy[:, None] ** 2 + dx[None, :] ** 2) <= VIS_RADIUS * VIS_RADIUS
        self.discovered[y0:y1, x0:x1] |= disk
        if boss_pos is not None and np.linalg.norm(np.array(boss_pos) - np.array([px, py])) <= VIS_RADIUS:
            self.boss_seen = True

    def _minimap(self, px, py, boss_pos) -> np.ndarray:
        mm = np.zeros((NUM_MM_CH, MM, MM), np.float32)
        cx = (np.arange(self.w) * MM) // self.w
        cy = (np.arange(self.h) * MM) // self.h
        cell = cy[:, None] * MM + cx[None, :]
        terr = mm[MM_CH_TERRAIN].reshape(-1)
        terr[cell[self.discovered & ~self.walkable]] = -1.0
        terr[cell[self.discovered & self.walkable]] = 1.0
        pmx, pmy = (int(px) * MM) // self.w, (int(py) * MM) // self.h
        mm[MM_CH_PLAYER, pmy, pmx] = 1.0
        if self.boss_seen and boss_pos is not None:
            bmx, bmy = (int(boss_pos[0]) * MM) // self.w, (int(boss_pos[1]) * MM) // self.h
            mm[MM_CH_BOSS, bmy, bmx] = 1.0
        return mm

    def build(self, state: dict) -> np.ndarray:
        """state keys: player{x,y,hp,hp_max,mp,mp_max,confused,petrified}, enemies[{x,y,hp,hp_max,
        is_boss,invuln}], player_bullets[{x,y}], now_ms. Returns flat float32 (len 9807)."""
        p = state["player"]
        px, py = float(p["x"]), float(p["y"])
        ppos = np.array([px, py], np.float32)
        enemies = state["enemies"]

        boss = next((e for e in enemies if e["is_boss"]), None)
        boss_pos = (float(boss["x"]), float(boss["y"])) if boss is not None else None
        if boss_pos is not None and np.linalg.norm(np.array(boss_pos) - ppos) <= ACTIVATION_RANGE:
            self.fight_active = True
        self._update_visibility(px, py, boss_pos)

        half = GRID // 2
        grid = np.zeros((NUM_CH, GRID, GRID), np.float32)
        ys, xs = np.mgrid[0:GRID, 0:GRID]
        wx = (px + xs - half).astype(int)
        wy = (py + ys - half).astype(int)
        inb = (wx >= 0) & (wx < self.w) & (wy >= 0) & (wy < self.h)
        grid[CH_WALL][inb] = (~self.walkable[np.clip(wy, 0, self.h - 1), np.clip(wx, 0, self.w - 1)])[inb].astype(np.float32)

        non_boss = np.array([[e["x"], e["y"]] for e in enemies if not e["is_boss"]], np.float32).reshape(-1, 2)
        if non_boss.shape[0]:
            self._scatter(grid, CH_ENEMY, non_boss - ppos, 0.6)
        boss_visible = bool(self.fight_active and boss_pos is not None and np.linalg.norm(np.array(boss_pos) - ppos) <= VIS_RADIUS)
        if boss_visible:
            self._scatter(grid, CH_ENEMY, (np.array(boss_pos, np.float32) - ppos).reshape(1, 2), 1.0)

        eb = self._active_bullets(float(state["now_ms"]))
        self.last_eb = eb
        if eb.shape[0]:
            rel = eb[:, :2] - ppos
            self._scatter(grid, CH_EBULLET, rel, 1.0)
            self._scatter(grid, CH_EBVX, rel, eb[:, 2])
            self._scatter(grid, CH_EBVY, rel, eb[:, 3])

        pb = np.array([[b["x"], b["y"]] for b in state["player_bullets"]], np.float32).reshape(-1, 2)
        if pb.shape[0]:
            self._scatter(grid, CH_PBULLET, pb - ppos, 1.0)
        # CH_GRENADE left zero: boss grenade telegraphs are not yet decoded from the real server.

        boss_hp_frac = 0.0
        boss_invuln = 0.0
        if self.fight_active and boss is not None:
            boss_hp_frac = max(float(boss["hp"]), 0.0) / max(float(boss["hp_max"]), 1.0)
            boss_invuln = 1.0 if boss["invuln"] else 0.0

        scalars = np.array(
            [
                float(p["hp"]) / max(float(p["hp_max"]), 1.0),
                float(p["mp"]) / max(float(p["mp_max"]), 1.0),
                1.0 if float(p["mp"]) >= SPELL_COST else 0.0,
                1.0 if boss_visible else 0.0,
                1.0 if p["confused"] else 0.0,
                1.0 if p["petrified"] else 0.0,
                boss_hp_frac,
                boss_invuln,
            ],
            np.float32,
        )
        return np.concatenate([grid.ravel(), self._minimap(px, py, boss_pos).ravel(), scalars]).astype(np.float32)


def action_to_intent(action: dict, player_speed: float = 0.55) -> dict:
    """Decode the 4-head MultiDiscrete action into a game intent (mirrors the env step decode):
    move 0=stand / 1..8=MOVE_DIRS, aim=AIM_DIRS unit vector, shoot/cast bools."""
    mv = action["move"]
    if mv == 0:
        dx, dy = 0.0, 0.0
    else:
        d = MOVE_DIRS[mv - 1]
        dx, dy = float(d[0]) * player_speed, float(d[1]) * player_speed
    aim = AIM_DIRS[action["aim"]]
    return {
        "dx": dx,
        "dy": dy,
        "aim_x": float(aim[0]),
        "aim_y": float(aim[1]),
        "shoot": bool(action["shoot"]),
        "cast": bool(action["cast"]),
    }
