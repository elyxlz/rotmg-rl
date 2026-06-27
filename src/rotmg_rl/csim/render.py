"""POV renderer for the C dungeon env — presentation only, no dynamics.

Draws the same player-POV view the old numpy env produced (camera on the player, like the real ROTMG
client: explored floor + fog-of-war minimap + entity dots + HP/MP/boss bars), but reads a render-state
snapshot taken from the C env (`CDungeonSingle.render_state()`) rather than re-simulating anything. The
C env (pufferlib/ocean/dungeon/dungeon.h) is the single source of dynamics; this module only paints what it reports.
"""

from __future__ import annotations

import numpy as np


def render_pov(rs: dict, walkable: np.ndarray, size: int = 480, pov_radius: float = 16.0, boss_radius: float = 2.0) -> np.ndarray:
    """rs: a CDungeonSingle.render_state() dict. walkable: (h, w) bool map. Returns an (size,size,3) uint8
    RGB frame. boss_radius is cosmetic (the boss dot size)."""
    h, w = walkable.shape
    ppt = size / (2 * pov_radius)  # screen px per world tile
    center = size / 2.0
    px, py = float(rs["px"]), float(rs["py"])
    p = np.array([px, py], np.float32)
    boss = np.array([float(rs["boss_x"]), float(rs["boss_y"])], np.float32)
    fight_active = bool(rs["fight_active"])
    boss_hp_max = float(rs["boss_hp_max"])
    boss_hp = float(rs["boss_hp"])
    phase = int(rs["phase"])
    discovered = np.asarray(rs["discovered"], bool).reshape(h, w)

    yy, xx = np.mgrid[0:size, 0:size]
    tx = np.floor(px + (xx - center) / ppt).astype(int)
    ty = np.floor(py + (yy - center) / ppt).astype(int)
    inb = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    floor = np.zeros((size, size), bool)
    floor[inb] = walkable[np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)][inb]
    img = np.where(floor[..., None], np.array([58, 50, 42], np.uint8), np.array([16, 14, 20], np.uint8))

    def dot(world, color, r):
        sx, sy = int((world[0] - px) * ppt + center), int((world[1] - py) * ppt + center)
        y0, y1, x0, x1 = max(0, sy - r), min(size, sy + r + 1), max(0, sx - r), min(size, sx + r + 1)
        if y0 < y1 and x0 < x1:
            img[y0:y1, x0:x1] = color

    for g in rs["grenades"]:
        dot(g[:2], (200, 60, 200), max(2, int(g[2] * ppt)))
    for s in rs["snakes"]:
        dot(s[:2], (200, 150, 50), max(2, int(0.6 * ppt)))
    for b in rs["enemy_bullets"]:
        dot(b[:2], (255, 140, 0), max(1, int(0.3 * ppt)))
    for b in rs["player_bullets"]:
        dot(b[:2], (120, 210, 255), max(1, int(0.25 * ppt)))
    if np.linalg.norm(boss - p) <= pov_radius + boss_radius:
        dot(boss, (230, 50, 50), int(boss_radius * ppt))
    dot(p, (70, 240, 110), max(2, int(0.6 * ppt)))  # player at screen center

    # minimap (whole dungeon) top-right -- fog of war: only discovered tiles show, like the obs the
    # policy actually sees. Discovered floor lit, discovered wall dark-grey, undiscovered black.
    mm = 120
    full = np.full((h, w, 3), (10, 9, 13), np.uint8)  # fog (undiscovered)
    full[discovered & ~walkable] = (44, 40, 50)  # discovered wall
    full[discovered & walkable] = (70, 64, 58)  # discovered floor
    mini = full[np.clip(np.arange(mm) * h // mm, 0, h - 1)][:, np.clip(np.arange(mm) * w // mm, 0, w - 1)]
    dots = [(int(px), int(py), (70, 240, 110))]
    if bool(rs["boss_seen"]):  # boss only on the minimap once seen (no cheat reveal)
        dots.append((int(boss[0]), int(boss[1]), (230, 50, 50)))
    for (wx, wy, col) in dots:
        mx, my = int(wx * mm / w), int(wy * mm / h)
        mini[max(0, my - 2):my + 3, max(0, mx - 2):mx + 3] = col
    img[6:6 + mm, size - mm - 6:size - 6] = mini

    # BOSS HP bar (red) across the top + 3-phase markers
    if fight_active or boss_hp < boss_hp_max:
        bw = size - mm - 24  # leave room for the minimap on the right
        img[8:16, 8:8 + bw] = (60, 20, 20)  # empty (dark)
        img[8:16, 8:8 + int(max(0.0, boss_hp / boss_hp_max) * bw)] = (235, 45, 45)  # boss hp fill
        for ph in range(3):  # phase 1/2/3 segments; active phase lit
            seg = bw // 3
            col = (250, 205, 60) if phase == ph + 1 else (85, 75, 40)
            img[18:21, 8 + ph * seg + 1 : 8 + (ph + 1) * seg - 1] = col

    # player HP (green) / MP (blue) bars with empty backing, bottom-left
    hp_frac = float(rs["player_hp"]) / max(float(rs["player_hp_max"]), 1.0)
    mp_frac = float(rs["player_mp"]) / max(float(rs["player_mp_max"]), 1.0)
    for j, (frac, col) in enumerate([(hp_frac, (60, 220, 60)), (mp_frac, (60, 120, 255))]):
        yb = size - 18 + j * 8
        img[yb:yb + 6, 8:8 + 160] = (35, 35, 38)
        img[yb:yb + 6, 8:8 + int(max(0, frac) * 160)] = col

    # status effects (confused = purple, petrified = grey) when active, above the player bars
    for k, (active, col) in enumerate([(bool(rs["confused"]), (200, 60, 200)), (bool(rs["petrified"]), (170, 170, 170))]):
        if active:
            img[size - 27:size - 20, 8 + k * 11 : 16 + k * 11] = col
    return img.astype(np.uint8)
