"""POV renderer for the REAL-server state stream — the milestone-5 fallback.

The real Flash/AIR client can't run on this box (ARM64 Linux has no Adobe/HARMAN AIR runtime, and only
the AS3 source ships), so we render the bot playing on the real server with the SAME POV view the sim
uses (rotmg_rl.csim.render.render_pov), driven by the world the RealObsBuilder reconstructs from
the live packet stream. This is honest evidence: it draws exactly what the policy is being fed (explored
walls + fog-of-war minimap + reconstructed enemy bullets), not a privileged view.
"""

from __future__ import annotations

import numpy as np

from rotmg_rl.deploy.v3.obs import RealObsBuilder

_BOSS_RADIUS = 0.8  # DungeonConfig.boss_radius (cosmetic dot size only)


def render_frame(ob: RealObsBuilder, state: dict, size: int = 480, pov_radius: float = 16.0) -> np.ndarray:
    w, h = ob.w, ob.h
    walk = ob.walkable
    ppt = size / (2 * pov_radius)
    center = size / 2.0
    pl = state["player"]
    p = np.array([float(pl["x"]), float(pl["y"])], float)

    yy, xx = np.mgrid[0:size, 0:size]
    tx = np.floor(p[0] + (xx - center) / ppt).astype(int)
    ty = np.floor(p[1] + (yy - center) / ppt).astype(int)
    inb = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
    floor = np.zeros((size, size), bool)
    floor[inb] = walk[np.clip(ty, 0, h - 1), np.clip(tx, 0, w - 1)][inb]
    img = np.where(floor[..., None], np.array([58, 50, 42], np.uint8), np.array([16, 14, 20], np.uint8))

    def dot(world, color, r):
        sx, sy = int((world[0] - p[0]) * ppt + center), int((world[1] - p[1]) * ppt + center)
        y0, y1, x0, x1 = max(0, sy - r), min(size, sy + r + 1), max(0, sx - r), min(size, sx + r + 1)
        if y0 < y1 and x0 < x1:
            img[y0:y1, x0:x1] = color

    enemies = state["enemies"]
    boss = next((e for e in enemies if e["is_boss"]), None)
    for e in enemies:
        if e["is_boss"]:
            continue
        dot((e["x"], e["y"]), (200, 150, 50), max(2, int(0.6 * ppt)))
    eb = ob.last_eb
    for i in range(eb.shape[0]):
        dot(eb[i, :2], (255, 140, 0), max(1, int(0.3 * ppt)))
    for b in state["player_bullets"]:
        dot((b["x"], b["y"]), (120, 210, 255), max(1, int(0.25 * ppt)))
    if boss is not None and np.linalg.norm(np.array([boss["x"], boss["y"]], float) - p) <= pov_radius + _BOSS_RADIUS:
        dot((boss["x"], boss["y"]), (230, 50, 50), int(_BOSS_RADIUS * ppt))
    dot(p, (70, 240, 110), max(2, int(0.6 * ppt)))  # player at screen center

    # fog-of-war minimap, top-right (same as the env)
    mm = 120
    disc = ob.discovered
    full = np.full((h, w, 3), (10, 9, 13), np.uint8)
    full[disc & ~walk] = (44, 40, 50)
    full[disc & walk] = (70, 64, 58)
    mini = full[np.clip(np.arange(mm) * h // mm, 0, h - 1)][:, np.clip(np.arange(mm) * w // mm, 0, w - 1)]
    mdots = [(int(p[0]), int(p[1]), (70, 240, 110))]
    if ob.boss_seen and boss is not None:
        mdots.append((int(boss["x"]), int(boss["y"]), (230, 50, 50)))
    for (wx, wy, col) in mdots:
        mx, my = int(wx * mm / w), int(wy * mm / h)
        mini[max(0, my - 2):my + 3, max(0, mx - 2):mx + 3] = col
    img[6:6 + mm, size - mm - 6:size - 6] = mini

    # boss HP bar (red), top
    if ob.fight_active and boss is not None:
        bw = size - mm - 24
        bf = max(float(boss["hp"]), 0.0) / max(float(boss["hp_max"]), 1.0)
        img[8:16, 8:8 + bw] = (60, 20, 20)
        img[8:16, 8:8 + int(bf * bw)] = (235, 45, 45)

    # player HP (green) / MP (blue) bars, bottom-left
    for j, (frac, col) in enumerate(
        [(float(pl["hp"]) / max(float(pl["hp_max"]), 1.0), (60, 220, 60)), (float(pl["mp"]) / max(float(pl["mp_max"]), 1.0), (60, 120, 255))]
    ):
        yb = size - 18 + j * 8
        img[yb:yb + 6, 8:8 + 160] = (35, 35, 38)
        img[yb:yb + 6, 8:8 + int(max(0.0, frac) * 160)] = col

    # status effects (confused=purple, petrified=grey)
    for k, (active, col) in enumerate([(pl["confused"], (200, 60, 200)), (pl["petrified"], (170, 170, 170))]):
        if active:
            img[size - 27:size - 20, 8 + k * 11:16 + k * 11] = col
    return img.astype(np.uint8)
