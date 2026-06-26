"""Load the real Snake Pit dungeon map (.jm) into a navigable grid.

The .jm is base64+zlib-compressed uint16 tile indices into a `dict` of tile types (ground +
objects). "Empty" ground is void/wall; named ground is floor; objects whose id contains "Wall"
block movement. This gives the navigation map for the whole-dungeon sim (M1).
"""

from __future__ import annotations

import base64
import json
import pathlib
import zlib
from dataclasses import dataclass

import numpy as np

MAP_PATH = pathlib.Path(__file__).resolve().parents[3] / "data" / "maps" / "snakepit.jm"


@dataclass
class DungeonMap:
    width: int
    height: int
    tile_index: np.ndarray  # (h, w) uint16, index into entries
    walkable: np.ndarray  # (h, w) bool
    entries: list


def load_jm(path: str | pathlib.Path = MAP_PATH) -> DungeonMap:
    d = json.loads(pathlib.Path(path).read_text())
    w, h = int(d["width"]), int(d["height"])
    raw = zlib.decompress(base64.b64decode(d["data"]))
    n = w * h
    idx = np.frombuffer(raw, dtype=">u2")[:n]
    if idx.size != n or idx.max() >= len(d["dict"]):
        idx = np.frombuffer(raw, dtype="<u2")[:n]  # fall back to little-endian
    idx = idx.reshape(h, w)
    return DungeonMap(w, h, idx, _walkable(d["dict"], idx), d["dict"])


def _walkable(entries: list, idx: np.ndarray) -> np.ndarray:
    wlk = np.zeros(idx.shape, bool)
    for i, e in enumerate(entries):
        ground = e.get("ground", "Empty")
        blocked = any("Wall" in (o.get("id") or "") for o in (e.get("objs") or []))
        if ground != "Empty" and not blocked:
            wlk[idx == i] = True
    return wlk


def find_objects(dmap: DungeonMap, id_substr: str) -> list[tuple[int, int]]:
    """All (x, y) tiles whose object id contains id_substr (case-insensitive)."""
    locs: list[tuple[int, int]] = []
    sub = id_substr.lower()
    for i, e in enumerate(dmap.entries):
        if any(sub in (o.get("id") or "").lower() for o in (e.get("objs") or [])):
            ys, xs = np.where(dmap.tile_index == i)
            locs.extend(zip(xs.tolist(), ys.tolist()))
    return locs
