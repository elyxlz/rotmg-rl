"""Load the real Snake Pit dungeon map (.jm) into a navigable grid.

The .jm is base64+zlib-compressed uint16 tile indices into a `dict` of tile types (ground +
objects). "Empty" ground is void/wall; named ground is floor; an object blocks movement when the
real client would block it: its id contains "Wall", or it is flagged OccupySquare/FullOccupy in the
static-objects XML (e.g. Grey Pillar, Broken Grey Pillar). This gives the navigation map for the
whole-dungeon sim (M1).
"""

from __future__ import annotations

import base64
import functools
import heapq
import json
import math
import pathlib
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MAP_PATH = REPO_ROOT / "data" / "maps" / "snakepit.jm"
_XML_DIR = REPO_ROOT / "vendor" / "betterSkillys" / "source" / "Shared" / "resources" / "xml" / "prod"
STATIC_OBJECTS_XML = _XML_DIR / "EmbeddedData_StaticObjectsCXML.xml"


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


@functools.lru_cache(maxsize=1)
def _occupy_ids() -> frozenset[str]:
    """Object ids the real client treats as blocking: those flagged <OccupySquare/> or <FullOccupy/>
    in the static-objects XML (Player.isFullOccupy / Square.isWalkable). Parsed once, cached."""
    root = ET.fromstring(STATIC_OBJECTS_XML.read_text(encoding="latin-1"))
    ids: set[str] = set()
    for obj in root.findall("Object"):
        oid = obj.get("id")
        if oid and any(child.tag in ("OccupySquare", "FullOccupy") for child in obj):
            ids.add(oid)
    return frozenset(ids)


def _walkable(entries: list, idx: np.ndarray) -> np.ndarray:
    occupy = _occupy_ids()
    wlk = np.zeros(idx.shape, bool)
    for i, e in enumerate(entries):
        ground = e.get("ground", "Empty")
        blocked = any("Wall" in (o.get("id") or "") or (o.get("id") or "") in occupy for o in (e.get("objs") or []))
        if ground != "Empty" and not blocked:
            wlk[idx == i] = True
    return wlk


def geodesic_field(walkable: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Dijkstra geodesic distance (in tiles) from every walkable tile to `target`, over the real
    walkable grid. 8-connected with no diagonal corner-cutting (a diagonal step is allowed only when
    both orthogonal neighbors are walkable), diagonal cost sqrt(2) so distances stay in tile units
    (comparable to the euclidean baseline the approach reward was tuned against). Unreachable tiles
    are +inf. This is the privileged training-only navigation potential -- never an obs channel."""
    h, w = walkable.shape
    dist = np.full((h, w), math.inf, np.float64)
    tx, ty = target
    dist[ty, tx] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, tx, ty)]
    r2 = math.sqrt(2.0)
    steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, r2), (1, -1, r2), (-1, 1, r2), (-1, -1, r2))
    while pq:
        d, x, y = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dx, dy, cost in steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h) or not walkable[ny, nx]:
                continue
            if dx != 0 and dy != 0 and (not walkable[y, nx] or not walkable[ny, x]):
                continue  # no corner-cut: the footprint blocks diagonals past a wall corner
            nd = d + cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, nx, ny))
    return dist


def _nearest_walkable(walkable, x, y):
    """The walkable tile nearest (x, y); returns (x, y) unchanged if it is already walkable."""
    if walkable[y, x]:
        return x, y
    ys, xs = np.where(walkable)
    i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[i]), int(ys[i])


def find_objects(dmap: DungeonMap, id_substr: str) -> list[tuple[int, int]]:
    """All (x, y) tiles whose object id contains id_substr (case-insensitive)."""
    locs: list[tuple[int, int]] = []
    sub = id_substr.lower()
    for i, e in enumerate(dmap.entries):
        if any(sub in (o.get("id") or "").lower() for o in (e.get("objs") or [])):
            ys, xs = np.where(dmap.tile_index == i)
            locs.extend(zip(xs.tolist(), ys.tolist(), strict=True))
    return locs


# The .jm's authored enemy ids -> the SNAKE_TYPES row each spawns with (the per-type stats live in the
# C env's SNAKE_TYPES table / config.py's mirror; this is the only id->type mapping). Every real Snake
# Pit enemy archetype has its own row so the authored map reproduces each one's HP/DEF/damage/follow
# faithfully (Pit Snake is a weak dmg-10 filler, Brown Python a tanky DEF-20 wanderer, etc.).
SNAKE_TYPE_BY_ID = {
    "Pit Viper": 0,
    "Fire Python": 1,
    "Yellow Python": 2,
    "Greater Pit Snake": 3,
    "Greater Pit Viper": 4,
    "Pit Snake": 5,
    "Brown Python": 6,
}


def snake_grates(dmap: DungeonMap | None = None) -> list[tuple[int, int]]:
    """Every Snake Grate tile the real .jm places, as (x, y). The Snake Grate is the Snake Pit's
    continuous snake source (BehaviorDb.SnakePit "Snake Grate"): an Idle->Spawn->wait-2000ms->Idle
    loop that, whenever no child of a type exists within a small radius, drops one Pit Snake and one
    Pit Viper at the grate tile. The env replenishes from these tiles so the snake population is
    SUSTAINED across an episode rather than thinning by attrition (every authored snake spawned once)."""
    m = dmap if dmap is not None else load_jm()
    return find_objects(m, "Snake Grate")


def authored_snakes(dmap: DungeonMap | None = None) -> list[tuple[int, int, int]]:
    """Every enemy the real .jm authors, as (x, y, snake_type_index) with exact id matching (NOT the
    substring match of find_objects, so "Pit Snake" never absorbs "Greater Pit Snake"). These are the
    real fixed enemy positions + types -- the env spawns from this list (a difficulty-scaled fraction of
    it), so the chokepoint clusters and the full ~405-enemy d=1 map are exactly the authored layout."""
    m = dmap if dmap is not None else load_jm()
    out: list[tuple[int, int, int]] = []
    for i, e in enumerate(m.entries):
        type_idx = None
        for o in e.get("objs") or []:
            oid = o.get("id") or ""
            if oid in SNAKE_TYPE_BY_ID:
                type_idx = SNAKE_TYPE_BY_ID[oid]
                break
        if type_idx is None:
            continue
        ys, xs = np.where(m.tile_index == i)
        out.extend((int(x), int(y), type_idx) for x, y in zip(xs.tolist(), ys.tolist(), strict=True))
    return out
