#!/usr/bin/env python3
"""Passive, read-only packet-telemetry reader for a self-hosted betterSkillys server.

READS the plaintext loopback game stream (127.0.0.1:<ephemeral> <-> 127.0.0.1:2050) for one
Flash-client connection, reassembles the server->client TCP payload in order, frames the
length-prefixed @realmlib/net packets, and reconstructs the same tick-state that
nrelay/src/policy-bridge.ts builds. It feeds that state into rotmg_rl.deploy.obs.RealObsBuilder
and emits the 9807-float observation each tick to /dev/shm for a policy process to consume.

It SENDS / INJECTS / ALTERS nothing. The only IO is: (1) reading tcpdump's pcap stream off `lo`,
(2) reading the static resource tables (GroundTypes.json / Objects.json) and packets.json byte map,
(3) writing the obs vector to /dev/shm. It never opens a socket to :2050 and never writes the wire.

Wire facts (verified against the running realmlib build on this box):
  frame      = [int32 length][int8 type][payload], length = TOTAL bytes incl. the 5-byte header
  byte ids   = packets.json:  MAPINFO=23 UPDATE=9 NEWTICK=12 ENEMYSHOOT=51 CREATE_SUCCESS=1
  RC4        = no-op in this build (crypto/rc4.js cipher() is `/*RC4-DISABLED*/ return;`) -> plaintext
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rotmg_rl.deploy.obs import RealObsBuilder

# ---- byte-id map (packets.json) -------------------------------------------------------------
PKT_MAPINFO = 23
PKT_UPDATE = 9
PKT_NEWTICK = 12
PKT_ENEMYSHOOT = 51

# ---- StatType ids (realmlib models/stat-type) ----------------------------------------------
ST_MAX_HP = 0
ST_HP = 1
ST_MAX_MP = 3
ST_MP = 4
ST_CONDITION = 29
ST_NAME = 31
ST_GUILD_NAME = 62
STRING_STATS = {ST_NAME, ST_GUILD_NAME}  # StatData.isStringStat() (betterSkillys: only 31, 62)

# ---- ConditionEffect bit positions (nrelay models/condition-effect.ts) ---------------------
CE_CONFUSED = 11
CE_PARALYZED = 14
CE_INVINCIBLE = 24
CE_INVULNERABLE = 25


def has_effect(condition: int, effect_bitpos: int) -> bool:
    # ported verbatim from nrelay hasEffect(): effectBit = 1 << (effect - 1); (condition & effectBit) === 1
    # (the `=== 1` quirk is intentional fidelity to the bridge that produced the training obs)
    effect_bit = 1 << (effect_bitpos - 1)
    return (condition & effect_bit) == 1


def has_any_effect(condition: int, *effect_bitpos: int) -> bool:
    # the INVINCIBLE|INVULNERABLE check the bridge does with a combined mask
    mask = 0
    for e in effect_bitpos:
        mask |= 1 << (e - 1)
    return (condition & mask) != 0


# ============================================================================================
# Reader: a forward-only cursor over a bytes buffer, mirroring @realmlib/net reader.js (big-endian)
# ============================================================================================
class Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.i = 0

    def remaining(self) -> int:
        return len(self.buf) - self.i

    def int32(self) -> int:
        v = struct.unpack_from(">i", self.buf, self.i)[0]
        self.i += 4
        return v

    def uint32(self) -> int:
        v = struct.unpack_from(">I", self.buf, self.i)[0]
        self.i += 4
        return v

    def short(self) -> int:
        v = struct.unpack_from(">h", self.buf, self.i)[0]
        self.i += 2
        return v

    def ubyte(self) -> int:
        v = self.buf[self.i]
        self.i += 1
        return v

    def boolean(self) -> bool:
        return self.ubyte() != 0

    def f32(self) -> float:
        v = struct.unpack_from(">f", self.buf, self.i)[0]
        self.i += 4
        return v

    def string(self) -> str:
        n = self.short()
        s = self.buf[self.i : self.i + n].decode("utf-8", "replace")
        self.i += n
        return s


# ---- data-type readers (realmlib/net/lib/data/*) -------------------------------------------
def read_world_pos(r: Reader) -> tuple[float, float]:
    return r.f32(), r.f32()


def read_stat(r: Reader) -> tuple[int, int]:
    # StatData.read: statType(ubyte); if string-stat -> readString else int32. Returns (type, intValue).
    st = r.ubyte()
    if st in STRING_STATS:
        r.string()
        return st, 0
    return st, r.int32()


def read_object_status(r: Reader) -> dict:
    # ObjectStatusData.read: objectId(int32), pos(WorldPos), stats[short]{StatData}
    object_id = r.int32()
    x, y = read_world_pos(r)
    n = r.short()
    stats: dict[int, int] = {}
    for _ in range(n):
        st, val = read_stat(r)
        stats[st] = val
    return {"object_id": object_id, "x": x, "y": y, "stats": stats}


def read_object_data(r: Reader) -> dict:
    # ObjectData.read: objectType(int32), status(ObjectStatusData)
    object_type = r.int32()
    status = read_object_status(r)
    status["object_type"] = object_type
    return status


def read_ground_tile(r: Reader) -> tuple[int, int, int]:
    # GroundTileData.read: x(short), y(short), type(int32)
    x = r.short()
    y = r.short()
    t = r.int32()
    return x, y, t


# ---- packet readers ------------------------------------------------------------------------
def parse_map_info(r: Reader) -> dict:
    width = r.short()
    height = r.short()
    name = r.string()
    return {"width": width, "height": height, "name": name}


def parse_update(r: Reader) -> dict:
    tiles_len = r.short()
    tiles = [read_ground_tile(r) for _ in range(tiles_len)]
    new_len = r.short()
    new_objects = [read_object_data(r) for _ in range(new_len)]
    drops_len = r.short()
    for _ in range(drops_len):
        r.int32()
    return {"tiles": tiles, "new_objects": new_objects}


def parse_new_tick(r: Reader) -> dict:
    r.int32()  # tickId
    r.int32()  # tickTime
    n = r.short()
    statuses = [read_object_status(r) for _ in range(n)]
    return {"statuses": statuses}


def parse_enemy_shoot(r: Reader) -> dict:
    bullet_id = r.int32()
    owner_id = r.int32()
    bullet_type = r.ubyte()
    sx, sy = read_world_pos(r)
    angle = r.f32()
    r.int32()  # damage
    if r.remaining() > 0:
        num_shots = r.ubyte()
        angle_inc = r.f32()
    else:
        num_shots, angle_inc = 1, 0.0
    return {
        "bullet_id": bullet_id,
        "owner_id": owner_id,
        "bullet_type": bullet_type,
        "origin_x": sx,
        "origin_y": sy,
        "angle": angle,
        "num_shots": num_shots,
        "angle_inc": angle_inc,
    }


# ============================================================================================
# Resource tables: replicate nrelay runtime's tiles[type].noWalk and objects[type].projectiles
# ============================================================================================
def _hex_int(s: str) -> int:
    return int(s, 16) if isinstance(s, str) and s.lower().startswith("0x") else int(s)


def load_no_walk(ground_types_path: Path) -> dict[int, bool]:
    data = json.loads(ground_types_path.read_text())
    rows = data["Ground"]
    out: dict[int, bool] = {}
    for row in rows:
        if "type" not in row:
            continue
        out[_hex_int(row["type"])] = "NoWalk" in row
    return out


def load_projectiles(objects_path: Path) -> dict[int, list[dict]]:
    """type(int) -> list of {speed (tiles/100ms), lifetime (ticks)} indexed by bulletType."""
    data = json.loads(objects_path.read_text())
    rows = data["Object"]
    out: dict[int, list[dict]] = {}
    for row in rows:
        if "type" not in row or "Projectile" not in row:
            continue
        projs = row["Projectile"]
        if isinstance(projs, dict):
            projs = [projs]
        plist: list[dict] = []
        for pr in projs:
            speed = float(pr["Speed"]) / 100.0 if "Speed" in pr else 0.6
            lifetime = float(pr["LifetimeMS"]) / 100.0 if "LifetimeMS" in pr else 30.0
            plist.append({"speed": speed, "lifetime": lifetime})
        out[_hex_int(row["type"])] = plist
    return out


# ============================================================================================
# Tick-state reconstruction: the stateful object store the bridge keeps across Update/NewTick
# ============================================================================================
BOSS_NAME_HINTS = ("stheno", "snake queen")


class GameState:
    """Mirrors the realmlib Client's running object store + the policy-bridge tick assembly.

    Update adds objects (objectId -> {object_type, pos, stats}); NewTick patches pos + stat deltas.
    The player is the object whose objectId == the CreateSuccess objectId; we instead detect it as
    the object carrying MAX_HP/MAX_MP/HP/MP player stats and the one the server moves like a player.
    The Wizard is identified by its NAME stat == the bot's char name; we latch the player objectId on
    the first NewTick status that carries an MP stat (enemies on this server do not report MP)."""

    def __init__(self, no_walk: dict[int, bool], projectiles: dict[int, list[dict]]) -> None:
        self.no_walk = no_walk
        self.projectiles = projectiles
        self.objects: dict[int, dict] = {}  # objectId -> {object_type,x,y,stats{}}
        self.player_id: int | None = None
        self.shots_buf: list[dict] = []
        self.map_w = 0
        self.map_h = 0

    def on_map_info(self, p: dict) -> None:
        self.map_w, self.map_h = p["width"], p["height"]
        self.objects.clear()
        self.player_id = None
        self.shots_buf.clear()

    def on_update(self, p: dict) -> list[tuple[int, int, bool]]:
        for od in p["new_objects"]:
            oid = od["object_id"]
            existing = self.objects[oid] if oid in self.objects else {"stats": {}}
            existing_stats = existing["stats"]
            existing_stats.update(od["stats"])
            self.objects[oid] = {
                "object_type": od["object_type"],
                "x": od["x"],
                "y": od["y"],
                "stats": existing_stats,
            }
        tiles_out: list[tuple[int, int, bool]] = []
        for x, y, ttype in p["tiles"]:
            walkable = not (ttype in self.no_walk and self.no_walk[ttype])
            tiles_out.append((x, y, walkable))
        return tiles_out

    def on_enemy_shoot(self, p: dict, now_ms: float) -> None:
        owner = self.objects[p["owner_id"]] if p["owner_id"] in self.objects else None
        speed, lifetime = 0.6, 30.0
        if owner is not None and owner["object_type"] in self.projectiles:
            plist = self.projectiles[owner["object_type"]]
            if 0 <= p["bullet_type"] < len(plist):
                speed = plist[p["bullet_type"]]["speed"]
                lifetime = plist[p["bullet_type"]]["lifetime"]
        self.shots_buf.append(
            {
                "origin_x": p["origin_x"],
                "origin_y": p["origin_y"],
                "angle": p["angle"],
                "count": p["num_shots"],
                "angle_inc": p["angle_inc"],
                "speed": speed,
                "lifetime": lifetime,
                "spawn_ms": now_ms,
            }
        )

    def on_new_tick(self, p: dict) -> None:
        for st in p["statuses"]:
            oid = st["object_id"]
            obj = self.objects[oid] if oid in self.objects else {"object_type": -1, "stats": {}}
            obj["x"] = st["x"]
            obj["y"] = st["y"]
            obj["stats"].update(st["stats"])
            self.objects[oid] = obj
            # latch the player: the only object that reports an MP stat is our character
            if self.player_id is None and ST_MP in obj["stats"] and ST_MAX_MP in obj["stats"]:
                self.player_id = oid

    def _stat(self, stats: dict, key: int, default: int = 0) -> int:
        return stats[key] if key in stats else default

    def build_tick(self, now_ms: float) -> dict | None:
        if self.player_id is None or self.player_id not in self.objects:
            return None
        pl = self.objects[self.player_id]
        ps = pl["stats"]
        cond = self._stat(ps, ST_CONDITION)
        player = {
            "x": pl["x"],
            "y": pl["y"],
            "hp": self._stat(ps, ST_HP),
            "hp_max": self._stat(ps, ST_MAX_HP, 1),
            "mp": self._stat(ps, ST_MP),
            "mp_max": self._stat(ps, ST_MAX_MP, 1),
            "confused": has_effect(cond, CE_CONFUSED),
            "petrified": has_effect(cond, CE_PARALYZED),
        }
        enemies: list[dict] = []
        for oid, obj in self.objects.items():
            if oid == self.player_id:
                continue
            es = obj["stats"]
            if ST_HP not in es or ST_MAX_HP not in es:
                continue  # not a damageable entity (portals, decor, loot)
            ename = es[ST_NAME] if ST_NAME in es else ""  # string stats are dropped to "" by read_stat
            is_boss = any(h in str(ename).lower() for h in BOSS_NAME_HINTS)
            econd = self._stat(es, ST_CONDITION)
            enemies.append(
                {
                    "x": obj["x"],
                    "y": obj["y"],
                    "hp": self._stat(es, ST_HP),
                    "hp_max": self._stat(es, ST_MAX_HP, 1),
                    "is_boss": is_boss,
                    "invuln": has_any_effect(econd, CE_INVINCIBLE, CE_INVULNERABLE),
                }
            )
        tick = {
            "player": player,
            "enemies": enemies,
            "player_bullets": [],  # damageEnemies projectiles are not in the server->client stream we read
            "now_ms": now_ms,
        }
        shots = self.shots_buf
        self.shots_buf = []
        return tick, shots


# ============================================================================================
# TCP reassembly off tcpdump's pcap stream (server->client direction only, in seq order)
# ============================================================================================
def tcpdump_stream(iface: str, port: int) -> subprocess.Popen:
    # -U unbuffered, -w - pcap to stdout, -i lo, filter tcp port 2050. READ-ONLY capture.
    cmd = ["tcpdump", "-i", iface, "-U", "-s", "0", "-w", "-", f"tcp port {port}"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def parse_pcap_records(stdout, port: int):
    """Yield (payload_bytes,) for each server->client TCP segment (src port == 2050), in capture order.

    Loopback is lossless and in-order, so concatenating payloads as they arrive suffices. We do not
    handle retransmits/reordering (they do not occur on lo for an established local stream)."""
    global_hdr = stdout.read(24)
    if len(global_hdr) < 24:
        return
    magic = struct.unpack_from("<I", global_hdr, 0)[0]
    if magic == 0xA1B2C3D4:
        endian, nano = "<", False
    elif magic == 0xD4C3B2A1:
        endian, nano = ">", False
    elif magic == 0xA1B23C4D:
        endian, nano = "<", True
    elif magic == 0x4D3CB2A1:
        endian, nano = ">", True
    else:
        raise SystemExit(f"unexpected pcap magic 0x{magic:08x}")
    # linktype is at offset 20; lo on Linux is usually LINKTYPE_NULL(0) or EN10MB(1) for `-i lo`.
    linktype = struct.unpack_from(endian + "I", global_hdr, 20)[0]
    while True:
        rec = stdout.read(16)
        if len(rec) < 16:
            return
        incl_len = struct.unpack_from(endian + "IIII", rec, 0)[2]
        data = stdout.read(incl_len)
        if len(data) < incl_len:
            return
        seg = _extract_tcp_payload(data, linktype, port)
        if seg is not None:
            yield seg


def _extract_tcp_payload(frame: bytes, linktype: int, server_port: int) -> bytes | None:
    off = 0
    if linktype == 1:  # EN10MB
        if len(frame) < 14:
            return None
        ethertype = struct.unpack_from(">H", frame, 12)[0]
        if ethertype != 0x0800:
            return None
        off = 14
    elif linktype == 0:  # NULL/loopback: 4-byte AF family header
        off = 4
    elif linktype == 113:  # LINUX_SLL
        off = 16
    else:
        off = 4
    if len(frame) < off + 20:
        return None
    vihl = frame[off]
    if (vihl >> 4) != 4:
        return None  # IPv4 only (loopback :2050 is IPv4 here)
    ihl = (vihl & 0x0F) * 4
    if frame[off + 9] != 6:  # protocol == TCP
        return None
    ip_total = struct.unpack_from(">H", frame, off + 2)[0]
    tcp_off = off + ihl
    if len(frame) < tcp_off + 20:
        return None
    src_port = struct.unpack_from(">H", frame, tcp_off)[0]
    data_off = (frame[tcp_off + 12] >> 4) * 4
    payload_start = tcp_off + data_off
    payload_end = off + ip_total
    if src_port != server_port or payload_end <= payload_start:
        return None
    return frame[payload_start:payload_end]


class FrameAssembler:
    """Concatenate inbound payloads and split into [int32 len][int8 type][payload] frames."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, chunk: bytes):
        self.buf += chunk
        while len(self.buf) >= 5:
            total = struct.unpack_from(">i", self.buf, 0)[0]
            if total < 5 or total > 8 * 1024 * 1024:
                # desync (we likely attached mid-stream); resync by dropping one byte
                self.buf.pop(0)
                continue
            if len(self.buf) < total:
                return
            ptype = self.buf[4]
            payload = bytes(self.buf[5:total])
            del self.buf[:total]
            yield ptype, payload


# ============================================================================================
# main loop
# ============================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--port", type=int, default=2050)
    ap.add_argument("--pcap-stdin", action="store_true", help="read a pcap stream on stdin (from `sudo tcpdump -w -`) instead of spawning tcpdump")
    ap.add_argument("--ground", default="/home/audiogen/rotmg-realgame/nrelay/resources/GroundTypes.json")
    ap.add_argument("--objects", default="/home/audiogen/rotmg-realgame/nrelay/resources/Objects.json")
    ap.add_argument("--out", default="/dev/shm/rotmg_obs.f32")
    ap.add_argument("--stats-every", type=int, default=20, help="print obs stats every N ticks (0=never)")
    ap.add_argument("--max-ticks", type=int, default=0, help="exit after N ticks (0=run forever)")
    args = ap.parse_args()

    no_walk = load_no_walk(Path(args.ground))
    projectiles = load_projectiles(Path(args.objects))
    print(f"[reader] loaded {len(no_walk)} ground types, {len(projectiles)} projectile defs", file=sys.stderr)

    state = GameState(no_walk, projectiles)
    obs_builder = RealObsBuilder()
    assembler = FrameAssembler()
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    proc = None
    if args.pcap_stdin:
        pcap_in = sys.stdin.buffer
        print(f"[reader] reading pcap from stdin, tcp port {args.port} (READ-ONLY)", file=sys.stderr)
    else:
        proc = tcpdump_stream(args.iface, args.port)
        pcap_in = proc.stdout
        print(f"[reader] tcpdump pid={proc.pid} on {args.iface} tcp port {args.port} (READ-ONLY)", file=sys.stderr)

    tick_n = 0
    map_set = False
    try:
        for (payload,) in ((p,) for p in parse_pcap_records(pcap_in, args.port)):
            for ptype, body in assembler.feed(payload):
                r = Reader(body)
                now_ms = time.time() * 1000.0
                if ptype == PKT_MAPINFO:
                    p = parse_map_info(r)
                    state.on_map_info(p)
                    obs_builder.__init__()  # fresh map/fog on a new map, matching the bridge's reset
                    obs_builder.set_map(p["width"], p["height"])
                    map_set = True
                    print(f"[reader] MAPINFO name={p['name']!r} size={p['width']}x{p['height']}", file=sys.stderr)
                elif ptype == PKT_UPDATE:
                    tiles = state.on_update(parse_update(r))
                    if map_set:
                        obs_builder.update_tiles(tiles)
                elif ptype == PKT_ENEMYSHOOT:
                    state.on_enemy_shoot(parse_enemy_shoot(r), now_ms)
                elif ptype == PKT_NEWTICK:
                    state.on_new_tick(parse_new_tick(r))
                    if not map_set:
                        continue
                    built = state.build_tick(now_ms)
                    if built is None:
                        continue
                    tick, shots = built
                    if shots:
                        obs_builder.add_shots(shots)
                    obs = obs_builder.build(tick)
                    tmp_path.write_bytes(obs.tobytes())
                    tmp_path.replace(out_path)
                    tick_n += 1
                    if args.stats_every and tick_n % args.stats_every == 0:
                        _print_stats(tick_n, obs, tick)
                    if args.max_ticks and tick_n >= args.max_ticks:
                        print(f"[reader] reached max-ticks={args.max_ticks}, exiting", file=sys.stderr)
                        return
    finally:
        if proc is not None:
            proc.terminate()


def _print_stats(tick_n: int, obs: np.ndarray, tick: dict) -> None:
    grid = obs[: 7 * 31 * 31].reshape(7, 31, 31)
    mm = obs[7 * 31 * 31 : 7 * 31 * 31 + 3 * 32 * 32].reshape(3, 32, 32)
    scal = obs[-8:]
    p = tick["player"]
    print(
        f"[obs] tick={tick_n} len={obs.size} finite={np.isfinite(obs).all()} "
        f"min={obs.min():.3f} max={obs.max():.3f} nonzero={int((obs != 0).sum())} "
        f"| player=({p['x']:.2f},{p['y']:.2f}) hp={p['hp']}/{p['hp_max']} mp={p['mp']}/{p['mp_max']} "
        f"enemies={len(tick['enemies'])} "
        f"| wall_cells={int((grid[0] > 0).sum())} enemy_cells={int((grid[3] > 0).sum())} "
        f"mm_terrain_nz={int((mm[0] != 0).sum())} scalars={np.round(scal, 3).tolist()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
