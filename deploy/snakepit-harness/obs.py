"""The obs consumer (standalone subprocess) + the obs-reading helpers the gates use.

As a subprocess (`python obs.py`): reads the proxy's tee'd server->client bytes on stdin, frames the
@realmlib/net packets via obs_reader, reconstructs tick-state, and writes the 9807-float observation
to OBS_PATH atomically each tick. It also prints a periodic `objs=<n>` heartbeat to stderr that the
dungeon gate parses for the live object count.

As a library (imported by run.py / dungeon.py): obs_is_fresh / obs_age / obs_object_count read the
shared state without re-parsing the wire.

FIX 2: reads stdin with .read1(N), never .read(N) -- on a pipe read(N) blocks until N bytes arrive
and freezes the consumer forever; read1 returns whatever is currently available.
FIX 3: obs_reader.GameState.on_map_info already clears self.objects on a world transition, and
on_update/on_new_tick latch player_id when an object first carries MP+MAX_MP -- both preserved by
delegating wholesale to obs_reader here.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
import pathlib

import numpy as np

import config


# ---- gate helpers (library use) ------------------------------------------------------------
def obs_age(path: Path = config.OBS_PATH) -> float | None:
    """Seconds since OBS_PATH was last written, or None if it does not exist."""
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def obs_is_fresh(path: Path = config.OBS_PATH, max_age: float = config.OBS_FRESH_SECS) -> bool:
    """FIX 6: 'in-game' / 'obs live' == the file exists AND its mtime is younger than max_age."""
    age = obs_age(path)
    return age is not None and age < max_age


def obs_object_count(log_path: Path = config.OBS_LOG) -> int | None:
    """Latest live object count from the consumer's `objs=<n>` heartbeat, or None if not seen yet.

    FIX 6: dungeon entry is judged by a RELATIVE jump in this count vs the Nexus baseline (counts
    vary), never a fixed absolute threshold, and only while the obs keeps updating.
    """
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"objs=(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def read_obs_vector(path: Path = config.OBS_PATH) -> np.ndarray | None:
    """The current obs vector if it is the expected length, else None."""
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.float32)
    if data.size != config.OBS_FLOATS:
        return None
    return data


# ---- standalone consumer -------------------------------------------------------------------
def _run_consumer() -> None:
    sys.path.insert(0, str(config.OBS_READER_DIR))
    sys.path.insert(0, str(config.POLICY_SRC))
    import obs_reader as reader  # noqa: E402  (path wired above)

    no_walk = reader.load_no_walk(config.GROUND_TYPES)
    projectiles = reader.load_projectiles(config.OBJECTS_JSON)
    state = reader.GameState(no_walk, projectiles)
    obs_builder = reader.RealObsBuilder()
    assembler = reader.FrameAssembler()

    out = config.OBS_PATH
    tmp = out.with_suffix(out.suffix + ".tmp")
    map_set = False
    ticks = 0
    last_report = time.time()
    print("[obs] consumer up", file=sys.stderr, flush=True)

    while True:
        chunk = sys.stdin.buffer.read1(65536)  # FIX 2: read1, never read
        if not chunk:
            break
        for ptype, body in assembler.feed(chunk):
            try:
                r = reader.Reader(body)
                now_ms = time.time() * 1000.0
                if ptype == reader.PKT_MAPINFO:
                    info = reader.parse_map_info(r)
                    state.on_map_info(info)              # FIX 3: clears objects on world transition
                    obs_builder.__init__()
                    obs_builder.set_map(info["width"], info["height"])
                    map_set = True
                elif ptype == reader.PKT_UPDATE:
                    tiles = state.on_update(reader.parse_update(r))  # FIX 3: latches player_id on MP+MAX_MP
                    if map_set:
                        obs_builder.update_tiles(tiles)
                elif ptype == reader.PKT_ENEMYSHOOT:
                    state.on_enemy_shoot(reader.parse_enemy_shoot(r), now_ms)
                elif ptype == reader.PKT_NEWTICK:
                    state.on_new_tick(reader.parse_new_tick(r))
                    if map_set:
                        built = state.build_tick(now_ms)
                        if built is not None:
                            tick, shots = built
                            pathlib.Path("/dev/shm/rotmg_pos.txt").write_text("%.3f %.3f" % (tick["player"]["x"], tick["player"]["y"]))
                            if shots:
                                obs_builder.add_shots(shots)
                            obs = obs_builder.build(tick)
                            tmp.write_bytes(obs.tobytes())
                            tmp.replace(out)             # atomic publish
                            ticks += 1
            except (struct_error_types()) as exc:
                print("[obs] parse err type=%d %r" % (ptype, exc), file=sys.stderr, flush=True)

        if time.time() - last_report > 2.0:
            last_report = time.time()
            print("[obs] ticks=%d objs=%d pid=%s" % (ticks, len(state.objects), state.player_id), file=sys.stderr, flush=True)


def struct_error_types() -> tuple[type[BaseException], ...]:
    import struct

    return (struct.error, IndexError, ValueError, KeyError)


if __name__ == "__main__":
    _run_consumer()
