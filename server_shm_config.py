"""The d-flow channel (Python side): write the live difficulty config into the SAME shm region the
C# server + the PufferLib C-shim share, so a d change applies to the next episode the C# server spawns
with NO server restart. The C-shim owns the obs/action/reward/done slots + the barrier ctrl words; this
writes ONLY the difficulty-config block at the region tail (after the 2 ctrl ints), which the C-shim
never touches, so there is no contention with the lockstep gate.

Region layout (must match SimShmBridge.cs / server_env.h byte-for-byte):
    [HEADER_INTS int32][N*OBS float32][N*4 float32][N float32][N float32][CTRL_INTS int32][CONFIG_INTS int32]
The config block is 5 int32s: [valid(MAGIC), spawn_geo_dist, agent_hp, agent_def, boss_hp]. The C# side
treats valid==MAGIC as "a live config is present" and falls back to its SIM_* env defaults otherwise, so
writing the block here is what switches the server from the fixed proof config to the live d-curriculum.

The trainer holds the file mmap'd for the whole run and pokes the 5 ints whenever d changes -- a handful
of bytes, far off the hot path. The gate already serializes visibility (the C-shim bumps req AFTER any
write this process makes is in the shared page; the C# world reads the config at its next spawn)."""

from __future__ import annotations

import mmap
import os
import struct

HEADER_INTS = 4
MAGIC = 0x52544D47  # 'RTMG', == SimShmBridge.MAGIC
OBS_LEN = 9807
N_ATNS = 4
CTRL_INTS = 2
CONFIG_INTS = 5

# config sub-offsets (int index within the 5-int block)
CFG_VALID = 0
CFG_SPAWN_GEO = 1
CFG_AGENT_HP = 2
CFG_AGENT_DEF = 3
CFG_BOSS_HP = 4

_DEFAULT_SHM_PATH = "/dev/shm/rotmg_sim_shm"


def _region_bytes(n: int) -> int:
    return (
        HEADER_INTS * 4
        + n * OBS_LEN * 4
        + n * N_ATNS * 4
        + n * 4  # rewards
        + n * 4  # dones
        + CTRL_INTS * 4
        + CONFIG_INTS * 4
    )


def _config_byte_offset(n: int) -> int:
    """Byte offset of the config block start (== SimShmBridge._cfgBase)."""
    return (
        HEADER_INTS * 4
        + n * OBS_LEN * 4
        + n * N_ATNS * 4
        + n * 4
        + n * 4
        + CTRL_INTS * 4
    )


class ShmConfigChannel:
    """A thin handle over the shm config tail. Open once (after the C# server created the region),
    then call write(config) whenever d changes. Verifies the region header (magic + N) so a wrong N
    or a not-yet-created region fails loudly instead of corrupting memory."""

    def __init__(self, n_agents: int, shm_path: str | None = None) -> None:
        self._n = n_agents
        self._path = shm_path or os.environ.get("SIM_SHM_PATH", _DEFAULT_SHM_PATH)
        size = _region_bytes(n_agents)
        st = os.stat(self._path)
        if st.st_size < size:
            raise RuntimeError(
                f"shm '{self._path}' is {st.st_size} bytes, need >= {size} for n={n_agents} "
                f"(is the C# server up with SIM_SHM=1 and the config tail build?)"
            )
        self._fd = os.open(self._path, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        magic, n = struct.unpack_from("<ii", self._mm, 0)
        if magic != MAGIC or n != n_agents:
            self.close()
            raise RuntimeError(f"shm header mismatch magic={magic:x} n={n} (want magic={MAGIC:x} n={n_agents})")
        self._cfg_off = _config_byte_offset(n_agents)

    def write(self, config: dict) -> None:
        """Publish a d-derived config: {spawn_geo_dist, agent_hp, agent_def, boss_hp}. Writes the four
        knob ints first, then stamps valid=MAGIC last (so the C# side never reads a half-written block:
        it only treats the block as live once valid is set, and we set it after the payload)."""
        base = self._cfg_off
        struct.pack_into("<i", self._mm, base + CFG_SPAWN_GEO * 4, int(config["spawn_geo_dist"]))
        struct.pack_into("<i", self._mm, base + CFG_AGENT_HP * 4, int(config["agent_hp"]))
        struct.pack_into("<i", self._mm, base + CFG_AGENT_DEF * 4, int(config["agent_def"]))
        struct.pack_into("<i", self._mm, base + CFG_BOSS_HP * 4, int(config["boss_hp"]))
        struct.pack_into("<i", self._mm, base + CFG_VALID * 4, MAGIC)

    def read(self) -> dict:
        """Read back the live config (for verification)."""
        base = self._cfg_off
        valid, spawn, hp, df, boss = struct.unpack_from("<iiiii", self._mm, base)
        return {
            "valid": valid == MAGIC,
            "spawn_geo_dist": spawn,
            "agent_hp": hp,
            "agent_def": df,
            "boss_hp": boss,
        }

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None
