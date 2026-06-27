"""Thin single-env handle over the C dungeon binding (rotmg_rl.csim.binding).

The C env (pufferlib/ocean/dungeon/dungeon.h) is the single source of the Snake Pit dynamics. This wraps ONE env
instance for evaluation + POV rendering using only numpy (no pufferlib / torch), so it runs in any
venv that has the compiled binding. `step` returns the flat [grid, minimap, scalars] obs plus the
gym-style (reward, terminated, truncated, info); `render_state` is a read-only snapshot of the live C
entity buffers for the POV renderer (never a re-simulation). `OBS_SIZE` lives here (not in
csim.dungeon) so this path stays free of the heavy pufferlib import.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from rotmg_rl.config import GRID, MM, NUM_CH, NUM_MM_CH, NUM_SCALARS, DungeonConfig
from rotmg_rl.csim import binding

OBS_SIZE = NUM_CH * GRID * GRID + NUM_MM_CH * MM * MM + NUM_SCALARS


class CDungeonSingle:
    """One C dungeon env behind a gym-ish surface (numpy-only). The obs buffer is shared with C and
    refilled in place every reset/step. The C env auto-resets in place on episode end; `step` reports
    the just-ended episode's outcome from the env's outcome latch (which survives that auto-reset)."""

    def __init__(self, config: DungeonConfig | None = None, seed: int = 0):
        self.cfg = config or DungeonConfig()
        self.obs = np.zeros(OBS_SIZE, np.float32)
        self._act = np.zeros(4, np.int32)
        self._rew = np.zeros(1, np.float32)
        self._term = np.zeros(1, np.uint8)
        self._trunc = np.zeros(1, np.uint8)
        self.handle = binding.env_init(self.obs, self._act, self._rew, self._term, self._trunc, seed, **asdict(self.cfg))
        self._closed = False

    def reset(self, seed: int = 0) -> np.ndarray:
        binding.env_reset(self.handle, seed)
        return self.obs

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._act[:] = action
        binding.env_step(self.handle)
        g = binding.env_get(self.handle)
        terminated = bool(self._term[0])
        ep_done = bool(g["ep_done"])
        truncated = ep_done and not terminated
        cleared = ep_done and bool(g["ep_cleared"])
        boss_hp_frac = g["ep_boss_hp_frac"] if ep_done else max(g["boss_hp"], 0.0) / g["boss_hp_max"]
        info = {"cleared": cleared, "ep_done": ep_done, "boss_hp_frac": float(boss_hp_frac), "steps": int(g["steps"])}
        return self.obs, float(self._rew[0]), terminated, truncated, info

    def render_state(self) -> dict:
        """Read-only snapshot of the live C entity buffers (player/snakes/bullets/grenades/boss/fog)."""
        return binding.env_get(self.handle)

    def get(self) -> dict:
        return binding.env_get(self.handle)

    def put(self, **kwargs) -> None:
        """Inject deterministic state (player pos / fight_active / phase / boss_hp) + refresh obs."""
        binding.env_put(self.handle, **kwargs)

    def close(self) -> None:
        if not self._closed:
            binding.env_close(self.handle)
            self._closed = True
