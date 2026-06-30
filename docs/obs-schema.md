# The observation vector — the 9807-float contract

The observation is the **single invariant tying training and deployment together**. A policy that
clears in the server-as-sim is playing the real game *only because* it sees the same vector both
times. This document is the canonical layout; the constants below are the literal values in
`src/rotmg_rl/config.py` (the Python home for the layout) and `_pufferlib/ocean/server_env/server_env.h`
(the C-shim's home). They must not be re-derived anywhere else.

## Shape

A flat `float32` array of length **9807**, in this exact flatten order:

```
[ grid 7×31×31  =  6727 ][ minimap 3×32×32  =  3072 ][ scalars 8 ]
   offset 0..6726            offset 6727..9798          offset 9799..9806
```

- `9807 = 7*31*31 + 3*32*32 + 8 = 6727 + 3072 + 8`.
- Each block is C-order (row-major): `grid[channel, row, col]`, `minimap[channel, my, mx]`.
- The grid is **egocentric**: `GRID = 2*VIS_RADIUS + 1 = 31` (`VIS_RADIUS = 15`), so the player sits
  at the center cell `(HALF, HALF) = (15, 15)` and a world object at relative `(rel_x, rel_y)` lands
  at `floor(rel)+15`, in-bounds only.
- All values are normalized to roughly `[-1, 1]` (the obs-integrity proof asserts every element is
  finite and in `[-1.0001, 1.0001]`).

## Grid — 7 egocentric channels, 31×31 each (`config.py`: `CH_*`, `GRID`, `NUM_CH`)

| idx | constant | what it carries |
|---|---|---|
| 0 | `CH_WALL` | `1.0` where the cell is **not** walkable (a wall/obstacle), else `0.0` |
| 1 | `CH_ENEMY` | a snake/minion at `0.6`; the **boss** at `1.0` (only when the fight is active and the boss is within `VIS_RADIUS`) |
| 2 | `CH_EBULLET` | `1.0` where a live **enemy bullet** is, forward-simulated to the current tick |
| 3 | `CH_EBVX` | that bullet's unit velocity x-component (`cos(angle)`) at the same cell |
| 4 | `CH_EBVY` | that bullet's unit velocity y-component (`sin(angle)`) at the same cell |
| 5 | `CH_PBULLET` | `1.0` where one of the **player's own** projectiles is |
| 6 | `CH_GRENADE` | reserved for boss grenade telegraphs; **left zero** in both producers (the real server's telegraphs are not yet decoded) |

## Minimap — 3 fog-of-war channels, 32×32 each (`config.py`: `MM`, `NUM_MM_CH`, `MM_CH_*`)

A global, downsampled view the player builds up by exploring — no cheats: a cell only fills once it
has been discovered. `MM = 32`; each minimap cell covers a block of the full dungeon
(`mx = (x*MM)//width`, `my = (y*MM)//height`).

| idx | constant | what it carries |
|---|---|---|
| 0 | `MM_CH_TERRAIN` | discovered-walkable `+1.0`, discovered-wall `-1.0`, undiscovered (fog) `0.0` |
| 1 | `MM_CH_PLAYER` | `1.0` at the player's current minimap cell |
| 2 | `MM_CH_BOSS` | `1.0` at the boss's minimap cell, **only after the boss has been seen** |

## Scalars — 8 (`config.py`: `NUM_SCALARS`, comment enumerates them)

In flatten order, all in `[0, 1]`:

| idx | name | value |
|---|---|---|
| 0 | `hp` | `player.hp / player.hp_max` |
| 1 | `mp` | `player.mp / player.mp_max` |
| 2 | `spell_ready` | `1.0` if `mp >= spell_cost` else `0.0` |
| 3 | `boss_visible` | `1.0` if the boss is fight-active and within `VIS_RADIUS` |
| 4 | `confused` | `1.0` if the player is under the Confused status |
| 5 | `petrified` | `1.0` if the player is under the Petrified status |
| 6 | `boss_hp_frac` | `boss.hp / boss.hp_max` once fight-active, else `0.0` |
| 7 | `boss_invuln` | `1.0` if the boss is currently invulnerable (phase-transition taunt) |

## The three producers (all emit the same 9807 floats)

1. **`SimObsBuilder.cs`** (training, C#) — `sim-server/Sim/WorldServer/core/worlds/SimObsBuilder.cs`.
   Builds the obs from real `WorldServer` game state each gated tick and writes it into the shared-memory
   obs slot. The header comment in that file mirrors this layout verbatim.
2. **`RealObsBuilder`** (deploy, Python) — `src/rotmg_rl/obs.py`. Reconstructs the **same** vector at
   deploy time from passively-parsed packets (player state, enemies, player bullets, walkable tiles, and
   `EnemyShoot` bursts forward-simulated). It imports the channel/scalar constants from `config.py`, never
   re-hardcoding them, so it cannot drift from the contract.
3. **The C-shim shuttle** — `_pufferlib/ocean/server_env/{server_env.h,binding.c}`. Does **not** compute
   the obs; it copies the `N×9807` float32 obs block out of `/dev/shm/rotmg_sim_shm` into the PufferLib
   vec-buffer the `DungeonEncoder` policy consumes (and writes the `N×4` actions back). `OBS_SIZE` in
   `server_env.h` is defined as `GRID_SIZE + MM_SIZE + NUM_SCALARS = 9807`; the binding hard-fails if the
   shm header's `obs_len` disagrees.

## The proof: bit-identical across the shm boundary

`src/rotmg_rl/tools/verify_obs.py` drives a few gated rollout steps against a live server-as-sim, then
reads the shm obs region **directly** (numpy over the same `/dev/shm` file) and compares it element-wise
to the vec-buffer obs the policy actually received. It asserts `max|vec - shm| == 0.0`, every element
finite and in `[-1, 1]`, for every agent — i.e. the obs `SimObsBuilder.cs` wrote is exactly the obs the
policy saw, with no corruption crossing the boundary. `tools/verify_obs_async.py` is the same proof for
the async-overlap server mode. Keep these green when touching the obs.
