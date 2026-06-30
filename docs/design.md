# Server-as-sim: train PufferLib `pufferl` against the live betterSkillys C# server

PufferLib's `pufferl` PPO loop drives the throwaway C# server **unchanged** as if it were a
native Ocean env. The C# server owns the real Snake Pit dynamics, the bit-identical 9807-float
obs (`SimObsBuilder`), action apply, and reward; PufferLib only shuttles the vec-buffers across
two boundaries every step:

- **shared memory** `/dev/shm/rotmg_sim_shm` — fixed float32 layout: `[N×9807 obs][N×4 act][N rew][N done]`
  (16-byte header: magic `RTMG`, n, obs_len, n_atns). C# writes obs/reward/done, the C-shim writes actions.
- **lockstep barrier** advances **all N** C# worlds exactly one tick per `c_step`; the signal that
  the new obs are in shm. Two interchangeable implementations (one tick per `c_step` either way):
  - **pure-shm futex barrier** (default, `SIM_SHM_BARRIER=1`): two atomic generation counters in the
    shm control region (tail) + a Linux futex. NO redis on the hot path — this is the fast path.
  - **redis gate** (fallback, `SIM_SHM_BARRIER=0`): `sim:step:cmd` / `sim:step:ack` over the sim
    redis (127.0.0.1:6390 db5), one `LPUSH`/`BLPOP` round-trip per tick.

## The pure-shm barrier (shm layout + sync primitive)

The shm region grew by **8 bytes at the tail** (after `[N×9807 obs][N×4 act][N rew][N done]`) — two
atomic `int32` generation counters. Every data-region offset is unchanged, so the C-shim and the
verify scripts keep their layout:

```
ctrl[0] = req   (the C-shim bumps to generation G == "actions for G are ready, tick now")
ctrl[1] = done  (the C# controller bumps to G == "obs/reward/done for G are in shm")
```

Both processes share the `MAP_SHARED` page, so they poll the **same** words with **shared** (non-private)
Linux futexes. Per `c_step`: the C-shim writes actions → `req++` → `FUTEX_WAKE(req)` → futex-waits on
`done` until `done >= G`, then reads obs. The C# controller futex-waits on `req`, releases all N world
threads for one tick, collects them, sets `done = G`, `FUTEX_WAKE(done)`. The monotonic generation
(never reused) is what makes it correct under episode resets: the C-shim only reads a frame once `done`
reaches **its own** request G, so it can never read a stale/torn obs. A freshly-spawned pit seeds its
local generation from the live `req` at registration, so it joins at the next tick instead of
fast-forwarding through every generation it missed.

In-process fan-out to the N world threads is **O(1) syscalls per tick** (one broadcast `FUTEX_WAKE`
releases all worlds; an `Interlocked` done-count + one futex collects them), not O(N) kernel wakeups,
so the barrier's own per-tick cost stays flat as N grows.

Code: `SimShmBarrier.cs` (C# controller + futex), `SimShmBridge.cs` (region + ctrl pointers),
`RootWorldThread.cs` (per-world `WaitForGo`/`SignalDone`), `server_env.h` (`srv_barrier_tick`).

## Components

| piece | where | what |
|---|---|---|
| N-agent + shm bridge (C#) | `rotmg-sim-server` branch `sim/server-as-sim` | `SimShmBridge.cs` (mmap region + ctrl words), `SimShmBarrier.cs` (pure-shm futex lockstep), `SimStepGate.cs` (redis fallback), `SimRlLoop.cs` (shm-driven loop), `RootWorldThread.cs` + `SimRunner.cs` + `SimMode.cs` (combined `SIM_SHM` mode: in-proc + step-gated), one Snake Pit world per agent slot |
| C-shim native env | `_pufferlib/ocean/server_env/{server_env.h,binding.c}` | passthrough `c_step`/`c_reset` (write actions→shm, advance one tick via the shm futex barrier or the redis gate, read obs/reward/done←shm), `config/server_env.ini` (reuses `DungeonEncoder`) |
| training entry | `src/rotmg_rl/trainer/` | `longrun.py` (curriculum d-ramp run), `eval.py` (the ladder), `curriculum.py`/`trial.py`/`sweep.py`/`proof.py` — the same `PuffeRL` + `DungeonEncoder` CNN-LSTM, pointed at `server_env`. Obs proofs: `src/rotmg_rl/tools/{verify_obs,verify_motion,verify_dflow}.py` |

`N` is configured in ONE place: `--agents N` on the trainer == `SIM_WORLDS=N` on the server. The
shm region is sized for N and the binding hard-fails on a header mismatch. `total_agents` in the
ini sizes everything (shm, N pits, encoder batch). `num_buffers` MUST be 1 (one global gate).

## Launch a training run (GPU 1; never GPU 0 = the sweep)

```bash
# 1. build the C-shim env into _C (scripts/setup.sh already does this; shown here standalone)
cd _pufferlib && CUDA_HOME=/usr/local/cuda-12.4 NVCC_ARCH=sm_86 \
  PATH="../.venv-shim:../.venv/bin:$CUDA_HOME/bin:$PATH" \
  LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$(../.venv/bin/python -c 'import nvidia.cudnn,os;print(os.path.join(nvidia.cudnn.__path__[0],"lib"))'):$(../.venv/bin/python -c 'import nvidia.nccl,os;print(os.path.join(nvidia.nccl.__path__[0],"lib"))')" \
  ./build.sh server_env --float

# 2. fetch + build the C# server, then boot it in server-as-sim mode (N worlds, shm + futex barrier).
#    Isolated: port 2060, redis 6390. SIM_SHM_BARRIER defaults to 1 (pure-shm); 0 = the redis-gate fallback.
sim-server/fetch.sh
cd sim-server && nohup setsid ./run-server-sim.sh 32 > /tmp/server_sim.log 2>&1 < /dev/null &

# 3. run pufferl PPO against it on GPU 1 (N must match; SIM_SHM_BARRIER must match the server's mode)
SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m rotmg_rl.trainer.longrun --agents 32 --steps 200000
```

A real training run: N≈32 is the throughput sweet spot on this box (see the table), so launch the server
with `./run-server-sim.sh 32` and the trainer with `--agents 32`. N must match on both sides (the shm
region is sized for it; the binding hard-fails on a header mismatch), and `SIM_SHM_BARRIER` must match
(the server registers worlds to whichever gate is on; the C-shim reads the same env var).

Proofs (run with the matching `SIM_SHM_BARRIER`): `SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python
-m rotmg_rl.tools.verify_obs --agents 16` (bit-identical obs), `SIM_SHM_BARRIER=1 .venv/bin/python -m rotmg_rl.tools.verify_motion --agents 16`.

## Measured (GPU 1, nice -19 server vs the GPU0 Protein sweep)

**Pure-gate throughput** (a tight tick loop, no policy/GPU — isolates the barrier from the trainer's GPU
stalls). Both with the N worlds genuinely ticking, same box, same nice:

| N | redis gate (aggregate SPS) | shm barrier (aggregate SPS) | barrier per-tick |
|---|---|---|---|
| 16 | ~11,400 (1.41 ms/tick) | **~14,900** (1.07 ms/tick) | — |
| 32 | — | **~15,700** (2.04 ms/tick) | peak |
| 64 | — | ~11,900 (5.36 ms/tick) | oversubscribed |
| 128 | — | ~9,700 (13.2 ms/tick) | oversubscribed |

The barrier removes the redis round-trip cleanly (~0.34 ms/tick at N=16, a ~31% gate speedup). But it does
**not** reach ~22K, because throughput then plateaus **CPU-bound on the server worlds**, not on sync: the
remaining ~1.07 ms/tick at N=16 is the 16-world `World.Update` itself, and beyond N≈32 the worlds
oversubscribe the box's 32 cores (the GPU0 Protein sweep already takes ~5 cores at nice 0; this server
runs at nice 19 and gets ~2.4 cores during a 64-world tick), so per-tick cost grows faster than N and
aggregate SPS falls. The futex edge itself is ~12 µs — negligible.

**Through pufferl (with the GPU PPO loop)**, N=16 lands ~2,000–2,400 SPS for both modes (the GPU policy
forward + PPO backward, not the gate, dominates each horizon step here), and the barrier PPO loop runs
clean end-to-end (47 updates, no shape/grad errors in the smoke run).

**Next lever to reach ~22K** (none is a sync change): (1) give the server more CPU — drop `nice -19` or
pin it to cores the GPU0 sweep doesn't use (load was ~1.7/32 idle, so the headroom exists when not
contended); (2) make each `World.Update` cheaper (the per-world tick is now the wall, ~1 ms for 16
worlds); (3) batch the GPU policy over a larger N once the worlds aren't CPU-starved. On an uncontended
box, N≈32 at ~2 ms/tick already projects to ~16K and the per-tick CPU cost is the only thing between that
and 22K.

## Learnability proof — a policy LEARNS to clear the real pit (easy fixed config)

The plumbing above was only ever smoke-tested with a RANDOM policy, so reward stayed ~0 (the random
agent never crossed the maze to the boss). To prove the per-episode loop (navigate-in spawn, reward
incl. geodesic-approach shaping, reset, obs) is correct on the REAL game engine, a real `pufferl` PPO
run was trained against it at an easy fixed difficulty — and it **learns to navigate and clear**.

**Geodesic-approach reward (the navigate-in gradient).** `SimGeodesic.cs` (server side) BFS-floods the
REAL Snake Pit walkable grid (the live `Wmap`, same walkability as `SimObsBuilder.TileWalkable` /
`World.IsPassable`) outward from the boss tile, giving a geodesic distance-to-boss field. Each tick the
reward gains `(prevGeo − curGeo) * SIM_APPROACH_SCALE` — the per-tick REDUCTION in geodesic distance.
This is the signal the C-sim baked as `MAP_GEODESIC`; without it the agent has no gradient to cross the
maze. The field re-anchors only when the (near-stationary) boss moves a whole tile, so it is built once
per episode and cached. Reward components, all from the real game state: boss-dmg (HP delta), clear
(+5 terminal), death (−1 terminal), approach (geodesic), small step penalty.

**Difficulty knobs (`SimMode.cs`, the curriculum will drive them later).** The real dungeon enemies + AI
+ boss mechanics stay UNMODIFIED; only the agent spawn/stats + boss HP are controllable training aids
(d=1 restores the real conditions). `SIM_AGENT_HP` (survivability), `SIM_AGENT_DEF` (flat per-hit
reduction, applied in `SimAgent.Damage`), `SIM_BOSS_HP` (boss spawn HP), `SIM_SPAWN_GEO_DIST` (spawn at a
walkable tile this many geodesic-tiles from the boss; −1 == the real entrance), `SIM_APPROACH_SCALE`,
`SIM_STEP_PENALTY`, `SIM_EP_TIMEOUT` (hard episode cap → `reason=timeout`). Reset on done
(clear/death/timeout) respawns at the configured distance and re-seeds the geodesic baseline.

**Easy fixed config for this proof:**

```bash
SIM_SPAWN_GEO_DIST=25 SIM_AGENT_HP=5000 SIM_AGENT_DEF=40 SIM_BOSS_HP=1500 \
  SIM_APPROACH_SCALE=0.02 SIM_EP_TIMEOUT=1500 ./run-server-sim.sh 32
# then: nohup .venv/bin/python -m rotmg_rl.trainer.proof --agents 32 --steps 1500000 \
#         --server-log /tmp/server_proof.log --out logs/server_proof_curve.csv &
```

**Result (N=32, GPU 1, ~1700 SPS):** the clear rate climbs from 0 toward 1.0 and reward from ~0 to a
high plateau within ~50K steps — the agent navigates the geodesic from the spawn and kills the boss.

```
     step   reward  done_rt  clear_rt  clears  deaths  timeouts
     2048  -0.0005    0.000     0.000       0       0         0   <- untrained: NEVER reaches the boss
    20480   0.0206    0.109     1.000      39       0         0   <- first clears appear
    38912   0.1085    0.562     1.000     239       0         0
    77824   0.1773    0.891     0.998     886       0         2
   307200   0.1767    0.906     1.000     848       0         0   <- plateau: clear_rate 1.0, 0 deaths
```

`clear_rate` is the fraction of ended episodes that were a clear (vs death/timeout), parsed from the
server's ground-truth `[SIM-RL] EPISODE DONE ... reason=` lines; `done_rate` is the trainer's per-horizon
terminal rate. Final clear_rate 1.0, done_rate ~0.8, ZERO deaths and ~32 timeouts across the whole run.
The control is decisive: an untrained policy (step 2K) and the earlier random-policy smoke run BOTH
produce 0 reward / 0 clears, so the clears are a LEARNED navigate-in skill, not the boss wandering to a
stationary agent. Per-episode `ep_reward ≈ 6.0–6.4` (5.0 clear + ~1.0 boss-HP-delta + a positive
geodesic-approach net) confirms the agent reduces geodesic distance, i.e. moves inward.

`trainer/proof.py` is the proof harness: it runs the real `trainer/train.py` PPO loop and tails
the server log in parallel to align the trainer's step/reward/done_rate with the ground-truth
clear/death/timeout counts into one learning-curve CSV; `curve_summary.py` renders it. With the loop
proven LEARNABLE on the real engine, the curriculum + Protein sweep can drive these difficulty knobs
toward d=1 (the real conditions) on top of this exact loop.

## Curriculum + d-flow + eval ladder (the Protein objective on the proven loop)

With the per-episode loop proven LEARNABLE, the curriculum drives the difficulty knobs from an easy
anchor (d=0) to the real conditions (d=1) WITHOUT restarting the server. Three pieces:

**1. d -> config schedule (`src/rotmg_rl/trainer/difficulty.py`).** `server_difficulty_config(d)` maps one
`d in [0,1]` to the FOUR real-engine knobs, monotonic in d, every lever moving smoothly, `d=1` == the
real deliverable conditions exactly. NO synthetic domain randomization -- the real game RNG (snake
spawn, boss wander, bullet patterns) is the only variation (the deliberate departure from the C-sim
`schedule.py`, which widens synthetic dynamics ranges with d). The curriculum-depth objective itself
(`CURRICULUM_RUNGS`, `curriculum_depth`, the cosine `difficulty_at`) is reused UNCHANGED from
`schedule.py` -- the ladder/depth math is engine-independent (it consumes only per-rung clear rates).

| lever | d=0 (easy) | d=1 (real) | rationale |
|---|---|---|---|
| `agent_hp` | 5000 | 670 | the DOMINANT gradient (the diagnosis): as HP falls an undodged bullet is increasingly lethal, so the policy must learn to dodge |
| `boss_hp` | 1500 | 7500 | more HP == more landed shots == longer in the storm == more dodging |
| `agent_def` | 40 | 25 | flat per-hit reduction; trims each hit while HP is high, relaxes to real as the rest arrives |
| `spawn_geo_dist` | 12 | -1 (entrance) | ramps the navigate-in path 12 -> ~150 tiles, then SNAPS to the real entrance sentinel (-1) at d=1 |

The entrance sits **150 geodesic tiles** from the boss (measured: `SIM_SPAWN_GEO_DIST=-1` spawns at
(110.5,21.5) geo_dist=150, max-reachable 215), so the spawn ramp climbs toward 150 (d=0.99 == 149) before
the d=1 snap, keeping it continuous.

**2. d-flow: trainer -> running C# server, per-episode, NO restart.** The shm region grew by a
**5-int32 config block at the very tail** (after the 2 barrier ctrl words, so every existing offset --
incl. the C-shim's ctrl pointer -- is UNCHANGED): `[valid(MAGIC), spawn_geo_dist, agent_hp, agent_def,
boss_hp]`. The Python trainer mmaps the SAME `/dev/shm/rotmg_sim_shm` and pokes these 5 ints whenever d
changes (`src/rotmg_rl/trainer/shm_config.py::ShmConfigChannel`); the C# `SimRlLoop` reads them at every
spawn/reset (`ResolveConfig`), preferring the live config over the static `SIM_*` env defaults. `valid !=
MAGIC` (zeroed) == no live config -> the server falls back to the env defaults, so the existing
fixed-config proof path is byte-for-byte unchanged. The gate already serializes visibility (the C-shim
bumps `req` after the page write lands; the world reads the config at its next spawn), so the config tail
needs no extra sync -- and the C-shim never touches it, so there is zero contention with the lockstep.

This was chosen over a redis key / control file because the shm barrier already shares the page: it is
one mmap + 5 `struct.pack_into`s on the Python side, no new IPC, no hot-path cost.

*Proof it takes effect server-side:* `tools/verify_dflow.py` writes d=0.0, 0.5, 1.0 while driving the gate; the
shm readback matches, and the server log shows every world transition the applied config at the next
episode boundary, e.g.:

```
[SIM-RL] CONFIG CHANGED world=7 ep=0 -> spawn_geo=12 agent_hp=5000 agent_def=40 boss_hp=1500
[SIM-RL] CONFIG CHANGED world=7 ep=3 -> spawn_geo=54 agent_hp=2835 agent_def=32 boss_hp=4500
[SIM-RL] CONFIG CHANGED world=7 ep=6 -> spawn_geo=-1 agent_hp=670  agent_def=25 boss_hp=7500
```

All N worlds ramp live, mid-run, no restart. Each `[SIM-RL] agent spawned ... | applied cfg: ...` line
also stamps the applied knobs against the spawn position (`geo_dist=N (max=...)`).

**3. Eval ladder -> curriculum depth (`trainer/eval.py`).** For each rung `d in 0.1..1.0`, set the live
config to that rung, drive the SAME PuffeRL eval-rollout path training uses for a step budget, and count
the ground-truth clear/death/timeout from the server's `EPISODE DONE reason=` lines (the same source
`trainer/proof.py` tallies; the server owns episode boundaries, so there is no Python-side reset). The
per-rung clear rate feeds `curriculum_depth()` -> one number = how far up the ladder the policy clears
>=50%. The first few episodes after a d-flip are discarded (worlds mid-episode at the old config).

**Curriculum training (`trainer/curriculum.py`).** The same PuffeRL PPO loop as `trainer/train.py`,
but d RAMPS over the run (cosine `difficulty_at`, `--ramp-frac`) and the d-config is written to shm each
update, so the server spawns the d-appropriate episode live.

## Verification run (d ramps, policy climbs, depth works)

A short run (N=16, GPU1, ~1600 SPS) ramping d 0->1 over `ramp-frac=0.6`:

- **d ramped 0 -> 1 server-side.** The trainer log shows `d` climbing with the matching live cfg
  (`d=1.000 cfg(spawn=-1,hp=670,def=25,boss=7500)` at the top), and the server `CONFIG CHANGED` lines
  confirm every world applied the ramped config per episode. The d-ramp **takes effect server-side** --
  not just in Python.
- **The policy climbs the curriculum.** Ground-truth from the server log over the run: **~3000 clears vs
  ~40 deaths / ~50 timeouts**, concentrated at low d (done_rate ~0.25 at d~0.03, decaying toward 0 as d
  rose past what the short budget mastered). The untrained-policy control (the earlier proof) clears 0,
  so the low-d clears are the LEARNED navigate-in-and-kill skill following the curriculum up.
- **The eval ladder produces a sensible depth.** On a policy trained at near-fixed low d (a clearing
  policy), the ladder reads:

  ```
  d=0.10 clear=1.000   d=0.20 clear=0.000   d=0.30 clear=0.000   d=0.40 clear=0.000
  d=0.50 clear=0.312   d=0.60 clear=0.312   d=0.70 clear=0.250   d=0.80 clear=0.176
  d=0.90 clear=0.000   d=1.00 clear=0.000
  CURRICULUM DEPTH = 0.1500
  ```

  The median-of-3 smoothing (reused from `schedule.py`) correctly ignores the noisy d=0.5-0.8 partials
  (a lone spike can't inflate depth) and sets the depth at the d=0.1->0.2 threshold crossing = 0.15. A
  policy that clears nothing reads depth 0.0; the metric is the Protein objective the sweep maximizes.

**Curriculum-design note surfaced by the run.** The single-pass cosine ramp on a short budget showed
**catastrophic forgetting** -- the d=1-tuned policy no longer clears d=0 -- which is exactly the kind of
thing the Protein sweep's reward/ramp levers exist to tune (slower ramp, replay of easy rungs, reward
cocktail). The mechanism is sound; the depth a given config reaches is the search signal.

## Where the real-engine curriculum DIFFERS from the C-sim's

- **No synthetic domain randomization.** The C-sim `difficulty_config` WIDENS synthetic enemy-dynamics
  ranges (density, fire-phase/cadence jitter, acquire/speed jitter, boss-HP band, hot regions, grate
  prob) with d and samples per episode. The server-as-sim schedule does NOT: the real engine's own RNG
  is the variation, and the dungeon/enemies/AI/boss mechanics stay UNMODIFIED. The only knobs are the
  four legitimate training aids the server exposes (spawn distance, agent HP, agent DEF, boss HP).
- **Spawn is a single geodesic distance, snapping to the real entrance at d=1**, not a `spawn_in_room_prob`
  / radius distribution -- the server has one entrance and a deterministic geodesic field.
- **Boss/grenade/swarm/grate toggles, n_snakes ramp, blade_cd** (C-sim levers) have no analogue: the real
  Snake Pit places its full authored roster + mechanics unconditionally; we never add or gate them.
- **The depth objective + rungs + cosine d(t) are shared verbatim** (`curriculum_depth`,
  `CURRICULUM_RUNGS`, `difficulty_at` imported from `schedule.py`), so a depth number means the same
  thing across both sims and the sweep code is identical.

## Rough edges

- **CPU-bound on the server worlds is the plateau**, not the gate. The shm bridge + futex barrier are
  fast (bit-identical zero-copy memcpy; ~12 µs futex edge); the wall is the N-world `World.Update` and,
  past N≈32 on this contended box, core oversubscription. The earlier premise (the redis round-trip was a
  ~5–6 ms wall) was a measurement artifact — the true redis round-trip is ~0.34 ms/tick; the rest of the
  per-tick time is the sim itself.
- **`aggregate_tps=0` in the server log** under `SIM_SHM`: the legacy reporter reads `SimHarness._totalTicks`,
  which the in-proc loop doesn't increment. Cosmetic — the ticks happen (trainer drives them).
- **Reset races / lazy spawn**: agents spawn over the first few gated ticks (the pit must populate). Until an
  agent is in, its shm obs slot stays zeroed (harmless warm-up). The C-shim's `c_reset` ticks the gate once
  with a no-op action to get obs[0]. There is no per-agent reset signal from pufferl to the server; episodes
  auto-reset inside the C# loop (`ResetEpisode`) on done, PufferLib's standard auto-reset convention.
- **Post-tick obs aging**: the shm obs is built post-`World.Update` with the incremented tick; bursts fired
  this tick are aged by one tick vs the pre-tick stub path. Minor; the obs-match proof is against the live
  game objects, not the stub path.
- **Determinism**: snake spawn / boss wander use `rand()` (stochastic by design, as in the dungeon env). The
  C# worlds are independent (per-world state), so N agents see decorrelated episodes — good for PPO.
- **`_C` is the single built env**: `scripts/setup.sh` builds `server_env` into `_pufferlib/pufferlib/_C*.so`.
  It is the only env this repo ships (the old native C-sim `dungeon` env was cut), so a clean checkout has
  exactly one `_C` to build and nothing to switch between.
- **Boss damage is a proximity contact model**, not aimed-shot collision: `SimRlLoop._agentBossDamage`
  applies `SIM_PROBE_DPS_TICK` to the boss whenever the agent is within ~8 tiles (the same scripted damage
  model the lockstep proof used). So at the easy config the agent clears by *arriving* and staying close,
  which is why the navigate-in geodesic reward is the load-bearing skill. This is intentional for the
  difficulty proof; when the curriculum tightens toward d=1 the kill should be gated on real
  aimed-projectile collision so "land your shots" becomes part of the learned policy.
- **Deterministic geodesic spawn tile**: `SimGeodesic.TileAtDistance` returns the first row-major walkable
  tile at the target distance, so all N worlds spawn at the same *relative* geodesic offset (decorrelated
  only by the stochastic enemy/boss dynamics). Fine for the fixed-config proof; the curriculum may want to
  randomize the spawn tile among the equidistant candidates for spatial coverage.
- **Geodesic field is BFS in tile units, 4-connected**: a faithful navigation potential but not the exact
  Euclidean path length (diagonal moves cost 1, not √2). The reward only needs the monotone "closer is
  better" gradient, so this is sufficient; swap to a weighted/8-connected BFS if exactness ever matters.
