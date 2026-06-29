# Server-as-sim: train PufferLib `pufferl` against the live betterSkillys C# server

PufferLib's `pufferl` PPO loop drives the throwaway C# server **unchanged** as if it were a
native Ocean env. The C# server owns the real Snake Pit dynamics, the bit-identical 9807-float
obs (`SimObsBuilder`), action apply, and reward; PufferLib only shuttles the vec-buffers across
two boundaries every step:

- **shared memory** `/dev/shm/rotmg_sim_shm` — fixed float32 layout: `[N×9807 obs][N×4 act][N rew][N done]`
  (16-byte header: magic `RTMG`, n, obs_len, n_atns). C# writes obs/reward/done, the C-shim writes actions.
- **redis lockstep gate** `sim:step:cmd` / `sim:step:ack` (sim redis 127.0.0.1:6390 db5) — one
  `LPUSH` advances **all N** C# worlds exactly one gated tick; the `ack` means the new obs are in shm.

## Components

| piece | where | what |
|---|---|---|
| N-agent + shm bridge (C#) | `rotmg-sim-server` branch `sim/server-as-sim` | `SimShmBridge.cs` (mmap region), `SimRlLoop.cs` (shm-driven loop), `RootWorldThread.cs` + `SimRunner.cs` + `SimMode.cs` (combined `SIM_SHM` mode: in-proc + step-gated), one Snake Pit world per agent slot |
| C-shim native env | `rotmg-rl` branch `sim/server-env` | `_pufferlib/ocean/server_env/{server_env.h,binding.c}` — passthrough `c_step`/`c_reset` (write actions→shm, tick the gate, read obs/reward/done←shm), `config/server_env.ini` (reuses `DungeonEncoder`) |
| training entry | `rotmg-rl` | `server_train.py` (same `PuffeRL` + `DungeonEncoder` CNN-LSTM, points at `server_env`), `verify_obs.py`, `verify_motion.py` |

`N` is configured in ONE place: `--agents N` on the trainer == `SIM_WORLDS=N` on the server. The
shm region is sized for N and the binding hard-fails on a header mismatch. `total_agents` in the
ini sizes everything (shm, N pits, encoder batch). `num_buffers` MUST be 1 (one global gate).

## Launch a training run (GPU 1; never GPU 0 = the sweep)

```bash
# 1. build the C-shim env into _C (replaces the dungeon _C; rebuild `./build.sh dungeon --float` to switch back)
cd ~/rotmg-rl/_pufferlib && CUDA_HOME=/usr/local/cuda-12.4 NVCC_ARCH=sm_86 \
  PATH="$HOME/rotmg-rl/.venv-shim:$HOME/rotmg-rl/.venv/bin:$CUDA_HOME/bin:$PATH" \
  LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$(.venv/bin/python -c 'import nvidia.cudnn,os;print(os.path.join(nvidia.cudnn.__path__[0],"lib"))'):$(.venv/bin/python -c 'import nvidia.nccl,os;print(os.path.join(nvidia.nccl.__path__[0],"lib"))')" \
  ./build.sh server_env --float

# 2. start the C# server in server-as-sim mode (N worlds, shm + gate). Isolated: port 2060, redis 6390.
cd ~/rotmg-sim-server && nohup setsid ./run-server-sim.sh 16 > /tmp/server_sim.log 2>&1 < /dev/null &

# 3. run pufferl PPO against it on GPU 1 (N must match)
cd ~/rotmg-rl && CUDA_VISIBLE_DEVICES=1 .venv/bin/python server_train.py --agents 16 --steps 200000
```

Proofs: `CUDA_VISIBLE_DEVICES=1 .venv/bin/python verify_obs.py --agents 16` (bit-identical obs),
`.venv/bin/python verify_motion.py --agents 16` (actions move the real agents).

## Measured (N agents through pufferl, GPU 1)

| N | aggregate SPS |
|---|---|
| 4 | ~1150 |
| 16 | ~2900 |

Aggregate SPS = gate-ticks/sec × N (one gate round-trip advances all N agents). The single
LPUSH/BLPOP round-trip (~5-6 ms incl. all N worlds' tick) is the wall, not the sim. Approaching
the ~22K bar wants larger N (≈64-128) or a batched/shared-memory gate (eliminate the per-tick
redis round-trip — see rough edges).

## Rough edges

- **Gate round-trip is the bottleneck**, not the sim. The shm bridge is fast (bit-identical, zero-copy
  memcpy); the redis LPUSH/BLPOP per tick caps ticks/sec. A pure-shm futex/condvar barrier (drop redis
  for the step signal, keep it only for setup) would lift the cap toward the in-process `0.84 ms/step × N`.
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
- **`_C` is shared**: building `server_env` overwrites the dungeon `_C.so`. Rebuild `./build.sh dungeon --float`
  to return to the native-sim training flow.
