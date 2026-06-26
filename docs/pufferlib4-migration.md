# PufferLib 4.0 migration

Status: **adopted and verified.** Our custom dungeon env + CNN-LSTM policy train end-to-end on
PufferLib 4.0's native C/CUDA trainer at **~600K SPS vs the 3.0 baseline ~49K (≈12x)**, metrics
flowing, without maintaining a fork (we pin a 4.0 commit and vendor a clone we assemble our env into).
The 3.0 path (`scripts/train_dungeon.py`) stays in place as a fallback until 4.0 has a long run behind
it.

Pinned PufferLib commit: `9a4eb87e6b58c0aa5f22affefb65c7006d384972` (branch `4.0`, release "4.0
Experiments"). PyPI is still 3.0.0 — 4.0 is GitHub-only.

How to build + train: `scripts/setup_box_puffer4.sh` (provisions `.venv4` + the pinned clone with our
env compiled into `_C`), then `scripts/train_dungeon4.py`. Integration files live in `puffer4/`.

---

## 1. What 4.0 is

A near-total native rewrite. The trainer, vectorization, and the policy forward/backward all live in
a compiled C/CUDA backend (`pufferlib/_C`), built from `src/*.cu` + `src/*.h` (`pufferlib.cu`,
`ocean.cu`, `kernels.cu`, `models.cu`, `puffernet.h`, `vecenv.h`, `bindings.cu`). Python
(`pufferl.py`, `torch_pufferl.py`, `models.py`) is a thin driver/dashboard + a PyTorch fallback.

The 4.0 `pufferlib` package is 7 files; **`PufferEnv`, `emulation`, `vector`, `ocean` (py),
`pytorch`, `spaces` are all removed.** `pufferl.train` lost its `vecenv=`/`policy=` args. The env is
**statically compiled into `_C`** by `build.sh <env>` and selected by `env_name`; the policy is
built from config (`[torch] network/encoder/decoder` -> classes in `pufferlib.models`).

### API breakage table (3.0 -> 4.0)

| 3.0 symbol we used | 4.0 |
|---|---|
| `pufferlib.PufferEnv`, `emulation.GymnasiumPufferEnv`, `vector.make` | removed — env compiles into `_C` |
| `pufferlib.ocean.torch.Recurrent` / `Policy` | removed — recurrence is a config-selected `pufferlib.models` network (`LSTM`/`GRU`/`MinGRU`) |
| `pufferlib.pytorch.layer_init` / `nativize_*` | removed |
| `pufferlib.spaces.MultiDiscrete` | removed — action shape is `#define ACT_SIZES` in the env's binding.c |
| `pufferl.train(env_name, args, vecenv=, policy=)` | `train(env_name, args=None, gpus=None)` — trainer owns env + policy |
| `pufferl.load_config("default")` -> flat train dict | `load_config(env_name)` reads `config/<env>.ini` merged on `config/default.ini` (nested: base/vec/env/policy/torch/train) |

> The 4.0 branch's `examples/*.py` still `import pufferlib.emulation/vector/ocean` — those are stale
> 3.0 leftovers; the real trainer (`pufferl.py`, `torch_pufferl.py`) never imports them and instead
> calls `_C.create_vec(args)`.

## 2. Install method

GitHub-only; the C/CUDA backend is built by `build.sh <env>` (NOT a plain `pip install` — it writes
`pufferlib/_C*.so` with the env statically linked). `scripts/setup_box_puffer4.sh` encodes the full
recipe; the load-bearing, non-obvious parts we had to solve on the box:

- **torch / CUDA major must match the toolkit.** The box nvcc is 12.4; the default `torch>=2.9` wheel
  is now CUDA 13 (`cu130`). We pin a CUDA 12.x wheel (`--index-url .../cu128`). (Note: in practice the
  already-present `2.12.1+cu130` satisfied `>=2.9` so uv didn't downgrade it, and the 12.4-nvcc /
  cu130-torch mix ran fine on driver 580 — but cu128 is the correct, reproducible pin.)
- **`build.sh` runs `ccache nvcc`**; the box has no ccache and no sudo, so the script drops a
  transparent `ccache` shim (`exec "$@"`) on PATH.
- **Linker can't find `-lcudnn -lnccl -lnvidia-ml`.** The pip nvidia wheels ship only versioned `.so`
  (`libcudnn.so.9`, `libnccl.so.2`) and NVML lives in `$CUDA_HOME/lib64/stubs`. The script adds
  unversioned symlinks into the wheel lib dirs and puts the stubs dir on `LIBRARY_PATH`. `build.sh`
  already rpaths the cudnn/nccl wheel dirs, so runtime loading works.
- `CUDA_HOME=/usr/local/cuda-12.4` (nvcc not on PATH), `NVCC_ARCH=sm_86` (RTX 3090), `clang` present.

## 3. How our env + policy were ported (no fork)

We pin a 4.0 commit, clone it to `.pufferlib4`, and **assemble our env into it** at setup time. Files
in `puffer4/` (see `puffer4/README.md` for the mapping):

- **Env C — one source of truth.** `src/rotmg_rl/csim/dungeon.h` (+ `snakepit_map.h`) is copied into
  the clone's `ocean/dungeon/`. It already matched the Ocean contract (`Log` of floats with `n`;
  `c_reset/c_step/c_render/c_close`; auto-reset on terminal). The only 4.0 differences are buffer
  dtypes, guarded by `#ifdef PUFFER4` (set in our binding.c): 4.0's `vecenv.h` wires `float*` action
  and terminal buffers (3.0 used `int*` / `unsigned char*`) and reads `num_agents` + `rng` (the env
  index) off the struct. c_step casts the float actions to int per dim. The 3.0 build and the parity
  test compile **without** `PUFFER4` and are byte-unchanged.
- **Binding** `puffer4/binding.c` -> `ocean/dungeon/binding.c`: `#define PUFFER4`, `OBS_SIZE`/
  `NUM_ATNS 4`/`ACT_SIZES {9,32,2,2}`/`OBS_TENSOR_T FloatTensor`, `#include "vecenv.h"`, `my_init`
  (config via `dict_get`), `my_log` (per-step boss_hp_frac/in_room/cleared/...). Replaces the 3.0
  standalone-extension binding + the vendored `env_binding.h`.
- **Policy** `puffer4/dungeon_encoder.py` -> appended to `pufferlib/models.py` as `DungeonEncoder`
  (Conv over the 7x31x31 grid + MLP over the 6 scalars, mirroring the 3.0 `CDungeonPolicy`). The
  config (`puffer4/dungeon.ini` -> `config/dungeon.ini`) selects `[torch] encoder = DungeonEncoder`,
  `network = LSTM` (stock recurrent core), `decoder = DefaultDecoder` (splits our MultiDiscrete from
  `ACT_SIZES`). The trainer logs "Detected discrete action space with 4 heads" — the [9,32,2,2] split
  is correct.

`scripts/train_dungeon4.py` is a thin launcher mapping our familiar knobs (`--boss-hp`, `--n-snakes`,
`--no-boss-shoots`, `--ent-coef`, ...) to `puffer train dungeon --env.*/--train.*` overrides.

## 4. Verification + SPS (measured on the box)

Smoke run (idle GPU 1, passive-boss config matching the live 3.0 run: `--boss-hp 300 --n-snakes 0
--no-grenades --no-minions --no-boss-shoots --spawn-in-room-prob 1.0 --ent-coef 0.02`, 1024 envs,
hidden 256, 5M steps):

- **End-to-end SPS ~565-600K**, vs the 3.0 C-env baseline **~49K** -> **≈12x faster end-to-end.**
- **Env stepping ~6% of step time** (~28ms for 1024x64 steps -> ~2.3M env-SPS on one GPU's 8 OMP
  workers; 3.0 peaked ~3.39M env-SPS with 16 threads). As on 3.0, the env is not the bottleneck — the
  win is the fused native rollout/learn pipeline (the ~49K -> ~600K jump is the GPU/training half).
- **Metrics flow** (our per-step Log): `boss_hp_frac`, `in_room`, `cleared`, `snakes`,
  `player_hp_frac`, `reward`, `perf` all render in the dashboard.
- **It learns**: over 5M steps `boss_hp_frac` fell 0.54 -> 0.38 (the agent damages the boss) and
  entropy 7.0 -> 4.9 — the same passive-boss bootstrap signal the 3.0 runs show (~0.046 cleared with
  an early policy).
- Ran entirely in `.venv4` on GPU 1; the live 3.0 job (GPU 0) and the working `.venv` were untouched.

## 5. Caveats / follow-ups

- **Long-run validation pending.** Only a 5M-step smoke is done; a full passive->full-boss run on 4.0
  to confirm it clears like 3.0 did has not been run yet (don't retire `train_dungeon.py` until then).
- **torch pin.** The box happened to run with `2.12.1+cu130`; the script pins `cu128` for
  reproducibility. If a clean `.venv4` is built, confirm `torch.version.cuda` is 12.x.
- **Vendored clone is pinned, not tracked.** `.pufferlib4` is a frozen commit we assemble into, not a
  maintained fork. To bump 4.0, change `PUFFER_REF` in `setup_box_puffer4.sh` and re-run; re-port only
  if the Ocean binding contract (`vecenv.h` / `binding.c` shape) changes.
- **Custom policy lives in the clone.** `DungeonEncoder` is appended to the clone's
  `pufferlib/models.py` by the setup script (idempotent); the source of record is
  `puffer4/dungeon_encoder.py`.
