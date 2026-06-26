# PufferLib 4.0 migration

Status: **adopted and verified**, with one important nuance about which backend gives the speedup
(see §4). Our env trains end-to-end on 4.0 without a fork (we pin a 4.0 commit and vendor a clone we
assemble our env into). The headline numbers:

- **Native `_C` backend: ~600K SPS (~12x over the 3.0 ~49K baseline)** — BUT it uses puffernet's
  built-in **flat Linear encoder**, not our CNN, and saves opaque flat-weight checkpoints.
- **`--slowly` torch backend: ~63K SPS (~1.3x over 3.0)** with **OUR `DungeonEncoder` CNN** (the
  architecture this spatial task wants) + torch-state-dict checkpoints that render to POV videos.

The CNN path (`--slowly`) is the **daily driver** (correct architecture + full observability);
`scripts/box4.sh` + `follow_along4.py` wire it up with wandb + rollout videos like the 3.0 box.sh.
Validated on the box: it learns and **clears** (eval rollouts `cleared=True`). The 3.0 path
(`scripts/train_dungeon.py`) stays as a fallback.

We also implemented our CNN **natively in the `_C` backend** to chase "12x WITH the CNN" (§7): it's
parity-verified and trains, but **12x-with-CNN turns out to be physically impossible** — the CNN's
compute (esp. the 30752->256 grid_fc GEMM, identical in torch) caps any CNN near the `--slowly` level,
not 600K. So `--slowly` remains the recommendation.

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

## 4. Two backends — the speedup is architecture-dependent (the key nuance)

4.0 has two trainer backends, and **which policy you get differs**:

| | native `_C` (default) | `--slowly` (torch) |
|---|---|---|
| Policy | **puffernet's flat Linear encoder** (built in C from `hidden_size`/`num_layers`; ignores `[torch] encoder`) | **our `DungeonEncoder` CNN** (`pufferlib.models`, config-selected) |
| SPS (1024 envs, passive smoke) | **~600K (~12x over 3.0)** | **~63K (~1.3x over 3.0)** |
| Checkpoint | flat float32 dump of `master_weights` (opaque) | torch state_dict (renderable, warm-startable) |
| Build | any | needs `--float` (`build.sh dungeon --float`) |

So the ~12x is **the native path with a flat encoder, not our CNN**. The native trainer builds its
own C policy (`src/puffernet.h`, a Linear encoder + LSTM) from `hidden_size`/`num_layers` and ignores
the `[torch] encoder = DungeonEncoder` setting — that only feeds the `--slowly` torch path. For a
spatial bullet-dodging task the CNN matters, so **`--slowly` is the daily driver** despite being only
modestly faster than 3.0; native is a fast-experimentation / sweep option that trades the CNN away.

Measured (GPU 1, `.venv4`, passive boss `--boss-hp 300`, no snakes/grenades/minions, `boss_shoots=0`,
`spawn_in_room`, `ent_coef 0.02`, 1024 envs, hidden 256):

- **`--slowly` (our CNN, 8.5M params):** ~63K SPS; **trains and clears.** A 50M-step run drove
  `boss_hp_frac` 0.54 -> ~0.34 (the agent damages the boss), entropy 7.0 -> 4.2 (annealing), per-step
  `cleared` ~0.028 — the same passive-boss bootstrap signal the 3.0 CNN runs show (~0.017-0.04).
  POV eval rollouts of the 16-20M checkpoints render **`cleared=True`** (boss killed in ~25-35 steps).
- **native (flat encoder, 1.9M params):** ~565-600K SPS; also learns the passive boss
  (`boss_hp_frac` 0.54 -> 0.38 over 5M) but with the flat encoder + opaque checkpoints.
- Env stepping is ~6% of step time either way (~2.3M env-SPS on 8 OMP workers; 3.0 peaked ~3.39M with
  16 threads) — the env is not the bottleneck. All runs stayed in `.venv4` on GPU 1; the live 3.0 job
  (GPU 0) + `.venv` were untouched.

## 5. Tooling (parallel to the 3.0 box.sh)

`scripts/box4.sh` mirrors `box.sh` for 4.0, pinned to GPU 1 / `.venv4` so it never touches a 3.0 run:

- `box4.sh train --slowly --boss-hp 300 --no-snakes ... --total-timesteps 50000000` — launches via
  `scripts/train_dungeon4.py`, which maps our knobs (`--boss-hp`, `--n-snakes`,
  `--no-grenades/minions/boss-shoots`, `--spawn-in-room-prob`, `--ent-coef`, `--init-checkpoint`
  warm-start, `--total-timesteps`, `--hidden`, `--slowly`) to `puffer train dungeon --env.*/--train.*`
  overrides. Logs to wandb project **`rotmg-dungeon`** with `--wandb-group` from `$WANDB_RUN_GROUP`.
- `box4.sh follow` — `scripts/follow_along4.py` watches `checkpoints4/dungeon/<run>/` for the newest
  `<step>.bin`, reconstructs `Policy(DungeonEncoder, DefaultDecoder, LSTM)`, loads the torch
  state_dict, runs a stochastic rollout on the numpy `DungeonEnv` (whose obs flattens to the same
  `[grid, scalars]` layout the C env feeds the encoder), and logs the POV mp4 + `rollout_cleared` to
  wandb (`rollouts4-<run_id>`). Only works for `--slowly` runs (native checkpoints are opaque); refuses
  otherwise. The 4.0 checkpoint filename **is** the global step, so videos log at the true step.
- `box4.sh wait` — blocks until the run ends (2 consecutive pgrep misses), then prints final metrics
  (notifier). `box4.sh status` / `kill` mirror box.sh.

## 6. Caveats / follow-ups

- **Native-speed CNN is the open prize.** Getting both ~12x AND our CNN means porting the conv
  encoder into `src/puffernet.h` (a C/CUDA conv2d forward/backward + registering it in the native
  model). Bounded but real work; until then it's 12x-flat **or** 1.3x-CNN.
- **torch pin.** The box ran with the already-present `2.12.1+cu130` (it satisfied `>=2.9`); the script
  pins `cu128` for reproducibility. On a clean `.venv4`, confirm `torch.version.cuda` is 12.x. The
  12.4-nvcc / cu13-torch mix ran fine on driver 580, but cu128 is the correct pin.
- **Vendored clone is pinned, not tracked.** `.pufferlib4` is a frozen commit we assemble into, not a
  maintained fork. Bump via `PUFFER_REF` in `setup_box_puffer4.sh`; re-port only if the Ocean binding
  contract (`vecenv.h` / `binding.c`) changes. `DungeonEncoder` is appended to the clone's
  `pufferlib/models.py` by the setup script (source of record: `puffer4/dungeon_encoder.py`).
- **Full-boss validation still pending.** Confirmed clearing on the passive-boss bootstrap (matching
  3.0's M-progression); a full passive->shooting-boss curriculum on 4.0 hasn't been run. Keep
  `train_dungeon.py` (3.0) until it has.

## 7. Native CUDA CNN encoder — the "12x WITH the CNN" attempt

Goal: get the ~12x native speed WITH our CNN (not the flat encoder, not the ~63K `--slowly` torch
path). We implemented our `DungeonEncoder` directly in 4.0's native `_C` backend.

**What was built (`puffer4/dungeon_encoder.cu`, wired into `ocean.cu`'s `create_custom_encoder`):**
the full encoder — conv1(7->32) + conv2(32->32), k3/pad1/GELU; grid_fc(30752->256)+GELU;
scalar_fc(6->64)+GELU; fuse(320->hidden)+GELU — with **forward AND backward** (im2col/col2im padded
conv via `puf_mm` GEMM; GELU fwd/bwd; bias grads; concat/split), weight + activation registration in
4.0's allocator model. 4.0 exposes a clean per-env encoder vtable (`Encoder` struct of fn pointers +
`create_custom_encoder(env_name, enc)`); the NMMO3 encoder in `ocean.cu` is a worked conv example.

**Correctness — VERIFIED (acceptance #2).** `scripts/check_encoder_parity.py` + `puffer4/test_encoder.cu`
load identical weights into the native encoder and the torch `DungeonEncoder` and compare on a fixed
input: **forward max-abs-err 1.5e-8 (rel 1e-7); backward (all 10 weight grads) rel 1.5e-5.** It is the
exact same architecture, fully differentiated — not a flat fallback. It also **trains and learns**
natively (acceptance #1): 8.2M-param CNN, `boss_hp_frac` 0.54 -> 0.37 on the passive-boss smoke,
value/entropy finite.

**Speed — the prize is not physically attainable (acceptance #3).** Measured (GPU 1, passive boss,
1024 envs, hidden 256):

| path | encoder | SPS | note |
|---|---|---|---|
| native flat | puffernet Linear | ~600K | the 12x — but wrong architecture |
| `--slowly` torch | our CNN (cuDNN) | ~63K | renderable; recommended |
| **native (this work)** | **our CNN (im2col)** | **~20K** | parity-correct, learns; im2col-bound |

The 600K (12x) comes from the flat encoder being **~10x cheaper compute** than the CNN. The CNN's
cost — dominated by the **30752->256 grid_fc GEMM, identical in torch** — caps *any* CNN
implementation far below 600K. So **"12x WITH the CNN" is not possible**: the realistic ceiling for a
native CNN is roughly the `--slowly` level (a well-optimized native CNN might reach ~2-3x of 63K from
the fused trainer, i.e. ~130-190K, but never 600K). Our im2col version lands at ~20K — *below*
`--slowly` — because it materializes a >1GB im2col column buffer per conv (pure memory bandwidth),
which our 31x31 grid makes expensive.

**Two issues remain for native-CNN to be worth it over `--slowly`:**
1. **Speed: replace im2col with cuDNN.** `src/cudnn_conv2d.cu` already ships cuDNN conv fwd/bwd with
   fused bias+activation (the build links `-lcudnn`). Swapping our two im2col convs for cuDNN
   `ConvWeights` should remove the col-buffer bottleneck and lift native-CNN toward its ~2-3x-over-
   `--slowly` ceiling. (The grid_fc GEMM stays — it's the same in torch.)
2. **A large-minibatch numerical hazard.** Native-CNN is stable + learns at `minibatch_size <= 1024`
   but NaNs after a few updates at 4096. It is **correct under `compute-sanitizer`** (memcheck +
   serialized: finite, decreasing loss, no OOB) — a size-dependent async/uninitialized hazard
   (suspected: a cuBLAS GEMM workspace/algorithm interaction at large M/K that the serialized
   sanitizer run avoids; `cublasGemmExDense` ignores the status return). `dungeon.ini` pins
   `minibatch_size = 1024` so native-CNN is stable by default.

**Disposition.** The hard kernel work is done and proven (native conv2d forward+backward, parity-
verified, trains). But native-CNN via im2col is *slower* than `--slowly`, and the 12x is unreachable
with the CNN. **Recommendation: keep `--slowly` as the CNN daily driver** (63K, renderable, simple).
The native CNN is committed as a correct foundation; finishing it (cuDNN convs + the minibatch fix)
is the path to a ~2-3x-over-`--slowly` win, if/when that margin is worth the opaque-checkpoint +
maintenance cost. Files: `puffer4/dungeon_encoder.cu`, `puffer4/test_encoder.cu`,
`scripts/check_encoder_parity.py`.
