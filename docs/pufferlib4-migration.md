# PufferLib 4.0 migration investigation

Status: **investigated, not adopted.** 3.0 (PuffeRL) remains the working default. 4.0 is a native
rewrite that **removes the exact extensibility surface this repo is built on** (custom Python/numpy
env + custom injected torch policy through `pufferl.train(name, args, vecenv, policy)`). Adopting it
is a re-architecture (fork PufferLib, move our C env into its build, fork its model registry for our
policy), not a port. The C-env speed it buys is marginal here because our training is already
GPU/policy-bound, not env-bound. Recommendation at the bottom.

Date: 2026-06-26. Investigated against PufferLib branch `4.0` (release "4.0 Experiments", Apr 5 2026,
`__version__ = 4.0`, `pyproject` version `4.0.0`). PyPI is still 3.0.0 — 4.0 is GitHub-only.

---

## 1. What 4.0 is

A near-total native rewrite. The trainer, vectorization, and the policy forward/backward all live in
a compiled C/CUDA backend (`pufferlib/_C`), built from `src/*.cu` + `src/*.h` (`pufferlib.cu`,
`ocean.cu`, `kernels.cu`, `models.cu`, `puffernet.h`, `vecenv.h`, `bindings.cu`, `bindings_cpu.cpp`).
Python (`pufferlib/pufferl.py`, `torch_pufferl.py`, `models.py`, `sweep.py`, `selfplay.py`, `muon.py`)
is now a thin driver/dashboard + a PyTorch fallback.

The `pufferlib` Python package in 4.0 contains **only**: `__init__.py` (one line: `__version__ = 4.0`),
`models.py`, `muon.py`, `pufferl.py`, `selfplay.py`, `sweep.py`, `torch_pufferl.py`.

### Removed modules we depend on

| 3.0 symbol we use | 4.0 status |
|---|---|
| `pufferlib.PufferEnv` (base class) | **removed** from the package (`__init__` is one line) |
| `pufferlib.emulation.GymnasiumPufferEnv` | **removed** (no `emulation.py`) |
| `pufferlib.vector.make` / `.Multiprocessing` / `.Serial` / `.PufferEnv` | **removed** (no `vector.py`) |
| `pufferlib.ocean.torch.Recurrent` (LSTM wrapper) + `Policy` | **removed** (no `ocean/` py package) |
| `pufferlib.pytorch.layer_init` / `nativize_dtype` / `nativize_tensor` | **removed** (no `pytorch.py`) |
| `pufferlib.spaces.MultiDiscrete` | **removed** (no `spaces.py`) |
| `pufferl.load_config("default")` returning a flat train dict | **changed** (see below) |
| `pufferl.train(env_name, args, vecenv=, policy=)` | **removed signature** — now `train(env_name, args=None, gpus=None)` with **no `vecenv`/`policy` args** |

> Note: `examples/{puffer_env,gymnasium_env,pufferl,vectorization}.py` on the 4.0 branch still
> `import pufferlib.emulation` / `pufferlib.vector` / `pufferlib.ocean`. These examples are **stale
> 3.0 leftovers** — those modules do not exist in the 4.0 package, and the real trainer
> (`pufferl.py`, `torch_pufferl.py`) never imports them. Do not trust the examples for 4.0.

### The 4.0 trainer flow (what actually runs)

`pufferl.train(env_name, args)` -> `_train` -> `_resolve_backend` returns the compiled `_C` backend
(or `torch_pufferl.PuffeRL` when `--slowly`). Both build the vecenv natively:

```python
# torch_pufferl.PuffeRL.create_pufferl  (the --slowly fallback)
vec = _C.create_vec(args, _C.gpu)         # <- env comes from the compiled _C, NOT from Python
policy = load_policy(args, vec)            # <- built from config via pufferlib.models
```

`_resolve_backend` asserts `_C.env_name == args['env_name']` — **the env is statically compiled into
`_C` by `build.sh <env>`.** There is no API to hand the trainer a Python env or a separately built C
extension. The env is selected by `env_name`, which must match a `config/<env>.ini` and the env
compiled into `_C`.

The policy is also config-driven: `args['torch']['network'|'encoder'|'decoder']` name classes in
`pufferlib.models`, instantiated against `vec.obs_size` / `vec.act_sizes`. (The torch `PuffeRL`
constructor does accept a `policy=` object, but you still cannot get a `vec` without compiling your
env into `_C`.)

---

## 2. Install method (4.0)

GitHub-only. Two upstream paths (from the docs at puffer.ai):

- **PufferTank (recommended):** Docker or a uv bootstrap.
  - Docker: `git clone https://github.com/pufferai/puffertank && cd puffertank && ./docker.sh test`
  - uv: `curl https://raw.githubusercontent.com/PufferAI/PufferTank/refs/heads/4.0/install.sh | sh` (requires CUDA)
- **From the repo:** clone PufferLib `4.0`, create a venv with the deps, then **`./build.sh <env>`**
  to compile `pufferlib/_C*.so` with that env baked in, then `pip install -e .`.

`build.sh` is the real entry point (it is NOT a normal `pip install` — the C/CUDA backend is built
separately into the package dir, `OUTPUT="pufferlib/_C${EXT_SUFFIX}"`).

Build toolchain `build.sh` needs:
- `clang` (compiles the env C), `ccache`, `g++` (links).
- `nvcc` for the default (CUDA) build: `CUDA_HOME=$(dirname $(dirname $(which nvcc)))`, links
  `-lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn`. cudnn/nccl are auto-detected
  from torch's bundled `nvidia.cudnn` / `nvidia.nccl` if not on the system.
- `raylib 5.5` (auto-downloaded by `build.sh` for rendering).
- Python deps (`pyproject`): `torch>=2.9`, `numpy`, `pybind11`, `setuptools`, `rich`,
  `rich_argparse`, `gpytorch`, `scikit-learn`, `wandb`. Python `>=3.10`.
- `./build.sh <env> --cpu` is a torch-only CPU fallback (no CUDA, forces float32, required for
  `--slowly`). `./build.sh all` builds every Ocean env.

Box specifics (ripbox): CUDA toolkits present at `/usr/local/cuda-12.4` (nvcc not on PATH — set
`CUDA_HOME=/usr/local/cuda-12.4`), `clang` present, 2x RTX 3090 (driver 580). Need `ccache` and to
confirm `torch>=2.9` cu12 wheels run on driver 580 (they should).

See `scripts/setup_box_puffer4.sh` for the exact, deferred box recipe.

---

## 3. Impact on our two envs

### (a) numpy `DungeonEnv` (`sim/dungeon.py`) + `DungeonPolicy` — DEAD on 4.0

This path is `GymnasiumPufferEnv(DungeonEnv)` -> `pufferlib.vector.make(...)` -> `ocean.torch.Recurrent`
-> `pufferl.train(name, args, vecenv, policy)`. **Every** one of those symbols was removed (table
above). 4.0 has no Python-env path at all: the vecenv must be a C Ocean env compiled into `_C`. The
numpy sim cannot be a 4.0 training target. It stays the **parity oracle** and the debug renderer; it
just can't train on 4.0. `scripts/train_dungeon.py` (default, non-`--c-env`) has no 4.0 equivalent.

### (b) C dungeon env (`csim/`) — portable, but it's a re-architecture, not a file port

Good news: our `csim/dungeon.h` already matches the 4.0 Ocean Env contract exactly. The 4.0 template
(`ocean/template/{template.h,binding.c}`) and ours share the same shape:

- `Log` struct of floats with `float n` as the **last** field (vec_log divides by `n`). ✓ ours matches.
- `Env` struct begins with `Log log;` then `observations`, `actions`, `rewards`, `terminals`. ✓ ours:
  `Log log; float* observations; int* actions; float* rewards; unsigned char* terminals;`.
- `c_reset(env)`, `c_step(env)`, `c_render(env)`, `c_close(env)`; `c_step` auto-resets on terminal.
  ✓ ours has all four and auto-resets.
- `binding.c`: `#include "dungeon.h"`, `#define Env Dungeon`, `#include "../env_binding.h"`, then
  `my_init` + `my_log` (+ our `my_put`/`my_get`). ✓ ours is already in this form.

So the env C itself ports in well under a day. **What does not port is the integration model:**

1. **The env must live inside a PufferLib repo clone**, not our repo. In 3.0 our env is a standalone
   Python C extension (`rotmg_rl.csim.binding`) built by our own `csim/build.py` against our vendored
   `vendor/puffer/env_binding.h`, and we drive it with our `CDungeon(pufferlib.PufferEnv)` wrapper +
   our `CDungeonPolicy` through `pufferl.train(...)`. In 4.0 the env source goes to
   `PufferLib/ocean/dungeon/{dungeon.h,binding.c}` (+ `snakepit_map.h`), gets a
   `PufferLib/config/dungeon.ini`, and is compiled into `_C` via `./build.sh dungeon`. Training then
   runs from the PufferLib clone (`puffer train dungeon` or `pufferl.train('dungeon', args)`), not
   from `rotmg-rl`. Our `csim/dungeon.py` (`CDungeon` wrapper), `csim/build.py`, and
   `vendor/puffer/env_binding.h` all become obsolete (4.0 has its own `ocean/env_binding.h` + the
   `_C` vec backend; there is no per-env Python wrapper).

2. **Our custom CNN-LSTM policy is not a stock `pufferlib.models` class.** 4.0 builds the policy from
   config (`torch.encoder/decoder/network` referencing `pufferlib.models`). To keep our architecture
   (Conv2d over the 7x31x31 grid + MLP over the 6 scalars + LSTM) we must either add a Dungeon encoder
   to `pufferlib.models` (a PufferLib fork) or fall back to a stock encoder that flattens the grid —
   which is exactly the spatial-structure loss our `CDungeonPolicy` exists to avoid.

3. **MultiDiscrete `[9,32,2,2]` action + 6733-float Box obs** must be declared the 4.0 Ocean way
   (in the env's C/registration + `.ini`) rather than via our `single_observation_space` /
   `single_action_space` on the wrapper. Mechanically small, but it moves into the fork.

Net: adopting 4.0 for the C env means **maintaining a PufferLib fork** (our env + our policy in their
tree) and tracking their experimental branch, in exchange for the native trainer.

---

## 4. SPS comparison

3.0 baseline (measured on the box, from PROGRESS.md M4):
- C env: **3.39M env-SPS** peak (full config, 1024 envs, 16 OpenMP threads), ~243K single-thread.
- End-to-end training: **~49K SPS**. Crucially, **the env is ~3% of step time** — training is
  policy/GPU-bound, not env-bound.

4.0 was **not benchmarked**: the native `_C` build was deferred (Section 6). What 4.0 would buy:
- Its headline win is removing Python overhead from env stepping + fusing the rollout/learn pipeline
  in CUDA. But we are **already not env-bound** on 3.0 (env ~3% of step time), so faster env stepping
  yields little end-to-end gain.
- The plausible 4.0 win is on the **GPU/training half** (fused native trainer, `puffernet.h` C
  policy, CUDA advantage kernel). That could raise the ~49K end-to-end SPS, but it is unmeasured and
  comes with the fork cost in Section 3, and it constrains the policy to 4.0's model registry.

An honest SPS number for 4.0-on-our-env requires the deferred build + the env/policy port. Estimating
without measuring would be a guess.

---

## 5. What works / what doesn't

Works (verified by source reading of branch `4.0`):
- 4.0's own bundled Ocean envs (`./build.sh breakout && puffer train breakout`) — the supported path.
- Our `csim/dungeon.h` is already Ocean-contract-shaped, so the **env C ports cleanly** into a fork.

Does not work / blocked:
- numpy `DungeonEnv` training on 4.0 — **unsupported** (no Python-env path; `emulation`/`vector` gone).
- Dropping our standalone `csim` extension into 4.0 — **unsupported**; env must compile into `_C`.
- Our `CDungeonPolicy` as-is — **unsupported** without forking `pufferlib.models`.
- `pufferl.train(name, args, vecenv, policy)` — **signature removed**; trainer owns env + policy.

---

## 6. Why the native build was deferred (safety)

The box was running a live 3.0 training job (`train_dungeon.py --c-env`, 50M steps, the `.venv`) and
`/home` was at **98% (≈50 GB free)**. A 4.0 build means a second `torch>=2.9` (multi-GB) in `.venv4`
plus a full CUDA compile of `_C` — heavy disk + CPU/GPU I/O that would contend with the running job
and risks filling the disk (which would break the live job's checkpointing). Per the task's explicit
guardrails (do not disturb the running 3.0 job or the working `.venv`, no heavy I/O on a recovering
box), the build was **deferred, not abandoned**. The architectural blocker in Section 3 stands
regardless of whether `_C` compiles, so the build would confirm timing, not unblock adoption.

To run it later when the box is idle with disk headroom: `scripts/setup_box_puffer4.sh`.

---

## 7. Recommendation: stay on 3.0 for now

Do **not** switch yet. Reasons:

1. **4.0 removes our core seam.** This project's value is a custom faithful env + a custom CNN-LSTM
   policy plugged into a generic trainer. 3.0's `pufferl.train(name, args, vecenv, policy)` gives us
   exactly that. 4.0 deletes it: env must be compiled into the monolithic `_C`, policy must come from
   `pufferlib.models`. Adopting it means **maintaining a PufferLib fork** and tracking an experimental
   branch.
2. **The speed win is marginal here.** On 3.0 our C env is already ~3% of step time; we're
   GPU/policy-bound. 4.0's native env stepping doesn't help the part that's actually our bottleneck.
3. **The numpy env path dies.** We lose the fast-iteration / oracle-as-trainable-env option entirely.
4. **It's experimental.** "4.0 Experiments", stale examples, GitHub-only, no PyPI release.

Revisit 4.0 if/when: (a) training becomes env-bound (it isn't), or (b) we're willing to fork
PufferLib to host our env + a Dungeon encoder in their model registry, or (c) 4.0 reaches a stable
release that restores a custom-env/custom-policy API. If we do adopt it, the C env port is small
(Section 3); the real work is the fork + the policy encoder + losing the numpy path.

Concrete next step if the user wants to proceed anyway: run `scripts/setup_box_puffer4.sh` on an idle
box with disk headroom, fork PufferLib, add `ocean/dungeon/` + `config/dungeon.ini` + a `DungeonNet`
encoder in `pufferlib/models.py`, `./build.sh dungeon`, then benchmark `puffer train dungeon` vs the
3.0 ~49K end-to-end SPS.
