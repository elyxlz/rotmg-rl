# PufferLib (vendored, pinned)

A pruned, vendored copy of [PufferLib](https://github.com/PufferAI/PufferLib) `4.0`, pinned at
commit `9a4eb87e6b58c0aa5f22affefb65c7006d384972`. PufferLib 4.0 is a native C/CUDA rewrite: an env
compiles statically into a monolithic `_C` backend, trained via the `PuffeRL` Python trainer. 4.0 is
GitHub-only (PyPI is still 3.0.0), so we vendor the source directly instead of pip-installing it, and
edit our env **in place** (no setup-time copy — the single-home rule that killed a stale-binding bug).

## Our env lives here

| path | what |
|---|---|
| `ocean/dungeon/dungeon.h` | the Snake Pit dynamics — the **single source of truth** for the env, `#ifdef PUFFER4` guards for 4.0 buffer dtypes. The eval binding (`src/rotmg_rl/csim/`) compiles against this same file. |
| `ocean/dungeon/snakepit_map.h` | baked map tables (regenerate with `rotmg_rl.csim.gen_map_header`) |
| `ocean/dungeon/binding.c` | the 4.0 Ocean binding: `#define PUFFER4`, `OBS_SIZE`/`NUM_ATNS`/`ACT_SIZES`/`OBS_TENSOR_T`, `my_init` (`dict_get`), `my_log` |
| `config/dungeon.ini` | env kwargs + LSTM/`DungeonEncoder` + train hparams |
| `pufferlib/models.py` (`DungeonEncoder`) | the `--slowly` torch CNN (grid + minimap + scalars), appended to upstream `models.py` |

## Build + install

`build.sh dungeon` compiles `ocean/dungeon/binding.c` into `pufferlib/_C` (CUDA backend), then
`pip install -e .` makes the package importable. `--float` (float32) is required for the `--slowly`
torch path (our CNN + renderable checkpoints); drop it for max native throughput (bf16). The box
setup script (`scripts/setup.sh`) drives this end to end (torch>=2.9, cuDNN/NCCL link shims, ccache
shim). raylib is downloaded by `build.sh` at build time (it is gitignored, not vendored).

## Pruning

Everything `build.sh dungeon` needs is kept: `pufferlib/` (python), `src/`, `config/`, `build.sh`,
and `ocean/dungeon/`. Dropped from upstream to stay lean: `resources/` (demo sprite/audio assets, the
bulk), every other `ocean/<env>/` demo env, `.git`, and `.github/`. Build artifacts (`_C*`, `*.so`,
`build/`, the downloaded `raylib-*/`) are gitignored.
