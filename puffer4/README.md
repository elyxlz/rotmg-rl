# puffer4 — PufferLib 4.0 integration for the dungeon env

PufferLib 4.0 is a native C/CUDA rewrite: the env compiles statically into a monolithic `_C` backend
(no Python-env / injected-policy API like 3.0). To adopt it without maintaining a fork, we pin a 4.0
commit, vendor a clone at `.pufferlib4`, and drop these files into it at setup time. See
`docs/pufferlib4-migration.md` for the full API analysis.

Pinned commit: `9a4eb87e6b58c0aa5f22affefb65c7006d384972` (branch `4.0`, "4.0 Experiments").

Files (assembled into the clone by `scripts/setup_box_puffer4.sh`):

| this dir | -> clone path | what |
|---|---|---|
| (from `src/rotmg_rl/csim/dungeon.h`) | `ocean/dungeon/dungeon.h` | env dynamics — single source of truth, `#ifdef PUFFER4` for 4.0 buffer dtypes |
| (from `src/rotmg_rl/csim/snakepit_map.h`) | `ocean/dungeon/snakepit_map.h` | baked map tables |
| `binding.c` | `ocean/dungeon/binding.c` | 4.0 Ocean binding: `#define PUFFER4`, OBS_SIZE/NUM_ATNS/ACT_SIZES/OBS_TENSOR_T, `my_init` (dict_get), `my_log` |
| `dungeon.ini` | `config/dungeon.ini` | env kwargs + LSTM/DungeonEncoder + 3.0-tuned train hparams |
| `dungeon_encoder.py` | appended to `pufferlib/models.py` | `DungeonEncoder` (CNN grid + MLP scalars), mirrors the 3.0 `CDungeonPolicy` |

The env C is shared with the 3.0 build (`src/rotmg_rl/csim/`): `dungeon.h` carries `#ifdef PUFFER4`
guards (actions/terminals are `float*` in 4.0; `num_agents`/`rng` fields added) so one file serves
both. The 3.0 default path (`scripts/train_dungeon.py`, the parity test) is unaffected.

Build + train: `scripts/setup_box_puffer4.sh` then `scripts/train_dungeon4.py`.
