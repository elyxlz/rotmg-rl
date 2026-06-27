# CLAUDE.md

Code-hygiene conventions for rotmg-rl. Read before changing code. See `README.md` for usage and
`GOAL.md` for the autonomous build loop.

## Structure

- **`_pufferlib/`** — vendored PufferLib 4.0 (pinned, pruned). Our env is edited **IN PLACE** in
  `_pufferlib/ocean/dungeon/` (`dungeon.h`, `snakepit_map.h`, `binding.c`) + `_pufferlib/config/dungeon.ini`
  + the `DungeonEncoder` in `_pufferlib/pufferlib/models.py`. This is the env's single home — there is
  **no setup-time copy** (that duplication once caused a stale-binding bug). Don't re-introduce a
  clone-and-copy step. See `_pufferlib/README.md`.
- **`src/rotmg_rl/`** — **all** Python logic, as importable modules (`training`, `sweep`, `eval`,
  `video`, `schedule`, `config`, `csim/`, `sim/`, `deploy/`). New logic goes here, never in an
  entrypoint.
- **`scripts/`** — thin shell orchestration **only** (`setup.sh` provisions + builds, `box.sh` drives a
  single run). No business logic; entrypoints forward args. `train.py` (repo root) is a thin bootstrap
  that re-execs under `.venv` and calls `rotmg_rl.training.main`. Run the other entries as modules:
  `python -m rotmg_rl.{eval,video,sweep,puffer_cli}`.

## The env is one source of truth

`_pufferlib/ocean/dungeon/dungeon.h` is the **single source of the Snake Pit dynamics**. It compiles
both into PufferLib's `_C` training backend and (via `src/rotmg_rl/csim/`) into the numpy-only
single-env wrapper for eval/render. **Never add a second env implementation.** A `Config` field added
or removed here must be matched in the ocean binding, the eval binding, and `dungeon.ini` —
`tests/test_config_sync.py` enforces that.

## Tests

`uv run pytest tests/` is the CPU tier (no GPU/torch needed): the C single-env path plus spec-derived
scenario tests (expected values hand-computed from the betterSkillys formulas) and a fixed-seed golden
trajectory tripwire. Build the eval binding first: `uv run python -m rotmg_rl.csim.build`. Keep this
green; ship tests with logic changes.

## Style

- `ruff format` + `ruff check` + `ty` clean. Match the surrounding style (long lines are fine; mirror
  the existing files).
- **Functional Python**: plain dataclasses + free functions, no OOP / classes-with-methods (the
  thin gym-ish `CDungeonSingle` wrapper is the deliberate exception).
- Minimize comments; reserve them for non-obvious mechanics. Comments describe current behavior, not
  history.

## Workflow

- **Push to `master`** via PRs. The training box is **pull-only** (`git reset --hard origin/master`),
  so anything merged lands there on the next pull.
- **Never disturb a running training sweep.** A live run uses already-loaded code; your edits don't
  affect it until it's restarted. Work in a worktree, don't touch the box's checkpoints/logs.
- `build.sh dungeon` (driven by `scripts/setup.sh`) compiles `_C`; the `--slowly` torch-CNN path is the
  renderable training path (`box.sh` defaults to it). The GPU box is x86 + CUDA; the `_C` build and any
  training run can only be verified there, not on an aarch64/CPU dev box.
