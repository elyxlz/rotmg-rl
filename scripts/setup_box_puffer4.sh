#!/usr/bin/env bash
# Provision PufferLib 4.0 (native C/CUDA rewrite) in a SEPARATE venv (.venv4) with OUR dungeon env
# compiled into its _C backend. Does NOT touch the working .venv (PufferLib 3.0) or any running 3.0
# job. We pin a 4.0 commit and vendor a clone we extend in place (NOT a long-lived maintained fork).
#
# What it does: clone pinned PufferLib -> .venv4 (torch>=2.9 + deps) -> copy our env into the clone's
# ocean/dungeon/ + config/dungeon.ini + a DungeonEncoder into pufferlib/models.py -> build.sh dungeon
# (compiles _C with our env statically linked) -> pip install -e the clone. Then train with
# scripts/train_dungeon4.py (or `cd .pufferlib4 && .venv4/bin/puffer train dungeon`).
#
# SAFETY: torch>=2.9 + a CUDA build is several GB + heavy I/O. Run with disk headroom. This script
# never runs `uv cache clean` (a cache-clean during a live run crashed a job before) — free space
# manually ONLY when `pgrep -f train_dungeon` is empty.
#
# Usage: scripts/setup_box_puffer4.sh
set -euo pipefail
export UV_LINK_MODE=copy
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PUFFER_REF="${PUFFER_REF:-9a4eb87e6b58c0aa5f22affefb65c7006d384972}"   # pinned 4.0 commit (branch HEAD 2026-06-22)
PUFFER_DIR="${PUFFER_DIR:-$REPO_ROOT/.pufferlib4}"
VENV4="$REPO_ROOT/.venv4"
VENV_PY="$VENV4/bin/python"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"   # box: nvcc lives here, not on PATH
export CUDA_HOME NVCC_ARCH="${NVCC_ARCH:-sm_86}" # 2x RTX 3090 = compute 8.6

command -v clang >/dev/null || { echo "ERROR: clang not found (build.sh needs it)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: no nvcc at $CUDA_HOME/bin (set CUDA_HOME)"; exit 1; }
command -v ccache >/dev/null || echo "WARN: ccache not found; build.sh uses it (slower without, not fatal)"

avail_g=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
echo "Free space on $REPO_ROOT: ${avail_g}G"
[ "$avail_g" -lt 12 ] && { echo "ERROR: <12G free; a torch>=2.9 + CUDA build needs headroom. Free space (no job running) first."; exit 1; }

# 1. Pinned PufferLib clone (vendored, GitHub-only; PyPI is still 3.0.0).
if [ ! -d "$PUFFER_DIR/.git" ]; then
    git clone --branch 4.0 https://github.com/PufferAI/PufferLib "$PUFFER_DIR"
fi
( cd "$PUFFER_DIR" && git fetch --depth 1 origin "$PUFFER_REF" && git checkout -q "$PUFFER_REF" )
echo "PufferLib pinned at $(cd "$PUFFER_DIR" && git rev-parse --short HEAD)"

# 2. Separate venv (.venv4) — never the working .venv. 4.0 pyproject deps; torch>=2.9 cu12.
uv venv "$VENV4"
uv pip install --python "$VENV_PY" "torch>=2.9" numpy pybind11 setuptools rich rich_argparse gpytorch scikit-learn wandb

# 3. Assemble our env into the clone (single source of truth: env .h files copied from csim/).
mkdir -p "$PUFFER_DIR/ocean/dungeon"
cp "$REPO_ROOT/src/rotmg_rl/csim/dungeon.h" "$PUFFER_DIR/ocean/dungeon/dungeon.h"          # -DPUFFER4 set inside binding.c
cp "$REPO_ROOT/src/rotmg_rl/csim/snakepit_map.h" "$PUFFER_DIR/ocean/dungeon/snakepit_map.h"
cp "$REPO_ROOT/puffer4/binding.c" "$PUFFER_DIR/ocean/dungeon/binding.c"
cp "$REPO_ROOT/puffer4/dungeon.ini" "$PUFFER_DIR/config/dungeon.ini"
# DungeonEncoder -> pufferlib/models.py (idempotent: only append once)
if ! grep -q "class DungeonEncoder" "$PUFFER_DIR/pufferlib/models.py"; then
    printf '\n\n' >> "$PUFFER_DIR/pufferlib/models.py"
    cat "$REPO_ROOT/puffer4/dungeon_encoder.py" >> "$PUFFER_DIR/pufferlib/models.py"
fi

# 4. Build _C with the dungeon env statically linked (default = CUDA backend).
( cd "$PUFFER_DIR" && PATH="$VENV4/bin:$CUDA_HOME/bin:$PATH" ./build.sh dungeon )

# 5. Install the package (the _C*.so is already built in place; no rebuild).
uv pip install --python "$VENV_PY" --no-build-isolation -e "$PUFFER_DIR"

# 6. Smoke: the native backend imports and reports our env baked in.
( cd "$PUFFER_DIR" && "$VENV_PY" -c "from pufferlib import _C; print('puffer4 _C OK, env =', getattr(_C, 'env_name', '?'))" )
echo
echo "Done. Train our env on 4.0:"
echo "  $REPO_ROOT/.venv4/bin/python scripts/train_dungeon4.py --total-timesteps 2000000"
echo "or directly:  ( cd $PUFFER_DIR && $VENV4/bin/puffer train dungeon )"
