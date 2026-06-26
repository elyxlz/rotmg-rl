#!/usr/bin/env bash
# EXPERIMENTAL: provision PufferLib 4.0 (native C/CUDA rewrite) in a SEPARATE venv (.venv4).
#
# This does NOT touch the working .venv (PufferLib 3.0 / PuffeRL), which stays the default.
# See docs/pufferlib4-migration.md for why 4.0 is investigated-but-not-adopted: it removes the
# custom Python-env + injected-policy API this repo trains through. This script only sets up the
# 4.0 trainer + builds its _C backend so 4.0 can be benchmarked; it does NOT port our env/policy
# into it (that is a PufferLib fork — see the migration doc).
#
# RUN ONLY on an idle box with disk headroom: torch>=2.9 + a CUDA _C build is several GB and heavy
# I/O. Do not run while a 3.0 training job is using .venv, and check `df -h ~` first.
#
# Usage:
#   scripts/setup_box_puffer4.sh                 # set up 4.0 + build a bundled env (breakout) to smoke-test
#   PUFFER_ENV=breakout scripts/setup_box_puffer4.sh
set -euo pipefail
export UV_LINK_MODE=copy
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PUFFER_REF="${PUFFER_REF:-4.0}"
PUFFER_ENV="${PUFFER_ENV:-breakout}"   # a bundled Ocean env, just to verify the native backend builds + trains
PUFFER_DIR="${PUFFER_DIR:-$REPO_ROOT/.pufferlib4}"   # PufferLib clone (env compiles INTO its package)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"       # box: nvcc is here, not on PATH
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"

command -v nvcc >/dev/null || { echo "nvcc not found (set CUDA_HOME); have /usr/local/cuda-12.4?"; exit 1; }
command -v clang >/dev/null || { echo "clang not found (build.sh needs clang)"; exit 1; }
command -v ccache >/dev/null || echo "WARN: ccache not found; build.sh uses it (slower without)"

# 1. Clone PufferLib 4.0 (GitHub-only; PyPI is still 3.0.0).
if [ ! -d "$PUFFER_DIR/.git" ]; then
    git clone --depth 1 --branch "$PUFFER_REF" https://github.com/PufferAI/PufferLib "$PUFFER_DIR"
fi

# 2. Separate venv (.venv4) — never the working .venv.
uv venv .venv4
VENV_PY="$REPO_ROOT/.venv4/bin/python"
# 4.0 pyproject deps; torch>=2.9 (cu12 wheel runs on driver 580).
"$VENV_PY" -m ensurepip -U >/dev/null 2>&1 || true
uv pip install --python "$VENV_PY" "torch>=2.9" numpy pybind11 setuptools rich rich_argparse gpytorch scikit-learn wandb

# 3. Build the native _C backend with $PUFFER_ENV statically linked, then install the package.
#    build.sh writes pufferlib/_C*.so INTO the clone; default mode needs nvcc + cudnn/nccl (it
#    auto-detects torch's bundled nvidia.cudnn / nvidia.nccl).
( cd "$PUFFER_DIR" && PATH="$REPO_ROOT/.venv4/bin:$PATH" ./build.sh "$PUFFER_ENV" )
uv pip install --python "$VENV_PY" --no-build-isolation -e "$PUFFER_DIR"

# 4. Smoke test: native trainer trains a bundled env end to end.
( cd "$PUFFER_DIR" && "$VENV_PY" -c "from pufferlib import pufferl, _C; print('puffer4 OK, _C env =', getattr(_C, 'env_name', '?'))" )
echo
echo "Set up. Smoke-train with:  ( cd $PUFFER_DIR && $REPO_ROOT/.venv4/bin/puffer train $PUFFER_ENV )"
echo "To train OUR env you must port it into the clone (ocean/dungeon/ + config/dungeon.ini +"
echo "a DungeonNet encoder in pufferlib/models.py) — see docs/pufferlib4-migration.md."
