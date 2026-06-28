#!/usr/bin/env bash
# Provision the training stack in .venv with our dungeon env compiled into PufferLib 4.0's _C backend.
# PufferLib 4.0 is vendored in-repo at _pufferlib/ (our env edited IN PLACE in ocean/dungeon/) — this
# builds it from source; there is no clone or setup-time copy.
#
# What it does: .venv (torch>=2.9 + deps) -> build.sh dungeon (compiles _C with our env statically
# linked) -> pip install -e _pufferlib/ + our package. Then train with `python3 train.py` (the
# one-command flow) or `scripts/box.sh train --slowly ...` (a single configurable run).
#
# SAFETY: torch>=2.9 + a CUDA build is several GB + heavy I/O. Run with disk headroom. This script
# never runs `uv cache clean` (a cache-clean during a live run crashed a job before) — free space
# manually ONLY when `pgrep -f 'train.py|rotmg_rl'` is empty.
#
# Usage: scripts/setup.sh
set -euo pipefail
export UV_LINK_MODE=copy
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PUFFER_DIR="$REPO_ROOT/_pufferlib"   # the vendored, committed PufferLib (env edited in place)
VENV="$REPO_ROOT/.venv"
VENV_PY="$VENV/bin/python"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"   # box: nvcc lives here, not on PATH
export CUDA_HOME NVCC_ARCH="${NVCC_ARCH:-sm_86}" # 2x RTX 3090 = compute 8.6

command -v clang >/dev/null || { echo "ERROR: clang not found (build.sh needs it)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ERROR: no nvcc at $CUDA_HOME/bin (set CUDA_HOME)"; exit 1; }

# build.sh hardcodes `NVCC="ccache $CUDA_HOME/bin/nvcc"`. If ccache isn't installed (no sudo on the
# box), drop in a transparent shim that just execs its args, and prepend it to PATH for the build.
SHIM_DIR="$REPO_ROOT/.venv-shim"
if ! command -v ccache >/dev/null; then
    mkdir -p "$SHIM_DIR"
    printf '#!/bin/sh\nexec "$@"\n' > "$SHIM_DIR/ccache"
    chmod +x "$SHIM_DIR/ccache"
    echo "ccache not found -> using transparent shim at $SHIM_DIR/ccache"
fi

avail_g=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
echo "Free space on $REPO_ROOT: ${avail_g}G"
[ "$avail_g" -lt 12 ] && { echo "ERROR: <12G free; a torch>=2.9 + CUDA build needs headroom. Free space (no job running) first."; exit 1; }

# 1. Venv + 4.0 deps. torch MUST match the box CUDA toolkit major: nvcc is 12.4, so use CUDA 12.x
# wheels (cu128), NOT torch's default cu130 wheels — a CUDA-13 torch mismatches the 12.4 nvcc/cudart
# the _C build links.
[ -d "$VENV" ] || uv venv "$VENV"
uv pip install --python "$VENV_PY" "torch>=2.9" --index-url https://download.pytorch.org/whl/cu128
# Pin numpy<2 (the _C extension is built against the numpy-1.x C ABI) and scipy<1.16 — scipy>=1.18
# requires numpy>=2.0, and an unpinned resolve mixes scipy 1.18 with numpy 1.26 -> the import-time
# `numpy has no attribute 'long'` crash (training.py imports the sweep deps even under --no-sweep).
uv pip install --python "$VENV_PY" "numpy<2" "scipy<1.16" pybind11 setuptools rich rich_argparse gpytorch scikit-learn wandb

# 2. Link prerequisites: build.sh links `-lcudnn -lnccl -lnvidia-ml`, but the pip nvidia wheels ship
# only versioned .so files (no libcudnn.so / libnccl.so for `-l`) and NVML lives in the CUDA stubs
# dir. Add unversioned symlinks into the wheel lib dirs + put cudnn/nccl/stubs on the linker path.
CUDNN_LIB=$("$VENV_PY" -c "import nvidia.cudnn, os; print(os.path.join(nvidia.cudnn.__path__[0], 'lib'))")
NCCL_LIB=$("$VENV_PY" -c "import nvidia.nccl, os; print(os.path.join(nvidia.nccl.__path__[0], 'lib'))")
for so in "$CUDNN_LIB"/libcudnn.so.*; do [ -e "$CUDNN_LIB/libcudnn.so" ] || ln -sf "$(basename "$so")" "$CUDNN_LIB/libcudnn.so"; done
for so in "$NCCL_LIB"/libnccl.so.*; do [ -e "$NCCL_LIB/libnccl.so" ] || ln -sf "$(basename "$so")" "$NCCL_LIB/libnccl.so"; done
export LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$CUDNN_LIB:$NCCL_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"

# 3. Build _C with the dungeon env statically linked, in place in the vendored tree. --float (float32)
# is required for the --slowly torch backend = our CNN + renderable checkpoints, and the native
# backend still runs on it. Drop --float (set PUFFER_BUILD_FLAGS='') for max native throughput (bf16),
# but then --slowly won't work.
( cd "$PUFFER_DIR" && PATH="$SHIM_DIR:$VENV/bin:$CUDA_HOME/bin:$PATH" LIBRARY_PATH="$LIBRARY_PATH" ./build.sh dungeon ${PUFFER_BUILD_FLAGS:---float} )
# LIBRARY_PATH (CUDA-12.4 stubs) is a LINK-time aid for build.sh only; leaving it set makes torch load
# the stub cudart at runtime ("undefined symbol cudaGetDriverEntryPointByVersion"). Drop it now.
unset LIBRARY_PATH

# 4. Install the vendored package (the _C*.so is already built in place; no rebuild) + our package.
uv pip install --python "$VENV_PY" --no-build-isolation -e "$PUFFER_DIR"
uv pip install --python "$VENV_PY" -e "$REPO_ROOT" --no-deps
uv pip install --python "$VENV_PY" gymnasium imageio imageio-ffmpeg

# 4.5. Build the single-env eval binding (csim/binding.c -> rotmg_rl.csim.binding). This is separate
# from the _C training backend above; train.py imports rotmg_rl.eval, which imports this binding, so a
# clean install must compile it or train/eval fail at import.
"$VENV_PY" -m rotmg_rl.csim.build

# 5. Smoke: the full import chain (training -> eval -> csim binding + _C) loads clean. Import order
# matters: rotmg_rl.training imports torch FIRST, so torch's CUDA-12.8 cudart loads before pufferlib's
# _C (built against 12.4) — the reverse order pins the older cudart and breaks torch. This mirrors the
# real `python3 train.py` order; do NOT front-load `from pufferlib import _C`.
"$VENV_PY" -c "import rotmg_rl.training; from pufferlib import _C; print('_C OK, env =', getattr(_C, 'env_name', '?'), '| rotmg_rl.training imports clean')"
echo
echo "Done. Train the Snake Pit (use --slowly for our CNN + renderable videos):"
echo "  python3 train.py --wandb                                   # the one-command sweep -> full run"
echo "  scripts/box.sh train --slowly --boss-hp 300 --total-timesteps 50000000   # a single configurable run"
echo "  scripts/box.sh follow   # POV rollout videos to wandb"
