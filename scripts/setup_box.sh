#!/usr/bin/env bash
# Reproduce the GPU-box training env (PufferLib 3.x / PuffeRL) — the working default.
#
# For PufferLib 4.0 (native C/CUDA rewrite, investigated-but-not-adopted) use the separate
# scripts/setup_box_puffer4.sh (provisions .venv4; never touches this .venv). See
# docs/pufferlib4-migration.md for why 4.0 is not the default.
#
# PuffeRL's C advantage kernel is NOT in the prebuilt wheel for a non-default torch, so it
# must be COMPILED against the installed torch. Order matters: install torch first, then build
# pufferlib with --no-build-isolation --no-deps (so it builds against the present torch and does
# NOT re-resolve/clobber it). torch 2.8.0+cu128 works on the box driver (580.x) and both 3090s.
set -euo pipefail
export UV_LINK_MODE=copy
cd "$(dirname "$0")/.."

uv venv
uv pip install "torch==2.8.0"
uv pip install setuptools wheel cython ninja
uv pip install --no-build-isolation --no-deps "pufferlib==3.0"
uv pip install numpy gymnasium wandb imageio imageio-ffmpeg pytest
uv pip install -e . --no-deps

uv run python -c "from pufferlib import pufferl; import torch; print('stack OK', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
