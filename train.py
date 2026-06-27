#!/usr/bin/env python3
"""Thin entrypoint for the one-command Snake Pit training flow. All logic lives in rotmg_rl.training;
this only bootstraps the training-stack venv on first run, then re-execs itself under it.

    python3 train.py --wandb            # sweep -> train the winner (~460M steps)
    python3 train.py --wandb --no-sweep # skip the sweep; train the full schedule directly
"""

from __future__ import annotations

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent
VENV_PY = REPO / ".venv" / "bin" / "python"

# bootstrap: ensure the training stack exists, then run under .venv (torch + the vendored pufferlib)
if pathlib.Path(sys.executable).resolve() != VENV_PY.resolve():
    import subprocess

    if not VENV_PY.exists():
        print("== .venv missing -> provisioning the training stack (one-time) ==", flush=True)
        subprocess.run(["bash", str(REPO / "scripts" / "setup.sh")], check=True)
    os.execv(str(VENV_PY), [str(VENV_PY), str(REPO / "train.py"), *sys.argv[1:]])

from rotmg_rl.training import main  # noqa: E402  (must follow the bootstrap re-exec above)

main()
