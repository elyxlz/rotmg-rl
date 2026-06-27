"""Compile the C dungeon env into an importable extension (no raylib/torch needed).

    uv run python -m rotmg_rl.csim.build

Produces `binding.<EXT_SUFFIX>.so` next to binding.c, importable as
`rotmg_rl.csim.binding`. Rendering is a no-op stub in C, so we don't link raylib here; the
Python sim handles all debug rendering. Avoids -ffast-math so float results stay bit-faithful
to the numpy oracle (parity test).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import sysconfig

import numpy


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parents[2]
    vendor = repo_root / "vendor" / "puffer"
    dungeon_dir = repo_root / "pufferlib" / "ocean" / "dungeon"  # the single home of dungeon.h + snakepit_map.h
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    out = here / f"binding{ext_suffix}"
    py_inc = sysconfig.get_path("include")

    cmd = [
        "cc",
        "-O3",
        "-march=native",
        "-funroll-loops",
        "-fopenmp",  # parallelize the per-env step loop across cores (vec_step)
        "-ffp-contract=off",  # no FMA contraction -> float results stay bit-faithful to the numpy oracle
        "-fno-math-errno",
        "-fPIC",
        "-shared",
        "-fwrapv",
        "-fno-strict-aliasing",
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        f"-I{numpy.get_include()}",
        f"-I{py_inc}",
        f"-I{here}",
        f"-I{dungeon_dir}",  # dungeon.h + snakepit_map.h live in the vendored pufferlib tree
        f"-I{vendor}",
        str(here / "binding.c"),
        "-lm",
        "-o",
        str(out),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"built {out}", flush=True)


if __name__ == "__main__":
    main()
