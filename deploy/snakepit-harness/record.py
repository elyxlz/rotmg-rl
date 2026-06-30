"""Recording: start/stop a full-run ffmpeg x11grab of :99, and grab single frames.

start() is idempotent -- it kills any prior recorder of OUR output first (matched by the output path)
so re-running never stacks recorders. The recorder carries a hard -t cap (RECORD_MAX_SECS) so a
recorder we lose the handle to always self-terminates.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import config
import proc


def output_path() -> Path:
    return config.VIDEO_DIR / config.VIDEO_NAME


def start() -> subprocess.Popen:
    out = output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    proc.pkill("ffmpeg.*%s" % out.name)  # idempotent: drop any prior recorder of this file
    time.sleep(0.3)
    argv = [
        "ffmpeg", "-y",
        "-f", "x11grab",
        "-video_size", config.SCREEN_SIZE,
        "-framerate", str(config.RECORD_FPS),
        "-i", config.DISPLAY,
        "-t", str(config.RECORD_MAX_SECS),
        "-pix_fmt", "yuv420p",
        "-an",
        str(out),
    ]
    recorder = proc.spawn(argv, config.RECORD_LOG, env=config.env_for_subprocess())
    print("[record] started -> %s (cap %ds)" % (out, config.RECORD_MAX_SECS), flush=True)
    return recorder


def stop(recorder: subprocess.Popen | None) -> int:
    proc.stop(recorder)
    out = output_path()
    size = out.stat().st_size if out.exists() else 0
    print("[record] stopped, %s bytes=%d" % (out, size), flush=True)
    return size


def frame(path: str) -> bool:
    res = subprocess.run(
        ["ffmpeg", "-y", "-f", "x11grab", "-video_size", config.SCREEN_SIZE, "-i", config.DISPLAY, "-frames:v", "1", path],
        capture_output=True,
        env=config.env_for_subprocess(),
        check=False,
    )
    return res.returncode == 0
