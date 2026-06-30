"""Idempotent process management: start named long-lived processes, kill stale ones, check liveness.

Every start() first kills any prior instance matching the same pattern, so re-running the
orchestrator never stacks duplicate proxies / clients / recorders. Kills are always guarded so a
missing process never aborts the caller (FIX 7: pkill guarded so the script never dies on a no-op).
"""

from __future__ import annotations

import signal
import socket
import subprocess
import time
from pathlib import Path


def pkill(pattern: str) -> None:
    """Kill every process whose command line matches `pattern`. Never raises (FIX 7)."""
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)


def is_alive(pattern: str) -> bool:
    """True if at least one process matches `pattern`."""
    res = subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False)
    return res.returncode == 0


def wait_port_free(port: int, timeout: float = 5.0) -> bool:
    """Block until `port` can be bound (i.e. the old listener is gone). Returns True if freed.

    FIX 7: the proxy must not race a dying predecessor for :2052 -- confirm the port is free first.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            probe.close()
            return True
        except OSError:
            probe.close()
            time.sleep(0.2)
    return False


def spawn(argv: list[str], log_path: Path, env: dict | None = None, cwd: str | None = None) -> subprocess.Popen:
    """Start a detached background process, both stdout+stderr -> log_path (truncated). Idempotency is
    the caller's job (kill the matching pattern first); this only launches."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "wb")
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)


def spawn_pipeline(left: list[str], right: list[str], left_log: Path, right_log: Path, env: dict | None = None) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Start `left | right`: left's stdout pipes into right's stdin; each gets its own stderr log.

    This is the proxy (tee server->client to stdout) -> obs consumer (frame + build obs) pipe.
    """
    left_log.parent.mkdir(parents=True, exist_ok=True)
    llog = open(left_log, "wb")
    rlog = open(right_log, "wb")
    left_proc = subprocess.Popen(left, stdout=subprocess.PIPE, stderr=llog, env=env, start_new_session=True)
    right_proc = subprocess.Popen(right, stdin=left_proc.stdout, stdout=subprocess.DEVNULL, stderr=rlog, env=env, start_new_session=True)
    left_proc.stdout.close()  # right_proc owns the read end; let the pipe close when left exits
    return left_proc, right_proc


def stop(proc: subprocess.Popen | None, grace: float = 2.0) -> None:
    """Terminate a Popen we hold a handle to, then SIGKILL the group if it lingers. Never raises."""
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        proc.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        pass
