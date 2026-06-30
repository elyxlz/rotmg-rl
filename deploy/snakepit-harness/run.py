"""The single idempotent orchestrator.

  ensure-fresh-pipeline -> ensure-client-ingame -> enter-dungeon -> start record + policy ->
  monitor until death/clear -> stop everything.

Every stage is safe to re-run: it kills its stale predecessors first and uses verification gates
(obs freshness, relative object-count jump) rather than blind sleeps to decide it succeeded.

  python run.py            full run
  python run.py --selftest non-destructive gate/logic self-check (no process killing, no driving)
"""

from __future__ import annotations

import subprocess
import sys
import time

import client
import config
import dungeon
import obs
import policy as policy_mod  # noqa: F401  (imported for the subprocess path / symmetry)
import proc
import record

PROXY_PAT = "snakepit-harness/proxy.py"
OBS_PAT = "snakepit-harness/obs.py"
POLICY_PAT = "snakepit-harness/policy.py"


# ---- stage 1: fresh obs pipeline -----------------------------------------------------------
def ensure_fresh_pipeline() -> tuple:
    """FIX 4: the obs pipeline MUST be started fresh per run. Stale framing accumulated across many
    client connections freezes the obs on an old snapshot. Kill + restart proxy and consumer, and
    confirm :2052 is free before binding (FIX 7)."""
    proc.pkill(OBS_PAT)
    proc.pkill(PROXY_PAT)
    time.sleep(1.0)
    if not proc.wait_port_free(config.PROXY_PORT):
        raise RuntimeError("proxy port %d still in use after killing old proxy" % config.PROXY_PORT)
    config.OBS_PATH.unlink(missing_ok=True)  # drop any stale obs so the freshness gate can't false-pass

    env = config.env_for_subprocess()
    left = [str(config.VENV_PY), str(config.HOME / "snakepit-harness" / "proxy.py")]
    right = [str(config.VENV_PY), str(config.HOME / "snakepit-harness" / "obs.py")]
    proxy_proc, obs_proc = proc.spawn_pipeline(left, right, config.PROXY_LOG, config.OBS_LOG, env=env)
    time.sleep(1.0)
    print("[run] fresh pipeline up (proxy pid=%d, obs pid=%d)" % (proxy_proc.pid, obs_proc.pid), flush=True)
    return proxy_proc, obs_proc


# ---- stage 4b: policy subprocess -----------------------------------------------------------
def start_policy() -> subprocess.Popen:
    proc.pkill(POLICY_PAT)
    time.sleep(0.3)
    argv = [str(config.VENV_PY), str(config.HOME / "snakepit-harness" / "policy.py")]
    p = proc.spawn(argv, config.POLICY_LOG, env=config.env_for_subprocess())
    print("[run] policy started (pid=%d)" % p.pid, flush=True)
    return p


# ---- stage 5: monitor ----------------------------------------------------------------------
def monitor() -> str:
    """Sample the obs until it goes stale (death / clear / world transition). FIX 6: liveness is the
    obs mtime, never a fixed timer."""
    start = time.time()
    while True:
        age = obs.obs_age()
        objs = obs.obs_object_count()
        elapsed = int(time.time() - start)
        if age is None or age > config.DEATH_STALE_SECS:
            print("[run] obs stale (age=%s) at t=%ds -> run over" % (age, elapsed), flush=True)
            return "ended"
        print("[run] t=%ds obs_age=%.1fs objs=%s" % (elapsed, age, objs), flush=True)
        time.sleep(config.MONITOR_POLL_SECS)


# ---- full orchestration --------------------------------------------------------------------
def main() -> int:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    proxy_proc, obs_proc = ensure_fresh_pipeline()
    recorder = None
    policy_proc = None
    try:
        if not client.ensure_ingame():
            print("[run] ABORT: could not get client in-game", flush=True)
            return 1
        if not dungeon.enter():
            print("[run] ABORT: could not enter the Snake Pit", flush=True)
            return 1
        recorder = record.start()
        policy_proc = start_policy()
        result = monitor()
        print("[run] %s" % result, flush=True)
        return 0
    finally:
        proc.stop(policy_proc)
        proc.pkill(POLICY_PAT)
        size = record.stop(recorder)
        proc.stop(obs_proc)
        proc.stop(proxy_proc)
        print("[run] cleanup done, video bytes=%d" % size, flush=True)


# ---- non-destructive self-check ------------------------------------------------------------
def selftest() -> int:
    """Dry-check the gate logic without killing anything or driving the client. Safe while a
    recording is in progress -- it only reads."""
    print("[selftest] obs path:", config.OBS_PATH)
    age = obs.obs_age()
    print("[selftest] obs_age:", age)
    print("[selftest] obs_is_fresh(<%.1fs):" % config.OBS_FRESH_SECS, obs.obs_is_fresh())
    print("[selftest] obs_object_count (from %s):" % config.OBS_LOG, obs.obs_object_count())
    vec = obs.read_obs_vector()
    print("[selftest] obs vector len:", None if vec is None else vec.size, "(expected %d)" % config.OBS_FLOATS)
    # dungeon gate arithmetic check (pure, no IO)
    for baseline in (10, 25, 0):
        threshold = max(int(baseline * config.DUNGEON_OBJ_JUMP), baseline + 1)
        print("[selftest] baseline=%d -> dungeon threshold=%d" % (baseline, threshold))
    print("[selftest] proxy port:", config.PROXY_PORT, "free now:", proc.wait_port_free(config.PROXY_PORT, timeout=0.5))
    print("[selftest] OK (no processes touched)")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
