#!/usr/bin/env python3
"""Protein hyperparameter sweep over the SERVER-AS-SIM env (the real betterSkillys engine), maximizing
CURRICULUM DEPTH -- how far up the difficulty ladder (d in 0.1..1.0) the curriculum-trained policy still
clears >=50%. This is the server-env analogue of rotmg_rl.sweep: it reuses that module's Protein search
space, the apply/extract hparam plumbing, and the depth objective UNCHANGED, but each trial trains+evals
against the C# server (server_trial.py) instead of the dungeon C-env.

Per trial:
  1. Protein suggests a 16-knob hparam vector (trial 1 uses the ini defaults; warm-start priors seed it).
  2. Launch the throwaway C# server in server-as-sim mode (SIM_ASYNC=1, N=48, CPU-partitioned worlds on
     0-27 / trainer on 28-31, sim redis 6390 db5, shm /dev/shm/rotmg_sweep_shm). The two WIRED reward
     knobs go in as env vars BEFORE boot: SIM_APPROACH_SCALE=rew_approach, SIM_STEP_PENALTY=-rew_step
     (sweep.py's rew_step is in [-0.003,0], the server wants a positive penalty magnitude).
  3. Run server_trial.py: curriculum-train (d ramps 0->1 over ramp_frac of the run) with the PPO/policy
     hparams, then run the eval ladder on the warm policy -> curriculum_depth.
  4. Tear the server down, observe (hparams, depth, cost=train_steps) back to Protein.

WIDENED SEARCH SPACE (boundary analysis -- these three were pinned at sweep.py's ceiling):
    hidden_size  [128, 2048]   (was [128, 1024])
    gae_lambda   [0.80, 0.997] (was [0.80, 0.99])
    rew_approach [0.0, 0.12]   (was [0.0, 0.06])
The other 13 knobs keep sweep.py's ranges.

WARM-START: csim_sweep_best_reference.json's two best C-sim configs (the lr/gamma/gae/reward basins) are
observed into Protein up front (as prior observations with a nominal cost + their reference depth) so it
searches around the known-good basins instead of cold.

REWARD-KNOB CAVEAT (one owner, honest wiring): the server-as-sim reward is composed in SimRlLoop.cs with
FIXED weights for boss-dmg(/bossHpMax), clear(+5), and death(-1); only the approach scale and the step
penalty are env-tunable. So of sweep.py's six rew_* dims, only rew_approach and rew_step actually reach
this engine. The other four (rew_boss_dmg/rew_clear/rew_speed/rew_death) remain in the Protein vector for
warm-start + dimensionality parity with the C-sim reference, but are INERT here -- rebuilding the C# server
per trial to vary those weights would defeat the real-engine faithfulness. This is the deliberate, single
documented place that fact lives.

    # dry run (verify the loop end-to-end before the 1.8-day commit):
    .venv/bin/python server_sweep.py --trials 2 --trial-steps 4000000 --launch-best-steps 0 --gpu 0
    # full sweep:
    .venv/bin/python server_sweep.py --trials 24 --trial-steps 35000000 --launch-best-steps 250000000 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pufferlib.sweep  # ty: ignore[unresolved-import]
from pufferlib import pufferl  # ty: ignore[unresolved-import]

from rotmg_rl.sweep import _hparams_from_args, build_sweep_config

REPO = Path(__file__).resolve().parent
SHM_PATH = "/dev/shm/rotmg_sweep_shm"
REDIS_PORT = 6390
REDIS_DB = 5
SRV_CPUS = "0-27"      # worlds on the 28 cores (the throughput bottleneck) -- the benchmarked partition
TRAINER_CPUS = "28-31"  # the GPU-bound trainer on the remaining 4
TORCH_THREADS = "4"
RESULT_RE = re.compile(r"TRIAL_RESULT_JSON (\{.*\})")
SERVER_READY_TIMEOUT = 120.0


def widen_space(cfg: dict) -> dict:
    """Apply the three boundary-analysis widenings in place (the rest keep sweep.py's ranges)."""
    cfg["policy"]["hidden_size"] = {"distribution": "uniform_pow2", "min": 128, "max": 2048, "mean": 512, "scale": "auto"}
    cfg["train"]["gae_lambda"] = {"distribution": "logit_normal", "min": 0.80, "max": 0.997, "mean": 0.90, "scale": "auto"}
    cfg["env"]["rew_approach"] = {"distribution": "uniform", "min": 0.0, "max": 0.12, "mean": 0.03, "scale": "auto"}
    return cfg


def seed_args_from_space(args: dict, cfg: dict) -> None:
    """Seed every swept knob in args to its space mean. The server_env ini carries the PPO knobs but NOT
    ramp_frac (schedule-only) nor the env reward dims, and Protein's suggest/from_dict needs every swept
    key present in args before the first suggest. Trial 1 then runs at these means (== the ini defaults
    for PPO/policy, the space mean for ramp_frac + the reward dims)."""
    for section in ("train", "policy", "env"):
        for k, space in cfg[section].items():
            args[section][k] = space["mean"]


def launch_server(n: int, hp: dict, log_path: Path, gpu: int) -> subprocess.Popen:
    """Boot the throwaway server in server-as-sim mode for one trial. The two wired reward knobs go in as
    env vars before boot. Returns the Popen (its own process group, so we can kill the whole tree)."""
    env = dict(os.environ)
    env.update({
        "SIM_SHM": "1", "SIM_ASYNC": "1", "SIM_SHM_BARRIER": "0",
        "SIM_SHM_PATH": SHM_PATH,
        "SIM_STEP_REDIS_PORT": str(REDIS_PORT), "SIM_STEP_REDIS_DB": str(REDIS_DB),
        "SIM_SERVER_NICE": "0", "SIM_SERVER_CPUS": SRV_CPUS,
        # episodes must really end (clear/death/timeout) so the ladder can tally clears.
        "SIM_RL_INVULN": "0",
        # the two WIRED reward knobs from the trial hparams.
        "SIM_APPROACH_SCALE": f"{float(hp['rew_approach']):.6f}",
        "SIM_STEP_PENALTY": f"{abs(float(hp['rew_step'])):.6f}",
        # episode hard cap (matches run-learn.sh's proof config).
        "SIM_EP_TIMEOUT": "1500",
        # difficulty knobs are driven LIVE over shm by the curriculum (ShmConfigChannel); the env values
        # here are only the fallback before the first config write -- seed them at the easy anchor.
        "SIM_AGENT_HP": "5000", "SIM_AGENT_DEF": "40", "SIM_SPAWN_GEO_DIST": "12", "SIM_BOSS_HP": "1500",
    })
    srv = Path.home() / "rotmg-sim-server"
    logf = log_path.open("w")
    proc = subprocess.Popen(
        ["bash", str(srv / "run-server-sim.sh"), str(n)],
        stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, start_new_session=True, cwd=str(srv),
    )
    return proc


def wait_server_ready(log_path: Path, deadline: float) -> bool:
    """Server is ready once the shm region exists and the launcher logged the region line (== run-learn.sh)."""
    while time.time() < deadline:
        if Path(SHM_PATH).exists() and log_path.exists() and "region '" in log_path.read_text(errors="ignore"):
            time.sleep(2.0)  # let worlds spawn + bind settle
            return True
        time.sleep(0.5)
    return False


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the server process group, then sweep any stray WorldServer + free the shm."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    subprocess.run(["pkill", "-9", "-f", "WorldServer.dll"], check=False)
    time.sleep(1.0)
    Path(SHM_PATH).unlink(missing_ok=True)


def run_trial(i: int, hp: dict, n: int, steps: int, gpu: int, out_dir: Path) -> dict | None:
    """Boot a server, run the train+eval worker, parse the depth result. Returns the result dict
    (depth/rates/train_sps/train_steps) or None on failure."""
    srv_log = out_dir / f"server_trial{i}.log"
    trn_log = out_dir / f"trainer_trial{i}.log"
    ckpt = out_dir / f"trial{i}.pt"
    print(f"[trial {i}] launching server (N={n}, approach={hp['rew_approach']:.4f}, step_pen={abs(hp['rew_step']):.5f})", flush=True)
    server = launch_server(n, hp, srv_log, gpu)
    try:
        if not wait_server_ready(srv_log, time.time() + SERVER_READY_TIMEOUT):
            print(f"[trial {i}] SERVER FAILED TO READY -- see {srv_log}", flush=True)
            return None

        env = dict(os.environ)
        env.update({
            "SIM_SHM_PATH": SHM_PATH, "SIM_ASYNC": "1", "SIM_SHM_BARRIER": "0",
            "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": TORCH_THREADS,
        })
        cmd = [
            "taskset", "-c", TRAINER_CPUS,
            str(REPO / ".venv" / "bin" / "python"), str(REPO / "server_trial.py"),
            "--agents", str(n), "--steps", str(steps),
            "--hparams", json.dumps(hp), "--server-log", str(srv_log),
            "--redis-port", str(REDIS_PORT), "--redis-db", str(REDIS_DB),
            "--out", str(ckpt),
        ]
        with trn_log.open("w") as tf:
            proc = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=tf, stderr=subprocess.STDOUT)
        text = trn_log.read_text(errors="ignore")
        m = None
        for line in text.splitlines():
            mm = RESULT_RE.search(line)
            if mm:
                m = mm
        if proc.returncode != 0 or m is None:
            tail = "\n".join(text.splitlines()[-25:])
            print(f"[trial {i}] TRIAL FAILED rc={proc.returncode} -- tail:\n{tail}", flush=True)
            return None
        return json.loads(m.group(1))
    finally:
        kill_tree(server)


def main() -> None:
    p = argparse.ArgumentParser(description="Protein sweep over server-as-sim -> curriculum depth.")
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--trial-steps", type=int, default=35_000_000, help="env steps PER AGENT per trial")
    p.add_argument("--launch-best-steps", type=int, default=250_000_000, help="full-train the winner this many steps (0 to skip)")
    p.add_argument("--agents", type=int, default=48)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--episodes-per-rung", type=int, default=12)
    p.add_argument("--out-dir", default="checkpoints/server_sweep")
    a = p.parse_args()

    out = REPO / a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = [sys.argv[0]]  # keep pufferl.load_config's argparse off our flags
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["train"]["ramp_frac"] = 0.6  # schedule-only knob; seed it so trial 1 (defaults) carries it through observe

    sweep_config = widen_space(build_sweep_config())
    seed_args_from_space(args, sweep_config)  # every swept knob present before suggest/observe
    method = sweep_config.pop("method")
    sweep_obj = getattr(pufferlib.sweep, method)(sweep_config)

    # ---- WARM-START: observe the two best C-sim reference configs so Protein searches their basins.
    ref_path = REPO / "csim_sweep_best_reference.json"
    ref = json.loads(ref_path.read_text())
    warm = [ref[k] for k in ("trial5", "trial11") if k in ref]
    for prior in warm:
        warm_args = deepcopy(args)
        for section, keys in (("train", ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "minibatch_size", "ramp_frac")),
                              ("policy", ("hidden_size", "num_layers"))):
            for k in keys:
                if k in prior:
                    warm_args[section][k] = prior[k]
        for k in ("rew_approach", "rew_boss_dmg", "rew_clear", "rew_speed", "rew_death", "rew_step"):
            if k in prior:
                warm_args["env"][k] = prior[k]
        warm_args["train"]["total_timesteps"] = a.trial_steps * a.agents
        # the reference reached cd1=0.917 each; observe that as a prior at a nominal cost.
        sweep_obj.observe(warm_args, 0.917, float(a.trial_steps * a.agents))
    print(f"== warm-started Protein from {len(warm)} C-sim reference configs (cd~0.917) ==", flush=True)

    best = {"depth": -1.0, "hp": None}
    results = []
    for i in range(a.trials):
        if i > 0:
            sweep_obj.suggest(args)
        hp = _hparams_from_args(args)
        print(f"\n=== TRIAL {i + 1}/{a.trials} === { {k: round(v, 6) for k, v in hp.items()} }", flush=True)

        t0 = time.time()
        res = run_trial(i + 1, hp, a.agents, a.trial_steps, a.gpu, out)
        cost = float(a.trial_steps * a.agents)
        if res is None:
            sweep_obj.observe(args, 0.0, max(time.time() - t0, 1.0), is_failure=True)
            continue
        depth = float(res["depth"])
        print(f"TRIAL {i + 1}: depth={depth:.4f}  SPS={res['train_sps']:.0f}  steps={res['train_steps']}", flush=True)
        sweep_obj.observe(args, depth, cost)
        results.append({"trial": i + 1, "hp": hp, **res})
        (out / "sweep_results.json").write_text(json.dumps(results, indent=2))
        if depth > best["depth"]:
            best = {"depth": depth, "hp": hp}
            (out / "best.json").write_text(json.dumps({"depth": depth, "hp": hp}, indent=2))
            print(f"  >> new best curriculum depth {depth:.4f}", flush=True)

    print(f"\n==== SWEEP DONE. best curriculum depth: {best['depth']:.4f} ====", flush=True)
    print(f"==== BEST CONFIG: {best['hp']} ====", flush=True)
    (out / "best.json").write_text(json.dumps(best, indent=2))

    if a.launch_best_steps > 0 and best["hp"] is not None:
        print(f"\n==== LAUNCH-BEST: full-training the winner at {a.launch_best_steps} steps/agent ====", flush=True)
        res = run_trial(9999, best["hp"], a.agents, a.launch_best_steps, a.gpu, out)
        if res is not None:
            print(f"==== LAUNCH-BEST DONE: depth={res['depth']:.4f} ====", flush=True)
            (out / "launch_best_result.json").write_text(json.dumps(res, indent=2))
        else:
            print("==== LAUNCH-BEST FAILED ====", flush=True)

    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
