#!/usr/bin/env python3
"""ONE long server-as-sim CURRICULUM training run -> a clearing policy at d=1 (the real fragile Wizard,
T7 staff+spell, shared aim, no armor). The curriculum d ramps 0->1 over ramp_frac of the run, then holds
at d=1 so the policy masters the real deliverable. PERIODIC eval ladders (every --eval-every steps) log
the curriculum depth + per-rung clear rates so we can watch the policy climb; frequent checkpoints (rolling
latest + best-by-depth) keep a usable policy at any moment.

WANDB: every update logs the human-readable story -- the reward COMPONENTS (approach/boss_dmg/clear/death/
step, parsed from the enriched server EPISODE DONE line), outcomes (clear/death/timeout rate, episode
length), combat closeness (boss_hp_frac remaining at episode end), survival (agent HP at end), the
curriculum d + depth + per-rung rates, the PPO stats (losses/entropy/approx_kl/grad_norm/explained_var),
SPS, and per-GPU utilization. Metrics that need deeper C# instrumentation (enemies_killed, min-geodesic
reached, ticks-to-reach-boss, spell-casts/MP/staff-accuracy) are NOT logged here -- the server doesn't emit
them and adding per-tick counters risked the verified async loop; flagged in the report, not faked.

    SIM_SHM_PATH=... SIM_ASYNC=1 SIM_SHM_BARRIER=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
      taskset -c 28-31 python server_longrun.py --agents 48 --steps 4200000 --hidden 2048 --server-log ...
  (steps is PER AGENT; global = steps*agents. run-longrun.sh owns the server + the reward env vars.)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch  # noqa: F401  # torch first so its cudart loads before pufferlib's _C
from pufferlib import _C, pufferl
from pufferlib.pufferl import unroll_nested_dict
from pufferlib.torch_pufferl import PuffeRL, load_policy

from rotmg_rl.schedule import CURRICULUM_RUNGS
from rotmg_rl.server_difficulty import curriculum_depth, difficulty_at, server_difficulty_config
from server_shm_config import ShmConfigChannel
from server_eval import tail_episodes

# C-sim trial-11 warm-start basin (the long-run defaults) + the server-as-sim tweaks the wall analysis
# called for: rew_approach bumped (navigate-in from the entrance needs strong approach shaping) and
# gae_lambda raised toward the long-horizon credit assignment that broke the C-sim plateau.
WARM = {
    "learning_rate": 0.000595, "gamma": 0.978875, "gae_lambda": 0.99,
    "ent_coef": 0.020979, "vf_coef": 1.197946, "max_grad_norm": 1.537359,
    "minibatch_size": 1024, "num_layers": 2,
    "rew_approach": 0.07, "rew_step": -0.000586,  # rew_* reach the engine as server env vars (launcher)
}

# The ENRICHED server EPISODE DONE line (post-patch): the front (t/world/step/reason) keeps EP_RE's shape
# so the eval ladder still matches; the tail carries the telemetry this run logs.
EP_FULL_RE = re.compile(
    r"EPISODE DONE t=(?P<t>\d+) world=(?P<world>\S+) step=(?P<step>\d+) reason=(?P<reason>\w+) "
    r"ep_reward=(?P<ep_reward>[-0-9.]+) ep_steps=(?P<ep_steps>\d+) boss_hp_frac=(?P<boss_hp_frac>[-0-9.]+) "
    r"agent_hp=(?P<agent_hp>[-0-9.]+) r_approach=(?P<r_approach>[-0-9.]+) r_boss_dmg=(?P<r_boss_dmg>[-0-9.]+) "
    r"r_clear=(?P<r_clear>[-0-9.]+) r_death=(?P<r_death>[-0-9.]+) r_step=(?P<r_step>[-0-9.]+)"
)


def tail_full_episodes(path: Path, episodes: deque, stop: threading.Event) -> None:
    """Tail the server log, appending the parsed enriched EPISODE DONE dict for each completed episode."""
    for _ in range(600):
        if path.exists():
            break
        time.sleep(0.1)
    with path.open("r") as f:
        f.seek(0, 2)
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.02)
                continue
            m = EP_FULL_RE.search(line)
            if m:
                g = m.groupdict()
                episodes.append({
                    "reason": g["reason"], "ep_reward": float(g["ep_reward"]), "ep_steps": int(g["ep_steps"]),
                    "boss_hp_frac": float(g["boss_hp_frac"]), "agent_hp": float(g["agent_hp"]),
                    "r_approach": float(g["r_approach"]), "r_boss_dmg": float(g["r_boss_dmg"]),
                    "r_clear": float(g["r_clear"]), "r_death": float(g["r_death"]), "r_step": float(g["r_step"]),
                })


def episode_aggregates(window: list[dict]) -> dict:
    """Per-update episode metrics from the completed episodes in this window (empty -> {})."""
    if not window:
        return {}
    n = len(window)
    mean = lambda k: sum(e[k] for e in window) / n  # noqa: E731
    return {
        "episodes/clear_rate": sum(1 for e in window if e["reason"] == "clear") / n,
        "episodes/death_rate": sum(1 for e in window if e["reason"] == "death") / n,
        "episodes/timeout_rate": sum(1 for e in window if e["reason"] == "timeout") / n,
        "episodes/mean_length_ticks": mean("ep_steps"),
        "episodes/mean_reward": mean("ep_reward"),
        "combat/boss_hp_frac_remaining": mean("boss_hp_frac"),
        "survival/agent_hp_at_end": mean("agent_hp"),
        "reward/approach": mean("r_approach"),
        "reward/boss_dmg": mean("r_boss_dmg"),
        "reward/clear": mean("r_clear"),
        "reward/death": mean("r_death"),
        "reward/step_penalty": mean("r_step"),
        "reward/total_per_episode": mean("ep_reward"),
        "episodes/count_in_window": n,
    }


def gpu_utils() -> dict:
    """Sample per-GPU utilization (cheap nvidia-smi call). Both 3090s are reported even if one is idle."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    metrics = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            metrics[f"gpu/util_gpu{parts[0]}"] = float(parts[1])
            metrics[f"gpu/mem_mb_gpu{parts[0]}"] = float(parts[2])
    return metrics


def run_ladder_on_trainer(trainer, agents, server_log, episodes_per_rung, rungs,
                          settle_episodes=4, max_rollouts_per_rung=400):
    """Eval ladder on the warm in-process trainer: write each rung's d-config to shm, drive eval rollouts,
    tally ground-truth clears from the server log (the same EPISODE DONE lines training tails)."""
    chan = ShmConfigChannel(agents)
    episodes: deque = deque(maxlen=500_000)
    stop = threading.Event()
    tailer = threading.Thread(target=tail_episodes, args=(Path(server_log), episodes, stop), daemon=True)
    tailer.start()
    time.sleep(0.3)
    rates: dict[float, float] = {}
    try:
        for d in rungs:
            chan.write(server_difficulty_config(d))
            settle_target = len(episodes) + settle_episodes * agents
            t_settle = time.time()
            while len(episodes) < settle_target and time.time() - t_settle < 60:
                trainer.rollouts()
            mark = len(episodes)
            for _ in range(max_rollouts_per_rung):
                trainer.rollouts()
                if len(episodes) - mark >= episodes_per_rung:
                    break
            window = list(episodes)[mark:]
            n_clear = sum(1 for (_, r) in window if r == "clear")
            n_ep = len(window)
            rates[d] = n_clear / n_ep if n_ep else 0.0
    finally:
        stop.set()
        chan.close()
    return rates


PPO_LOG_KEYS = {  # pufferl trainer.log() flat keys (losses nest under 'loss/') -> our wandb names
    "SPS": "train/SPS", "loss/policy_loss": "ppo/policy_loss", "loss/value_loss": "ppo/value_loss",
    "loss/entropy": "ppo/entropy", "loss/approx_kl": "ppo/approx_kl", "loss/clipfrac": "ppo/clipfrac",
    "loss/explained_variance": "ppo/explained_variance", "loss/importance": "ppo/importance",
    "env/reward": "env/reward_mean", "env/done_rate": "env/done_rate",
}


def main() -> None:
    p = argparse.ArgumentParser(description="One long server-as-sim curriculum run -> clearing policy at d=1.")
    p.add_argument("--agents", type=int, default=48)
    p.add_argument("--steps", type=int, required=True, help="env steps PER AGENT (global = steps*agents)")
    p.add_argument("--ramp-frac", type=float, default=0.55)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--eval-every", type=int, default=18_000_000, help="global steps between eval ladders")
    p.add_argument("--server-log", required=True)
    p.add_argument("--redis-port", type=int, default=6390)
    p.add_argument("--redis-db", type=int, default=5)
    p.add_argument("--out-dir", default="checkpoints/longrun")
    p.add_argument("--episodes-per-rung", type=int, default=16)
    p.add_argument("--wandb-project", default="rotmg-server-as-sim")
    p.add_argument("--wandb-name", default=None)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = [sys.argv[0]]
    args = pufferl.load_config("server_env")
    args["vec"]["total_agents"] = a.agents
    args["vec"]["num_buffers"] = 1
    args["env"]["n_agents"] = a.agents
    args["env"]["redis_port"] = a.redis_port
    args["env"]["redis_db"] = a.redis_db
    args["train"]["total_timesteps"] = a.steps * a.agents
    for k in ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "minibatch_size"):
        args["train"][k] = WARM[k]
    args["policy"]["hidden_size"] = a.hidden
    args["policy"]["num_layers"] = WARM["num_layers"]

    import wandb
    run = wandb.init(
        project=a.wandb_project,
        name=a.wandb_name or f"longrun-h{a.hidden}-{int(time.time())}",
        config={"agents": a.agents, "steps_per_agent": a.steps, "global_steps": a.steps * a.agents,
                "hidden_size": a.hidden, "num_layers": WARM["num_layers"], "ramp_frac": a.ramp_frac,
                "eval_every": a.eval_every, **{k: WARM[k] for k in WARM}},
    )
    print(f"WANDB_URL {run.url}", flush=True)

    print(f"== LONG RUN: N={a.agents} steps/agent={a.steps} global={a.steps*a.agents} hidden={a.hidden} "
          f"layers={WARM['num_layers']} ramp_frac={a.ramp_frac} lr={WARM['learning_rate']} gamma={WARM['gamma']} "
          f"gae={args['train']['gae_lambda']} mb={WARM['minibatch_size']} eval_every={a.eval_every} ==", flush=True)

    vec = _C.create_vec(args, _C.gpu)
    policy = load_policy(args, vec)
    trainer = PuffeRL(args, vec, policy, verbose=False)

    chan = ShmConfigChannel(a.agents)
    total = a.steps * a.agents
    chan.write(server_difficulty_config(0.0))

    # tail completed episodes for the per-update reward-component + outcome telemetry.
    ep_buf: deque = deque(maxlen=2_000_000)
    ep_stop = threading.Event()
    ep_tailer = threading.Thread(target=tail_full_episodes, args=(Path(a.server_log), ep_buf, ep_stop), daemon=True)
    ep_tailer.start()

    last_log = 0.0
    last_gpu = 0.0
    updates = 0
    last_cfg = None
    next_eval = a.eval_every
    best_depth = -1.0
    history = []
    ep_mark = 0
    gpu_cache: dict = {}
    try:
        while trainer.global_step < total:
            d = difficulty_at(trainer.global_step, total, a.ramp_frac)
            cfg = server_difficulty_config(d)
            if cfg != last_cfg:
                chan.write(cfg)
                last_cfg = cfg
            trainer.rollouts()
            trainer.train()
            updates += 1

            flat = dict(unroll_nested_dict(trainer.log()))
            log = {"step": trainer.global_step, "curriculum/d": d}
            for src, dst in PPO_LOG_KEYS.items():
                if src in flat:
                    log[dst] = flat[src]
            window = list(ep_buf)[ep_mark:]
            ep_mark = len(ep_buf)
            log.update(episode_aggregates(window))
            if time.time() - last_gpu > 15.0:  # GPU util is a 5s subprocess; sample at most every 15s
                last_gpu = time.time()
                gpu_cache = gpu_utils()
            log.update(gpu_cache)
            wandb.log(log, step=trainer.global_step)

            if time.time() - last_log > 5.0 or trainer.global_step >= total:
                last_log = time.time()
                sps = flat["SPS"] if "SPS" in flat else 0.0
                print(f"[train] step={trainer.global_step} updates={updates} d={d:.3f} "
                      f"cfg(spawn={cfg['spawn_geo_dist']},hp={cfg['agent_hp']},def={cfg['agent_def']},boss={cfg['boss_hp']}) "
                      f"SPS={sps:.0f} reward={flat['env/reward'] if 'env/reward' in flat else 0:.4f} "
                      f"done_rate={flat['env/done_rate'] if 'env/done_rate' in flat else 0:.4f}", flush=True)
                trainer.save_weights(str(out / "latest.pt"))

            if trainer.global_step >= next_eval:
                next_eval += a.eval_every
                policy.eval()
                rates = run_ladder_on_trainer(trainer, a.agents, a.server_log, a.episodes_per_rung, CURRICULUM_RUNGS)
                policy.train()
                chan.write(server_difficulty_config(difficulty_at(trainer.global_step, total, a.ramp_frac)))
                last_cfg = None
                ep_mark = len(ep_buf)  # discard the eval-window episodes from the train telemetry
                depth = curriculum_depth(rates)
                d1 = rates[1.0] if 1.0 in rates else 0.0
                bars = " ".join(f"{r:.2f}" for _, r in sorted(rates.items()))
                print(f"\n=== EVAL @ step={trainer.global_step}  DEPTH={depth:.4f}  d=1_clear={d1:.3f}  rung_rates=[{bars}] ===\n", flush=True)
                eval_log = {"curriculum/depth": depth, "curriculum/d1_clear_rate": d1}
                for rung, rate in rates.items():
                    eval_log[f"ladder/clear_rate_d{rung:.1f}"] = rate
                wandb.log(eval_log, step=trainer.global_step)
                history.append({"step": trainer.global_step, "depth": depth, "d1_clear": d1,
                                "rates": {str(k): v for k, v in rates.items()}})
                (out / "eval_history.json").write_text(json.dumps(history, indent=2))
                if depth > best_depth:
                    best_depth = depth
                    trainer.save_weights(str(out / "best.pt"))
                    print(f"  >> new best depth {depth:.4f} -> best.pt", flush=True)

        trainer.save_weights(str(out / "final.pt"))
        print(f"== LONG RUN DONE: {updates} updates, {trainer.global_step} steps -> {out}/final.pt (best_depth={best_depth:.4f}) ==", flush=True)
        print("LONGRUN_COMPLETE", flush=True)
    finally:
        ep_stop.set()
        chan.close()
        trainer.close()
        wandb.finish()


if __name__ == "__main__":
    main()
