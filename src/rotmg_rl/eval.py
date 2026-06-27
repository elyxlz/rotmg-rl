"""TRUE per-episode clear-rate evaluation for a trained dungeon policy (the >=80% deliverable metric,
NOT the per-step `cleared` rate the dashboard shows). Runs N stochastic episodes on the single-env C
wrapper (the same C dynamics training uses) and reports the per-episode clear fraction.

`eval_clear_rate` is the in-process objective the training loop + Protein sweep observe; `main` is the
standalone entry for a saved checkpoint:

    python -m rotmg_rl.eval --checkpoint checkpoints/curriculum/finish.pt \
        --episodes 100 --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.0
"""

from __future__ import annotations

import argparse

import numpy as np
import pufferlib.models as models  # ty: ignore[unresolved-import]  pufferlib is pip-installed only on the GPU box
import torch  # ty: ignore[unresolved-import]  torch is a GPU-box-only dep, not installed on this CPU dev box

from rotmg_rl.config import DungeonConfig
from rotmg_rl.csim.single import OBS_SIZE, CDungeonSingle
from rotmg_rl.schedule import BOSS_HP, N_SNAKES_MAX, difficulty_config

ACT_SIZES = [9, 32, 2, 2]  # MultiDiscrete: move, aim, shoot, cast


def build_policy(hidden: int, num_layers: int, device) -> torch.nn.Module:
    """Reconstruct the --slowly torch policy: Policy(DungeonEncoder, DefaultDecoder, LSTM)."""
    encoder = models.DungeonEncoder(OBS_SIZE, hidden)
    decoder = models.DefaultDecoder(ACT_SIZES, hidden)
    network = models.LSTM(hidden, num_layers=num_layers)
    return models.Policy(encoder, decoder, network).to(device)


def eval_clear_rate(
    policy, episodes: int, d: float = 1.0, boss_hp: float = BOSS_HP, n_snakes_max: int = N_SNAKES_MAX, seed0: int = 50_000
) -> float:
    """TRUE per-episode clear rate at difficulty d (default full) on the single-env C wrapper -- the
    sweep objective. Eval uses the deterministic full-difficulty spawn (always entrance) at d=1."""
    device = next(policy.parameters()).device
    cfg_kw = difficulty_config(d, n_snakes_max)
    cfg_kw["spawn_in_room_prob"] = 0.0 if d >= 1.0 else cfg_kw["spawn_in_room_prob"]
    cfg_kw["n_snakes_jitter"] = 0  # eval the nominal density, not the training band
    cfg = DungeonConfig(boss_hp_max=boss_hp, **cfg_kw)
    clears = 0
    with torch.no_grad():
        for i in range(episodes):
            env = CDungeonSingle(cfg, seed=seed0 + i)
            obs = env.reset(seed=seed0 + i)
            state = policy.initial_state(1, device)
            for _ in range(cfg.max_steps):
                logits, _, state = policy.forward_eval(torch.tensor(obs, device=device).unsqueeze(0), state)
                action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    clears += int(info["cleared"])
                    break
            env.close()
    return clears / max(1, episodes)


@torch.no_grad()
def run_episode(policy, cfg, device, seed, max_steps):
    env = CDungeonSingle(cfg, seed=seed)
    obs = env.reset(seed=seed)
    state = policy.initial_state(1, device)
    try:
        for t in range(max_steps):
            x = torch.tensor(obs, device=device).unsqueeze(0)
            logits, _, state = policy.forward_eval(x, state)
            action = [int(torch.distributions.Categorical(logits=lg).sample()) for lg in logits]
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                return bool(info["cleared"]), t + 1, float(info["boss_hp_frac"])
        return False, max_steps, float(info["boss_hp_frac"])
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--boss-hp", type=float, default=7500.0)
    p.add_argument("--n-snakes", type=int, default=40)
    p.add_argument("--no-grenades", action="store_true")
    p.add_argument("--no-minions", action="store_true")
    p.add_argument("--no-boss-shoots", action="store_true")
    p.add_argument("--spawn-in-room-prob", type=float, default=0.0)
    p.add_argument("--random-spawn-prob", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(args.hidden, args.num_layers, device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.eval()

    cfg = DungeonConfig(
        boss_hp_max=args.boss_hp,
        n_snakes=args.n_snakes,
        enable_grenades=not args.no_grenades,
        enable_minions=not args.no_minions,
        boss_shoots=not args.no_boss_shoots,
        spawn_in_room_prob=args.spawn_in_room_prob,
        random_spawn_prob=args.random_spawn_prob,
    )
    clears, lengths, hp_left = [], [], []
    for i in range(args.episodes):
        c, n, hp = run_episode(policy, cfg, device, seed=10_000 + i, max_steps=cfg.max_steps)
        clears.append(c)
        lengths.append(n)
        hp_left.append(hp)
    rate = float(np.mean(clears))
    print(f"=== EVAL {args.checkpoint} ({args.episodes} eps, boss_hp={args.boss_hp:g}) ===")
    print(f"  CLEAR RATE: {rate:.1%}   ({sum(clears)}/{args.episodes})")
    print(f"  avg episode length: {np.mean(lengths):.0f} steps")
    print(f"  avg boss_hp_frac at end: {np.mean(hp_left):.3f}  (cleared eps end at 0)")
    print(f"  >=80% deliverable: {'MET' if rate >= 0.8 else 'not met'}")


if __name__ == "__main__":
    main()
