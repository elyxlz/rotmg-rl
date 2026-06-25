# rotmg-rl

Cold-start reinforcement learning agent that clears one ROTMG (Realm of the Mad God)
dungeon — **Snake Pit** — trained entirely in a fast custom simulator, then deployed on the
real game.

- **No supervised data / no behavioral cloning.** Pure cold-start RL.
- **PufferLib** (recurrent CNN-LSTM PPO) over a custom C simulator of the dungeon.
- **Sim-to-real via a shared observation schema**: the policy only ever sees a world-agnostic
  egocentric tensor, produced identically from the sim's ground truth and from the real
  game's network packets, so a sim-trained policy can drive the real client.

See [`docs/specs/2026-06-25-rotmg-rl-design.md`](docs/specs/2026-06-25-rotmg-rl-design.md)
for the full design, [`GOAL.md`](GOAL.md) for the autonomous build loop, and
[`PROGRESS.md`](PROGRESS.md) for the running build log.

## Usage

Training runs on a GPU box. Install with the `train` extra (torch + pufferlib):

```bash
uv sync --extra train --group dev
```

```bash
# Single PPO run (multiprocessing over physical cores)
uv run --extra train python scripts/train.py --name run1 --boss-hp 120 \
    --total-timesteps 5000000 --num-envs 256 --num-workers 16 --backend multiprocessing

# Staged cold-start curriculum toward the full boss (the M3 gate)
uv run --extra train python scripts/curriculum.py

# Ground-truth greedy eval (M3: >=90% clear over >=200 episodes on the full boss)
uv run --extra train python scripts/eval_policy.py --checkpoint checkpoints/curr-s5.pt \
    --episodes 200 --boss-hp 600

# Headless artifacts (no GPU/display needed)
uv run python scripts/bench_env.py            # env throughput
uv run python scripts/record_episode.py       # render a scripted episode to mp4
```

To follow training live, set `WANDB_API_KEY` (or `wandb login`) on the box; otherwise runs
log offline and can be synced later with `wandb sync`.

