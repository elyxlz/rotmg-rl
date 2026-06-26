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
# Reproduce the GPU-box training env (PufferLib 3.x + the fast C env)
bash scripts/setup_box.sh

# Train a stage via the box manager (kills the prior run, logs to wandb, saves checkpoints)
./scripts/box.sh train --c-env --total-timesteps 30000000 --num-envs 1024 \
    --boss-hp 7500 --gamma 0.95 --ent-coef 1e-4 --spawn-in-room-prob 1.0
./scripts/box.sh follow              # POV rollout videos -> wandb (run in background)
./scripts/box.sh status              # procs + wandb URL + latest metrics
./scripts/box.sh metrics <run-id>    # read metrics via the wandb API (the source of truth)

# Stochastic eval: TRUE per-episode clear rate (the >=80% deliverable criterion)
uv run --extra train python scripts/eval_dungeon.py --checkpoint checkpoints/dungeon.pt \
    --c-env --episodes 200 --boss-hp 7500 --spawn-in-room-prob 1.0 --render videos/eval.mp4

# Protein hyperparameter sweep (gamma / ent_coef / reward balance)
uv run --extra train python scripts/sweep_dungeon.py --max-runs 24 --metric cleared
```

To follow training live, set `WANDB_API_KEY` (or `wandb login`) on the box; otherwise runs
log offline and can be synced later with `wandb sync`.

