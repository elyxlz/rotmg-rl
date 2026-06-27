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

Training runs on a GPU box and is a single stack: **PufferLib 4.0** (our CNN via the `--slowly`
torch path; see `docs/pufferlib4-migration.md`). **One command** cold-starts and runs the whole
proven curriculum (passive -> shooting -> combat -> spawn-distance ramp -> gamma-0.97 finish) as one
process and **one continuous wandb run**, to the ~95% full-dungeon policy:

```bash
python3 train.py --wandb        # ~460M steps, ~2.5h on one 3090 -> checkpoints/curriculum4/finish.pt
```

`train.py` self-provisions on first run (builds `.venv4` + the 4.0 native backend via
`scripts/setup_box_puffer4.sh`), then trains. `--dry-run` prints the plan + ETA; `--smoke N` caps each
phase to N steps to test the machinery.

```bash
# TRUE per-episode clear rate (the >=80% deliverable) of the final policy
.venv4/bin/python scripts/eval_dungeon4.py --checkpoint checkpoints/curriculum4/finish.pt \
    --episodes 100 --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.0

# Single configurable stage + observability (curriculum iteration), on GPU 1 / .venv4:
./scripts/box4.sh train --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 1.0   # defaults to --slowly (our CNN)
./scripts/box4.sh follow     # POV rollout videos -> wandb (background)
./scripts/box4.sh status     # procs + wandb URL + latest metrics
./scripts/box4.sh metrics    # latest per-episode (score) + per-step metrics
```

The old PufferLib 3.x training/eval/sweep tooling is in `scripts/archive/`. To follow training live,
set `WANDB_API_KEY` (or `wandb login`) on the box.

