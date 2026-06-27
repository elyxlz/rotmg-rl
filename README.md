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
torch path; see `docs/pufferlib4-migration.md`). **One command** does everything: a Protein
(cost-aware Bayesian) hyperparameter sweep, then the full continuous-difficulty schedule with the
winning config, as one process and one continuous wandb run, to the full-dungeon policy:

```bash
python3 train.py --wandb        # sweep (16 trials, reduced boss HP) -> train the winner (~460M steps)
                                # -> checkpoints/curriculum4/finish.pt
python3 train.py --wandb --no-sweep   # skip the sweep; train the full schedule directly with the good defaults
```

The schedule is a single difficulty `d(t)` in `[0,1]` that cosine-ramps over `--ramp-frac` of training
then holds at 1.0, driving spawn distance, threat density (snakes/grenades/minions), and boss intensity
jointly so the policy always faces slightly-harder-than-mastered (no phase cliffs). `train.py`
self-provisions on first run (builds `.venv4` + the 4.0 native backend via
`scripts/setup_box_puffer4.sh`). `--dry-run` prints the plan; sweep knobs: `--sweep-trials`,
`--trial-steps`, `--sweep-boss-hp`, `--full-steps`, `--eval-episodes`. For the sweep alone (find +
print the best config, no full run) use `scripts/sweep_dungeon4.py`.

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

