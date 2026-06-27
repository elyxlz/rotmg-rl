# rotmg-rl

Cold-start reinforcement learning agent that clears one ROTMG (Realm of the Mad God)
dungeon — **Snake Pit** — trained entirely in a fast custom simulator, then deployed on the
real game.

- **No supervised data / no behavioral cloning.** Pure cold-start RL.
- **PufferLib 4.0** (recurrent CNN-LSTM PPO) over a custom **C** simulator of the dungeon.
- **Sim-to-real via a shared observation schema**: the policy only ever sees a world-agnostic
  egocentric tensor, produced identically from the sim's ground truth and from the real game's
  network packets, so a sim-trained policy can drive the real client.

See [`GOAL.md`](GOAL.md) for the autonomous build loop, [`PROGRESS.md`](PROGRESS.md) for the
current state, and [`docs/snakepit-spec.md`](docs/snakepit-spec.md) +
[`docs/real-game-analysis.md`](docs/real-game-analysis.md) for the env / real-game references.

## Layout

One flat stack:

```
pufferlib/        # vendored PufferLib 4.0 (pinned, pruned); our env edited IN PLACE in ocean/dungeon/
src/rotmg_rl/     # ALL Python logic, importable: training, sweep, eval, video, schedule, csim/, deploy/
scripts/          # thin shell orchestration only: setup.sh (provision + build), box.sh (single run)
train.py          # thin entrypoint -> rotmg_rl.training.main
tests/            # CPU tier: spec-derived scenario tests + a fixed-seed golden tripwire
```

`pufferlib/ocean/dungeon/dungeon.h` is the **single source of the Snake Pit dynamics**. It compiles
into PufferLib's `_C` backend for training and, via the standalone binding in `src/rotmg_rl/csim/`,
into a numpy-only single-env wrapper for eval + POV rendering. See
[`pufferlib/README.md`](pufferlib/README.md) for the vendored stack and build.

## Usage

Training runs on a GPU box. **One command** does everything — a Protein (cost-aware Bayesian)
hyperparameter sweep, then the full continuous-difficulty schedule with the winning config, as one
process and one continuous wandb run:

```bash
python3 train.py --wandb              # sweep -> train the winner (~460M steps) -> checkpoints/curriculum/finish.pt
python3 train.py --wandb --no-sweep   # skip the sweep; train the full schedule directly with the good defaults
```

The schedule is a single difficulty `d(t)` in `[0,1]` that cosine-ramps over `--ramp-frac` of training
then holds at 1.0, driving spawn distance, threat density (snakes/grenades/minions), and boss intensity
jointly so the policy always faces slightly-harder-than-mastered (no phase cliffs). `train.py`
self-provisions on first run (`scripts/setup.sh` builds `.venv` + `build.sh dungeon` for the `_C`
backend). `--dry-run` prints the plan; sweep knobs: `--sweep-trials`, `--trial-steps`,
`--sweep-boss-hp`, `--full-steps`, `--eval-episodes`. For the sweep alone (find + print the best
config, no full run): `python -m rotmg_rl.sweep`.

```bash
# TRUE per-episode clear rate (the >=80% deliverable) of a checkpoint
.venv/bin/python -m rotmg_rl.eval --checkpoint checkpoints/curriculum/finish.pt \
    --episodes 100 --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 0.0

# Single configurable run + observability (curriculum iteration):
./scripts/box.sh train --boss-hp 7500 --n-snakes 40 --spawn-in-room-prob 1.0   # defaults to --slowly (our CNN)
./scripts/box.sh follow     # POV rollout videos -> wandb (background)
./scripts/box.sh status     # procs + wandb URL + latest metrics
./scripts/box.sh metrics    # latest per-episode (score) + per-step metrics
```

To follow training live, set `WANDB_API_KEY` (or `wandb login`) on the box.

The CPU test tier (the C single-env path + scenario/golden tests) needs no GPU:

```bash
uv run python -m rotmg_rl.csim.build   # compile the standalone eval binding
uv run pytest tests/ -q
```
