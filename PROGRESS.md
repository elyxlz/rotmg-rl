# Progress

Current-state summary. See `GOAL.md` for the autonomous build loop, `docs/snakepit-spec.md` +
`docs/real-game-analysis.md` for the env/real-game references, and `pufferlib/README.md` for the
vendored training stack. Append new entries on top as the build advances.

## Where things stand (2026-06-27)

**Env.** The Snake Pit dynamics are a single C source, `pufferlib/ocean/dungeon/dungeon.h` (config +
map in `src/rotmg_rl/config.py` + `snakepit_map.h`). Faithful T7 Wizard fight derived from the
betterSkillys source: Staff of Destruction (90-170 with the x2.0 maxed-ATT multiplier, ~8 shots/s),
Burning Retribution spell, DEF 8 / HP 810 / MP 455, HP regen 1.54/tick (the dominant survivability
fix), boss `ReturnToSpawn` anchoring. Local 31x31 egocentric vision + a fog-of-war minimap (no
cheats). `parity` + `config_sync` tests green.

**Training (one stack, one command).** `python3 train.py --wandb` runs a Protein (cost-aware
Bayesian) hyperparameter sweep then trains the winner, as one process + one continuous wandb run, on
PufferLib 4.0's `--slowly` torch CNN. A single difficulty `d(t)` in `[0,1]` cosine-ramps over
`--ramp-frac` of training then holds at 1, driving spawn distance + threat density + boss intensity
jointly (no phase cliffs); the vecenv is re-created from the d-derived config every `--refresh` steps
while the policy + optimizer + global LR anneal persist (`_bind_vec`). All logic lives in
`src/rotmg_rl/` (`schedule`, `training`, `sweep`, `eval`, `video`); `scripts/` is shell only.

**Results.** The continuous schedule validated at **75% d=1 (full difficulty, 7500-HP boss) clear over
100 episodes** on a 12M smoke run, where the old brittle 6-phase chain cliffed to 0.00. The earlier
phase curriculum had already reached **95% full-dungeon clear** (`full_dungeon_95.pt`, 57/60 stochastic
eps: entrance spawn + 40 snakes + grenades + minions + 7500-HP 3-phase boss). The lever that broke the
~73% plateau was **gamma 0.95 -> 0.97**: the short in-room nuke wants a myopic discount, but the
~600-step navigate-then-clear is a long horizon that must value the eventual clear.

**Deploy (untouched).** The live betterSkillys bridge (`src/rotmg_rl/deploy/`) still runs the deployed
3.0-trained `full_dungeon_95.pt` via `CDungeonPolicy`. Switching it to a policy trained on the current
stack is a deliberate later step (retrain + revalidate first).

**Open question — sim-to-real fidelity.** The deploy revealed a real gap: the real Wizard spell is a
weak 360 nova (not our nuke), the real Snake Pit is a large maze with the boss far in, and real
tier-0 gear is short-range. The full-length run is on hold pending the sim-fidelity decision (if the
sim is overhauled, the env + schedule change, so validating the current schedule end to end may be
premature). Resolve fidelity, then run the full validation once.
