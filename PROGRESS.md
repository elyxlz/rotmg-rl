# Progress

Current-state summary. See `GOAL.md` for the autonomous build loop, `docs/snakepit-spec.md` +
`docs/real-game-analysis.md` for the env/real-game references, and `_pufferlib/README.md` for the
vendored training stack. Append new entries on top as the build advances.

## Swarm-survival fidelity fix (2026-06-28, branch `fix/swarm-survival`)

The mortal no-cheat policy self-navigated to bossDist ~52 then died in the snake swarm every run while
the sim reported 96% clears. Measured the real gap from the live bridge logs + the betterSkillys source:

- **Player HP/DEF.** The live char is **HP 670** (bridge `hp=670/670`), **DEF 25** (not the assumed
  810 / 8). The incoming-damage histogram is bimodal — a spike at exactly 25 and a cluster at 2-4,
  nothing between. With the bot's own formula `max(dmg*3/20, dmg-def)` + `toFixed(0)`, only DEF=25
  yields that split: Greater snake raw 50 -> 25, Python 25 -> 4, Pit Viper 20 -> 3, Pit Snake 10 -> 2
  (DEF 8 would show 42s and 17s, which never appear). Set sim to 670 / 25.
- **Swarm lethality = DENSITY, not per-bullet.** Every per-snake stat in `SNAKE_TYPES` already matches
  the XML exactly. The gap is count: the real `.jm` packs **405 snakes** through the maze (mostly weak
  Pit Snake/Viper fillers; ~14% lethal Greaters), and the bridge shows **~27 in view** at the death
  cluster. The sim spawned only **40** uniformly -> local density ~7, measured **3.4 dmg/tick** vs the
  real **~33/tick**. Fix: `n_snakes` 40 -> **200** (in-view density now mean ~27, matching the real
  cluster) and `SNAKE_WEIGHTS` retuned to the real .jm proportions (`[0.716,0.054,0.086,0.054,0.089]`).
  Held-still DPS rose 3.4 -> ~12/tick at the matched density; the residual vs the cornered real 33/tick
  is the stationary-vs-moving measurement floor, not per-snake under-modeling (left untouched, faithful).
- **Latency read.** The deaths are under-modeled damage, not reaction lag: HP fell from 670 to dead in
  one ~2s burst of ~29 clean 25-dmg Greater-snake hits — a density spike the old sim could never
  produce — not a few large hits the policy reacted to slowly.

Deploy stays matched: the bridge normalizes obs hp by the server `hp_max` (670), the sim by
`player_hp_max` (670). Retrain + revalidate after merge.

## Where things stand (2026-06-27)

**Env.** The Snake Pit dynamics are a single C source, `_pufferlib/ocean/dungeon/dungeon.h` (config +
map in `src/rotmg_rl/config.py` + `snakepit_map.h`). Faithful T7 Wizard fight derived from the
betterSkillys source: Staff of Destruction (90-170 with the x2.0 maxed-ATT multiplier, ~8 shots/s),
Burning Retribution spell, DEF 8 / HP 810 / MP 455, HP regen 1.54/tick (the dominant survivability
fix), boss `ReturnToSpawn` anchoring. Local 31x31 egocentric vision + a fog-of-war minimap (no
cheats). The spec-derived scenario + golden + `config_sync` tests are green.

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

**Deploy (untouched).** The live betterSkillys bridge (`src/rotmg_rl/deploy/`) still runs the
previously deployed `full_dungeon_95.pt` via `CDungeonPolicy`. Switching it to a policy trained on the
current stack is a deliberate later step (retrain + revalidate first).

**Open question — sim-to-real fidelity.** The deploy revealed a real gap: the real Wizard spell is a
weak 360 nova (not our nuke), the real Snake Pit is a large maze with the boss far in, and real
tier-0 gear is short-range. The full-length run is on hold pending the sim-fidelity decision (if the
sim is overhauled, the env + schedule change, so validating the current schedule end to end may be
premature). Resolve fidelity, then run the full validation once.
