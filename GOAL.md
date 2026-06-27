# GOAL

Paste the block below into `/goal`. See `docs/real-game-analysis.md` (real mechanics) +
`docs/snakepit-spec.md` + `PROGRESS.md`.

```text
/goal DELIVER, autonomously, without stopping to ask: a full .mp4 SCREEN RECORDING of the REAL
betterSkillys game client, connected to the REAL betterSkillys server, with the RL policy
controlling the character, ENTERING and COMPLETING the Snake Pit dungeon end to end. Iterate as
long as it takes. Do NOT pause for confirmation; make the calls yourself and keep going until the
mp4 exists and a live real-server clear is verified.

GROUND TRUTH (box: ssh ripbox [multiplexed]; 2x3090, 16 cores)
- Repo: ~/Repos/rotmg-rl (read PROGRESS.md first). betterSkillys source + client vendored at
  vendor/betterSkillys (faithful-sim ground truth + the real client for M5). Real Snake Pit is a
  FIXED map (EmbeddedData_SnakePitCXML.xml), not procedural -> train on the real layout.
- Faithful sim: the C env pufferlib/ocean/dungeon/dungeon.h (single source of dynamics; config + layout in config.py).
  Local 31x31 vision (VIS 15), mouse-aim (32 dir), Wizard (staff+Spell+MP), snakes, 3-phase Stheno
  (grenades/minions/status). PuffeRL 4.0 + CNN-LSTM; single-env wrapper (csim/single.py) for eval/render.

TOOLING (use these, don't reinvent or hand-ssh):
- scripts/box.sh {kill|train <args>|follow|status|metrics} -- all box management, one clean run.
- scripts/wandb_metrics.py (box.sh metrics) -- ALWAYS read metrics via the wandb API, never
  log-grep (the rich dashboard hides boss_hp_frac/learning_rate; the API revealed both).
- train+rollout runs share a wandb group (box.sh sets WANDB_RUN_GROUP -> cfg['wandb_group']).

KEY FINDINGS (so far):
- The RL LEARNS: passive-boss bootstrap drove boss_hp_frac 0.55->0.22 (agent learns to damage).
- But LR ANNEALS to 0 over total_timesteps -- too-short runs stop learning mid-progress. Use a
  long horizon (>=50M) so LR stays alive until it clears.
- The Python env (~20K SPS) starves the GPU -> M4 (C env) is the speed unlock.

HARD CONSTRAINTS
- Cold-start RL, no demos. Latest PufferLib. Character = Wizard. Sim stays FAITHFUL (local vision).
- Deliverable is the REAL client recording, NOT the sim render. Real-game work only on the
  self-hosted betterSkillys server (never official). NR-CORE is dead.
- Keep the workspace tidy/simple/abstracted; keep PROGRESS.md + GOAL.md up to date.

THE PLAN (keep iterating each stage until it works; don't stop between stages)
M4 (PRIORITY: SPEED THE LOOP). Rewrite the env in C (PufferLib Ocean style) so it is BLAZING fast.
    The Python/numpy sim does ~20K SPS and starves the GPU; Ocean C envs do millions. Port the
    faithful sim (sim/dungeon.py) to a C Ocean env so experiments run in minutes, not hours.
    Do this EVEN IF the RL mechanics don't work well yet -- fast iteration is how we FIND the
    mechanics that work. Keep the Python sim as the reference/oracle; the C env must match it
    (same obs layout, action space, dynamics) so a policy trained in C still transfers.
M3  Train a policy that COMPLETES the faithful sim >=80% (stochastic eval). Flat cold-start does
    NOT clear (proven). Bootstrap exploration with a PASSIVE BOSS (boss_shoots=False -> learn
    aim+kill), then an ADAPTIVE CURRICULUM (weak->full boss, add threat/snakes/grenades, shift
    spawns fight->navigation). ALWAYS read run metrics via the wandb API (scripts/wandb_metrics.py),
    not log-grepping. Watch entropy (collapse at ent_coef 0.001 -> raise it) + boss_hp_frac.
    Then AUTO-TUNE hparams (ent_coef/lr/reward coefs) with PufferLib's PROTEIN sweep (CARBS
    successor: pufferl.sweep, cost-aware Bayesian over env 'score'). Needs the C env (M4) for speed
    + a dense env 'score' metric = (1-boss_hp_frac)+cleared. Stop hand-guessing hparams.
M5  Deploy to drive the REAL client: run the betterSkillys visual client headless (Xvfb), read
    live game state (intercepted packets) -> the same local observation, inject the policy's
    actions (WASD + mouse + click + spell key) into the client. Build a gap harness; refit sim if
    transfer fails, then retrain. (betterSkillys source/client vendored at vendor/betterSkillys.)
M6 = DELIVERABLE: screen-record the real client completing a real Snake Pit on the live server.
    Save the .mp4 + copy to the user's machine.

LOOP: read wandb metrics (API) -> advance the lowest unmet milestone -> VERIFY with a measured
number (wandb cleared rate, eval, reviewed recording) -> update PROGRESS.md + commit/push -> stop
old wandb runs cleanly -> continue. Never claim success without evidence. Stop only when the .mp4 exists.
```
