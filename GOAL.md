# GOAL

Paste the block below into `/goal`. See `docs/real-game-analysis.md` (real mechanics) +
`docs/snakepit-spec.md` + `PROGRESS.md`.

```text
/goal DELIVER, autonomously, without stopping to ask: a full .mp4 SCREEN RECORDING of the REAL
betterSkillys game client, connected to the REAL betterSkillys server, with the RL policy
controlling the character, ENTERING and COMPLETING the Snake Pit dungeon end to end. Iterate as
long as it takes. Do NOT pause for confirmation; make the calls yourself and keep going until the
mp4 exists and a live real-server clear is verified.

GROUND TRUTH (box: ssh -p 62022 audiogen@81.105.49.222, alias ripperred; 2x3090, 16 cores)
- Repo: ~/Repos/rotmg-rl (read PROGRESS.md + docs/ first). Env recipe: scripts/setup_box.sh.
- Real server LIVE: betterSkillys (App :8080, World :2050, Redis). Source = ground truth for the
  faithful sim. Real Snake Pit analyzed in docs/real-game-analysis.md.
- Faithful sim BUILT (sim/dungeon.py, 6 tests): local 31x31 vision (VISIBILITY_RADIUS 15, no
  global cheats), mouse-aim (32 dirs), exploration reward, snakes, Wizard (staff+Spell+MP), full
  3-phase Stheno (grenades->Confused/Petrify, Stheno Swarm minions). Trains on PufferLib 3.x
  (PuffeRL) via scripts/train_dungeon.py + CNN-LSTM policy (puffer_policy.py).

HARD CONSTRAINTS
- Cold-start RL, no demos. Latest PufferLib (PuffeRL). Character = Wizard (staff + Spell).
- The sim must stay FAITHFUL to the real game (local vision, real mechanics). The deliverable is
  the REAL client recording, NOT the sim render (sim render is only for our debugging).
- Real-game work only on the self-hosted betterSkillys server (never official). NR-CORE is dead.

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
    spawns fight->navigation). Tune rewards/curriculum until it clears. ALWAYS read run metrics
    via the wandb API (scripts/wandb_metrics.py), not log-grepping, to inform the next experiment.
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
