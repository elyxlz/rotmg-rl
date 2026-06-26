# GOAL

Paste the block below into `/goal` to run the autonomous build loop. See
[`docs/specs/2026-06-25-rotmg-rl-design.md`](docs/specs/2026-06-25-rotmg-rl-design.md) and
[`PROGRESS.md`](PROGRESS.md) for design + build log.

```text
/goal Build a cold-start RL agent that ENTERS and COMPLETES the entire Snake Pit dungeon (from
the dungeon entrance, through the rooms, to killing the boss Stheno through all her phases, to
"dungeon cleared") and run it on a REAL open-source ROTMG server. Work incessantly until the
deliverable mp4 exists and a real completion on the server is verified.

THE DELIVERABLE (the definition of done)
- A full .mp4 of the BEST policy ENTERING and COMPLETING the Snake Pit dungeon end to end,
  rendered to look like the ACTUAL game (real ROTMG sprites/tiles/projectiles from the server
  assets, HP bars, etc.) -- not abstract dots. Saved in the repo and copied to the user's machine.
- Plus M6: the same policy completes a real Snake Pit on the betterSkillys server (live).

GROUND TRUTH
- Repo (source of truth): ~/Repos/rotmg-rl. Read PROGRESS.md + the spec FIRST each time.
- GPU box: `ssh -p 62022 audiogen@81.105.49.222` (alias ripperred). 2x RTX 3090, 16 physical
  cores, Ubuntu 22.04. uv at ~/.local/bin; .NET 8 SDK at ~/.dotnet; Docker usable (no sudo).
- Open server (REAL target + ground-truth behaviors): ~/rotmg-realgame/betterSkillys
  (.NET 8, builds clean). Real Snake Pit logic: source/WorldServer/logic/db/BehaviorDb.SnakePit.cs
  (Stheno: 3 phases, aimed low-count spreads + invuln gates; minions: Pit Snakes/Vipers/Pythons,
  Stheno Swarm/Pet). Assets/XMLs: source/Shared/resources.
- Keep PROGRESS.md updated + committed every iteration; push regularly.

HARD CONSTRAINTS (never violate)
- Cold-start RL only. NO supervised data / behavioral cloning / human demos.
- Use the LATEST PufferLib (3.x) and its built-in PuffeRL trainer for the learning loop (NOT a
  hand-rolled PPO loop). Recurrent (LSTM) policy.
- One dungeon: the WHOLE Snake Pit (navigation + combat + boss), not just the boss arena.
- The policy sees ONLY the shared world-agnostic observation tensor; the same tensor is produced
  by the sim and by the real server adapter. No raw pixels for the policy.
- Real-game testing uses the self-hosted betterSkillys server (open source, local). Do NOT touch
  official ROTMG servers. NR-CORE is dead (do not revisit).
- Sim fidelity: reimplement the Snake Pit FAITHFULLY from betterSkillys source (dungeon layout,
  Stheno's real phases/patterns, minions, projectile speeds from the resource XMLs). The faithful
  sim is what collapses the sim-to-real gap (the RocketSim approach).

THE LOOP (each iteration)
1. Read spec + PROGRESS.md; identify the lowest unmet milestone.
2. Smallest change that advances it. Train on the GPU box (PuffeRL, multi-core + 3090s).
3. VERIFY with a concrete measurement before claiming anything (completion-rate over >=200 eps,
   reviewed rollout video, passing eval). Stochastic action sampling is the deployment metric.
4. Log to Weights & Biases + record a rollout video. Update PROGRESS.md (milestone, current
   completion rate, what changed, next). Commit + push.
5. If blocked, diagnose root cause and continue; only ping the user for a true external blocker.

MILESTONES (advance in order; each gates the next)
M0  Stack ready: latest PufferLib (3.x) + PuffeRL smoke run logs a curve; betterSkillys server
    RUNS (Redis via Docker + resources configured); a headless client connects + reads state.
M1  Faithful Snake Pit sim from betterSkillys source: dungeon map (entrance -> rooms -> boss
    room) + navigation, real Stheno 3-phase fight, minions, projectile properties from XMLs.
    Plus a GAME-FAITHFUL renderer using the real ROTMG assets (sprites/tiles/projectiles).
    Target >=1M steps/s/core; numpy reference first if needed, then fast/C if required.
M2  Cold-start training on PuffeRL (recurrent PPO): observation covers navigation + combat;
    reward shapes whole-dungeon progress (explore -> reach boss -> clear phases -> COMPLETE);
    curriculum + intrinsic motivation for the long, sparse navigation horizon.
M3  Sim completion: policy completes the full simulated dungeon (enter -> clear) >=90%
    (stochastic, >=200 eps). Record the game-faithful completion mp4 = THE DELIVERABLE.
M4  Robustness: >=90% completion across the domain-randomization range.
M5  Deploy adapters to the betterSkillys server (headless protocol I/O): read state -> shared
    observation, send actions; gap-measurement harness (sim vs real); refit sim if needed.
M6  Real completion: the same policy enters and completes a real Snake Pit on the betterSkillys
    server, verified end to end.

REUSE FROM v1 (boss-only radial sim, now superseded): the training infra pattern, the deploy
bridge (observation schema, bullet reconstruction, policy server, RealmShark adapter), and the
gap-measurement harness all carry over. The OLD sim + policy (radial-burst, boss-only) are
superseded by the faithful whole-dungeon sim; rebuild them.

After the deliverable mp4 exists AND M6 is verified: harden, document repro, stop new scope.
```
