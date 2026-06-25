# GOAL

Paste the block below into `/goal` to run the autonomous build loop. See
[`docs/specs/2026-06-25-rotmg-rl-design.md`](docs/specs/2026-06-25-rotmg-rl-design.md) for
the full design.

```text
/goal Build a cold-start RL agent that clears one ROTMG dungeon (Snake Pit), and run that
same policy on the real game to complete a real dungeon. Work incessantly toward this until
the Real milestone is verified, then harden.

GROUND TRUTH
- Repo (source of truth): ~/Repos/rotmg-rl  — read docs/specs/2026-06-25-rotmg-rl-design.md
  FIRST every time you lose context; it is the authoritative design.
- Remote GPU box for training: `ssh -p 62022 audiogen@81.105.49.222` (alias ripperred).
  2x RTX 3090, 32 cores, Ubuntu 22.04. gcc/git/docker present; install `uv`; no system CUDA
  toolkit needed (use PyTorch CUDA wheels). Push from local, pull on the box, train there.
- Keep PROGRESS.md updated and committed every iteration; push regularly.

HARD CONSTRAINTS (never violate)
- Cold-start RL only. NO supervised data, NO behavioral cloning, NO human demonstrations.
- PufferLib (PuffeRL: recurrent CNN-LSTM PPO). One dungeon only: Snake Pit.
- The policy sees ONLY the shared world-agnostic observation tensor (egocentric multi-channel
  grid + scalar vector). Two adapters produce it identically: sim (ground truth) and real
  game (packets via an nrelay/RealmShark fork). Never let the policy see anything
  world-specific. No raw pixels.
- Real-game testing happens on a self-hosted PRIVATE server (NR-CORE) FIRST. Do not touch
  official ROTMG servers without an explicit new instruction from me.

THE LOOP (each iteration)
1. Read the spec + PROGRESS.md. Identify the lowest unmet milestone (below).
2. Take the smallest change that advances it. Train on the GPU box when training is involved.
3. VERIFY with a concrete measurement before claiming anything — never assert success without
   evidence (a measured clear-rate number, a reviewed rollout video, a passing eval).
4. Log to Weights & Biases (reward, clear rate, episode length, curriculum stage, RND reward,
   entropy/KL) and record a greedy rollout video. Update PROGRESS.md with: milestone status,
   current clear rate, what changed, what's next. Commit + push.
5. If blocked, diagnose the root cause and continue; don't stall waiting on me unless a
   hard-constraint decision is needed.

MILESTONES (advance in order; each gates the next)
M0  Repo scaffold + uv env on the box + wandb logging working (smoke run logs a curve).
M1  PufferLib Ocean-style C sim of Snake Pit (player movement, shooting, Stheno + minion
    bullet patterns, HP, collision, geometry) at >=1M steps/sec/core, with a raylib renderer
    and a play.py viewer.
M2  Cold-start training stack: recurrent PPO + dense reward shaping + curriculum
    (stationary enemy -> moving -> minions -> boss phase 1 -> full boss -> full dungeon) +
    RND intrinsic reward + domain randomization from day one.
M3  Sim milestone: policy clears simulated Snake Pit >=90% of episodes. (verify: eval over
    >=200 episodes; review a rollout video.)
M4  Robustness milestone: >=90% clear rate across the full domain-randomization range.
M5  Deploy adapters (ALL headless on the Linux box, protocol I/O only, no GUI/display):
    NR-CORE private server up on Linux; headless nrelay fork that BOTH reads state
    (incl. EnemyShoot packets, bullets reconstructed by local sim) and SENDS actions
    (Move/PlayerShoot); observation+action adapters; a gap-measurement harness that replays
    real packet captures through the policy and checks its actions look sane.
M6  Real milestone (DONE): the same policy completes a real Snake Pit on the private server.
    Expect 1-2 iterations of measure-gap-on-real-data -> fix sim fidelity -> retrain.

After M6: harden (raise clear rate, reduce variance, document repro) and stop opening new
scope. Report status concisely each iteration; only ping me for a hard-constraint decision
(e.g. going to official servers) or when M6 is verifiably done.
```
