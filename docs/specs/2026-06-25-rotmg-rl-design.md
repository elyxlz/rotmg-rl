# Cold-start RL agent that clears one ROTMG dungeon

**Date:** 2026-06-25
**Status:** Design approved, pending spec review

## Goal

Train a reinforcement-learning policy that completes **one easy ROTMG dungeon**, then run
that same policy on the **real game** and have it clear a real instance of that dungeon.

Hard constraints (from the user):

- **One dungeon only**, starting with one of the easiest. Chosen: **Snake Pit** (boss:
  Stheno the Snake Queen). Trivially swappable later.
- **Cold-start RL. No supervised / behavioral-cloning data.** No human demonstrations.
- **PufferLib** as the RL stack.
- The end deliverable is a policy that runs on the **actual game** and completes a real
  dungeon (tested first on a self-hosted private server, then optionally official).

## The core idea (the whole project hinges on this)

The policy must operate in two worlds: the **sim** (where it trains at millions of
steps/sec) and the **real game** (where it must perform). This only works if the policy
never sees anything world-specific.

> The policy sees one abstract observation tensor. Two separate adapters produce that exact
> same tensor: one from the sim's ground truth, one from the real game's network packets.
> The policy cannot tell which world it is in.

```
            ┌─────────────── TRAIN ───────────────┐
  PufferLib C sim ──► sim adapter ──► OBSERVATION ──► policy ──► action ──► sim
                                       (shared schema)

            ┌─────────────── DEPLOY ──────────────┐
  real ROTMG client ──► protocol reader ──► REAL adapter ──► OBSERVATION ──► policy
       (nrelay / RealmShark packets)                                          │
                                                                              ▼
                                          input injector ◄── action ◄── (same policy)
                                               │
                                               ▼
                                          real client
```

If both adapters emit the same observation schema **and** the sim's dynamics (movement
speed, bullet speed/patterns, enemy behavior, tick rate) are close enough to real, the
policy transfers. That "close enough" is the central risk of the project (see Risks).

## Components

### 1. Shared observation schema (world-agnostic)

Egocentric, centered on the player:

- **Multi-channel local grid** (e.g. 32×32 around the player). Channels:
  - `walls / obstacles`
  - `enemies`
  - `enemy projectiles` (position) + a paired channel encoding projectile velocity/heading
  - `hazard zones`
  - `self`
- **Scalar vector**: HP%, MP%, ability-ready flag, boss HP%, relative direction to objective.

Rationale: (a) the proven bullet-hell representation; (b) reconstructable from real-game
packet data (projectile positions, enemy positions, HP, tiles are all in the ROTMG
protocol); (c) abstract enough that sim and real look identical. We deliberately do **not**
use raw pixels: the sim cannot render pixel-identical to the real client, so pixels would
widen the sim-to-real gap, not close it.

### 2. The sim (PufferLib Ocean-style C environment)

Faithful reimplementation of **Snake Pit** only. Models:

- Player movement physics, shooting.
- Boss (Stheno) + minion bullet patterns and HP.
- Collision, dungeon geometry.

Throughput target: ≥1M steps/sec/core. Written in C against the PufferLib Ocean env
authoring pattern.

### 3. Action space (identical in sim and real)

- Movement: 8-direction (or continuous).
- Aim direction for shooting.
- One ability key.

Kept identical across worlds so a single policy output maps to both a sim step and an
input-injector command.

### 4. Real-game adapters (deploy-time only, not needed to train)

The entire real-game loop is **headless protocol I/O** and runs on the Linux GPU box. There
is no GUI client, no screen capture, no synthetic mouse/keyboard, no display.

- **Headless client (read + write)**: a fork of `nrelay` (Node.js, runs on Linux). It is a
  real protocol-speaking client that both:
  - reads state: enemy/boss positions, HP, and `EnemyShoot` packets;
  - sends actions: `Move` and `PlayerShoot` packets driven by the policy.
- **Observation adapter**: reconstructs the bullet field by locally simulating projectiles
  from `EnemyShoot` packets (the same technique vrelay's predictive autonexus uses), then
  builds the shared `GameState` -> `build_observation` tensor, identical to the sim's.
- **Action adapter**: maps the policy's MultiDiscrete action (move dir, aim dir) to `Move` /
  `PlayerShoot` packets sent through the headless client.
- **Server**: NR-CORE private server runs headless on Linux (dotnet/mono). One open item is
  confirming its target runtime (.NET Framework via mono vs modern .NET); resolved at M5.

### 5. Visualization and progress tracking

The user wants to watch the agent and follow training progress, so this is a first-class
component, not an afterthought.

- **Training metrics → Weights & Biases**: reward curves, dungeon clear rate, episode
  length, curriculum stage, RND intrinsic reward, KL/entropy. Remote-accessible so progress
  is followable from anywhere while training runs on the GPU box. PufferLib has native wandb
  integration.
- **Rendered rollout videos**: a render mode for the Snake Pit sim (raylib, as used by
  PufferLib Ocean envs). Every N updates, record a greedy-policy episode and log it to wandb
  as video, so the agent's behavior is watchable as it evolves (does it dodge? approach the
  boss? clear?).
- **Live local viewer**: a `play.py` that loads a checkpoint and renders an episode in real
  time on demand, for eyeballing a specific checkpoint.
- **Progress log**: a `PROGRESS.md` in the repo updated each iteration (milestone reached,
  current clear rate, what changed, what is next), committed so progress is auditable.

## Training infrastructure

- **Remote GPU box** `baby-ai-ripper` (`ssh -p 62022 audiogen@81.105.49.222`, aliased
  `ripperred`): 2× RTX 3090 (24 GB), 32 CPU cores, Ubuntu 22.04, gcc + git + docker present.
- `uv` and `nvcc` are absent: install `uv` for the Python env; no system CUDA toolkit is
  needed (PyTorch CUDA wheels ship their own runtime; PufferLib speed comes from CPU C envs
  compiled with gcc across the 32 cores).
- The git repo is the source of truth; training runs on the box via ssh. Sim throughput
  target uses all 32 cores; the learner uses the 3090s.

## Cold-start training recipe (no demos)

Cold-start on a sparse "boss died" reward yields ~0 signal. With supervised data ruled out,
demonstrations are replaced by dense shaping + curriculum + intrinsic motivation, all made
viable by the unlimited sim step budget.

- **Algorithm**: recurrent PPO (CNN-LSTM) via PufferLib's PuffeRL (PPO + Muon + GAE/VTrace).
  Recurrence handles partial observability (off-screen threats).
- **Dense reward shaping**: `+` damage dealt to boss, `+` surviving ticks, `+` progress
  toward objective, large `+` for boss kill / dungeon clear; `−` damage taken, large `−`
  for death.
- **Curriculum**: stationary single enemy → moving enemy → minions → boss phase 1 → full
  boss → full dungeon. Each stage unlocks only when the prior is solved.
- **Intrinsic motivation (RND)**: novelty bonus so the agent explores and finds the boss
  rather than camping a safe corner.
- **Domain randomization from day one**: randomize bullet speeds, enemy HP, movement
  constants, tick timing, spawn positions, plus small observation noise. Double duty:
  prevents overfitting to sim quirks **and** is the primary sim-to-real defense (a policy
  robust to randomized dynamics tolerates the gap to the real game).

## Success criteria

1. **Sim milestone**: policy clears simulated Snake Pit ≥90% of episodes.
2. **Robustness milestone**: clear rate ≥90% across the full domain-randomization range
   (proxy for "won't shatter on the real game").
3. **Real milestone**: the same policy, via protocol adapter + input injector, completes a
   real Snake Pit on a self-hosted **private server** (NR-CORE) first, before any official
   server.
4. **Visualization milestone**: a live wandb dashboard tracks every run, and rendered
   rollout videos of the current policy are logged periodically and on demand.

## Risks and open questions

- **Sim-to-real is a genuine bet.** If the sim's bullet patterns / movement feel differ
  enough from real ROTMG, the policy dodges "ghosts" and dies. Domain randomization
  mitigates but cannot guarantee transfer. Expected: one or two iterations of *measure the
  gap on real data → fix the sim → retrain*.
- **Gap-measurement harness**: before touching live input, replay real packet captures
  through the policy and check that its chosen actions look sane. This quantifies the gap
  cheaply and safely.
- **Tooling is hobby-grade**: nrelay / RealmShark / NR-CORE work but ship no RL-ready API;
  the observation/action adapter layer is custom integration work against the packet
  protocol.
- **Ban / ToS risk** on official servers is real but unquantified by any source. Mitigated
  by testing on a private server first.
- **Open**: exact Snake Pit bullet-pattern / enemy-AI fidelity needed for transfer is
  unknown until measured.

## Decisions taken

- Deploy target for first real test: **private server (NR-CORE)**, not official.
- Next step after this spec: **write a phased implementation plan** (sim first, then
  training loop, then deploy adapters). No code until the plan is reviewed.

## Non-goals

- Multiple dungeons, full autonomous play, leveling, or economy.
- Any supervised / behavioral-cloning / offline-RL component.
- Pixel-based observation.
- A general ROTMG bot framework.
