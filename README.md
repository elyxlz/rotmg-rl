# rotmg-rl

A reinforcement-learning agent that learns, from scratch, to clear a **real Snake Pit dungeon on a live RotMG server** — walk in through the portal, navigate the maze, dodge a swarm at zero defense, and kill the boss — and records the whole thing as one continuous, cheat-free, first-person video.

No scripted navigation, no aim-assist, no invincibility. The policy chooses every move, aim, shot, and ability itself, plays the real game client live against the real server, and survives only by playing well.

> **Result:** ~99% clear rate at the *real* fragile-Wizard difficulty (base HP, **0 defense**, real boss HP, spawn at the entrance), trained cold-start.

---

## The idea: the game *is* the simulator

The hard part of RL-on-a-real-game is the sim-to-real gap: you build a fast simulator, train in it, and the policy falls apart on the real game because your sim got the physics subtly wrong.

We delete the gap. Instead of reimplementing the dungeon, **we turn the real game server into the RL environment.** The actual betterSkillys C# `WorldServer` — the same code that runs the live game — is driven tick-by-tick by the trainer over shared memory. Real collision, real projectiles, real enemy AI, the real authored map. The policy trains against the exact dynamics it will face at deploy time.

The one invariant tying training and deployment together is the **observation vector** — a 9807-float view of the world (a `7×31×31` egocentric grid + a `3×32×32` fog-of-war minimap + 8 scalars). It is produced *bit-for-bit identically* by the C# engine during training and by a passive packet parser at deploy time, so a policy that clears in the "sim" is playing the real game.

---

## Architecture

```
                 ┌──────────────────────────── TRAINING ────────────────────────────┐
   PufferLib PPO  ◄──── /dev/shm ────►  betterSkillys C# WorldServer (real engine)
   (DungeonEncoder                      SimRlLoop: fixed-dt tick, real projectiles,
    CNN + recurrent,                    geodesic spawn, the 9807-float obs
    MultiDiscrete {move,aim,shoot,cast})
                 └───────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────── DEPLOY ──────────────────────────────┐
   trained .pt ──►  policy  ──►  live RotMG client  ──►  real :2050 server
                       ▲             │  (real input: WASD / mouse / space)
                       │             ▼
                 RealObsBuilder ◄── passive packet parse ◄── (same 9807-float obs)
                       │
                       └──►  ffmpeg POV recording ──►  clear.mp4
                 └───────────────────────────────────────────────────────────────────┘
```

- **`src/rotmg_rl/trainer/`** — the PPO core: the curriculum, the long run, eval ladders, the Protein sweep. One scalar `d ∈ [0,1]` ramps every difficulty knob jointly (HP, defense, boss HP, spawn distance) so the policy always faces slightly-harder-than-mastered.
- **`sim-server/`** — the C# engine as an RL env, shipped as a **patch/overlay** on a pinned upstream betterSkillys (only our ~3500 lines of `Sim*.cs`, applied to a fresh checkout — nothing forked-and-buried). With no `SIM_*` env set the server is byte-identical stock.
- **`src/rotmg_rl/obs.py`** — the obs vector, the single shared contract (its C# twin is `SimObsBuilder.cs`; `src/rotmg_rl/tools/verify_obs.py` proves they agree bit-for-bit).
- **`deploy/`** — two ways to drive the live game: a headless packet bridge (`deploy/nrelay/`), and a no-root proxy + input-injection path (`deploy/proxy/` + `deploy/flash/`) that plays the *real Flash client* for an authentic recorded POV.

See [`docs/design.md`](docs/design.md) for the shared-memory protocol, the curriculum, and the honest rough edges; [`docs/obs-schema.md`](docs/obs-schema.md) for the exact 9807-float layout.

---

## Reproduce

```bash
scripts/reproduce.sh   # the one-command path: setup → fetch+build the C# server → boot N worlds
                       #   → train (d-ramp curriculum) → eval ladder → deploy live → record the clear
```

Or run the stages individually:

```bash
scripts/setup.sh                       # python env (torch + PufferLib) + build the server_env C-shim into _C
sim-server/fetch.sh                    # fetch pinned betterSkillys @ pin, apply the Sim overlay, dotnet build
sim-server/run-server-sim.sh 32        # boot N=32 Snake Pit worlds (shm + futex barrier; isolated :2060)
python -m rotmg_rl.trainer.longrun --agents 32 --steps 200000 --server-log logs/server.log  # train
python -m rotmg_rl.trainer.eval --checkpoint checkpoints/longrun/best.pt --agents 32 --server-log logs/server.log  # ladder
```

Deploy + record (needs the live betterSkillys server + a built client; see [`deploy/README.md`](deploy/README.md)):

```bash
( cd deploy/nrelay && npm install && npx tsc -p . && node start-bot.js )   # headless live-play
deploy/record.sh                       # record the real-Flash-client POV of the clear
```

Training reaches a near-perfect clear at the real difficulty in a few hours on a single 3090.

---

## Repo layout

```
src/rotmg_rl/
  trainer/        # PPO core: train · curriculum · longrun · eval · trial · sweep · proof · difficulty · shm_config
  obs.py config.py schedule.py   # the shared obs spine + curriculum schedule
  deploy/         # policy runner + the live-play harness (server.py, policy.py, render.py)
  tools/          # verify_obs / verify_motion / verify_dflow / verify_obs_async (executable fidelity proofs)
sim-server/       # the C# engine-as-env: Sim*.cs overlay + overlay/seam.patch + pinned-upstream fetch.sh
                  #   run-server-sim.sh + wServer.sim.json (the isolated sim launch)
_pufferlib/       # vendored PufferLib 4.0 (the server_env Ocean env + DungeonEncoder), pinned
deploy/           # nrelay bridge · no-root proxy + input injection · flash client · ffmpeg recorder (record.sh)
docs/             # design.md · obs-schema.md · real-game-analysis.md · snakepit-spec.md
proof/            # the committed learnability curve (server_proof_curve.csv + summary)
```

The `[project.scripts]` entry points (also runnable as `python -m rotmg_rl.trainer.<name>`):

| command | module | what |
|---|---|---|
| `rotmg-train` | `rotmg_rl.trainer.train` | a single server-as-sim PPO run |
| `rotmg-curriculum` | `rotmg_rl.trainer.curriculum` | the d-ramp curriculum run |
| `rotmg-longrun` | `rotmg_rl.trainer.longrun` | the long curriculum run + periodic eval ladders + checkpoints |
| `rotmg-eval` | `rotmg_rl.trainer.eval` | the eval ladder → curriculum depth |
| `rotmg-trial` | `rotmg_rl.trainer.trial` | one sweep trial |
| `rotmg-sweep` | `rotmg_rl.trainer.sweep` | the Protein hyperparameter sweep |
| `rotmg-proof` | `rotmg_rl.trainer.proof` | the learnability proof harness (the curve in `proof/`) |

## Contributing

- `uv run pytest tests/` — fast, hermetic tests (no Docker, no GPU).
- The obs vector is the spine: any change to it must keep `src/rotmg_rl/obs.py`, `sim-server/Sim/WorldServer/core/worlds/SimObsBuilder.cs`, and `docs/obs-schema.md` in lockstep, with `src/rotmg_rl/tools/verify_obs.py` green.
- The shared-memory region layout is documented in exactly one place ([`docs/design.md`](docs/design.md)) and referenced by both the C-shim (`_pufferlib/ocean/server_env/server_env.h`) and `SimShmBridge.cs`.

## The deliverable

The north star is one continuous, unedited, first-person `.mp4` of the policy clearing a real Snake Pit on the live server with zero cheats — a disclosed AI demonstration in the AlphaStar / OpenAI Five tradition. The full integrity spec is in [`GOAL.md`](GOAL.md).
