# Progress

Autonomous build log. Newest entry on top. See `GOAL.md` for the loop and
`docs/specs/2026-06-25-rotmg-rl-design.md` for the design.

## Milestones

| ID | Milestone | Status |
|----|-----------|--------|
| M0 | Repo scaffold + uv env on the GPU box + wandb smoke run | DONE |
| M1 | PufferLib C sim of Snake Pit (>=1M steps/s/core) + renderer + play.py | in progress |
| M2 | Cold-start training stack (recurrent PPO + shaping + curriculum + RND + DR) | deps ready |
| M3 | Sim milestone: clear simulated Snake Pit >=90% (eval >=200 eps) | not started |
| M4 | Robustness milestone: >=90% across full domain-randomization range | not started |
| M5 | Deploy adapters (NR-CORE server + protocol reader + input injector + gap harness) | not started |
| M6 | Real milestone: clear a real Snake Pit on the private server (DONE) | not started |

## Log

### 2026-06-25 — M1 foundation + M2 deps verified on box
- **M0 DONE**: box provisioned (uv 0.11.6, repo cloned, env synced); wandb smoke logged a
  200-step curve offline (`smoke/reward=0.928`). Online following needs `WANDB_API_KEY` on box.
- **Observation schema** (`observation.py`): world-agnostic `GameState` -> egocentric 6ch
  15x15 grid + 6 scalars. The sim-to-real contract; both adapters build it identically.
- **Snake Pit numpy env** (`sim/snakepit.py`): player 8-dir move/shoot, boss drift + radial
  bursts, bullet collisions, dense shaped reward, clear/death/timeout. 4 tests pass on box.
- **Renderer + scripts**: headless `render()` rgb frames; `record_episode.py` wrote a valid
  mp4; `bench_env.py` -> **6,583 steps/s/core** (numpy baseline; C port needed for 1M target).
- **M2 deps verified on box**: torch 2.12.1+cu130, CUDA on 2x RTX 3090, pufferlib 2.0.6.
- Next: wire PufferLib recurrent-PPO training on the numpy env to produce the FIRST real
  learning curve + policy rollout video (de-risks M2 cheaply), then C-port the env for speed.

### Blockers needing the user (non-stopping)
- `WANDB_API_KEY` on the box for ONLINE progress following (offline works meanwhile).
- M5/M6 (real game) need infra the headless Linux box can't provide: a running ROTMG client
  + NR-CORE private server + a host that can inject input. Sim milestones (M0-M4) run fully
  on the GPU box; real-game deploy will need the user's environment.
