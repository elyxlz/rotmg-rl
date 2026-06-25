# Progress

Autonomous build log. Newest entry on top. See `GOAL.md` for the loop and
`docs/specs/2026-06-25-rotmg-rl-design.md` for the design.

## Milestones

| ID | Milestone | Status |
|----|-----------|--------|
| M0 | Repo scaffold + uv env on the GPU box + wandb smoke run | in progress |
| M1 | PufferLib C sim of Snake Pit (>=1M steps/s/core) + renderer + play.py | not started |
| M2 | Cold-start training stack (recurrent PPO + shaping + curriculum + RND + DR) | not started |
| M3 | Sim milestone: clear simulated Snake Pit >=90% (eval >=200 eps) | not started |
| M4 | Robustness milestone: >=90% across full domain-randomization range | not started |
| M5 | Deploy adapters (NR-CORE server + protocol reader + input injector + gap harness) | not started |
| M6 | Real milestone: clear a real Snake Pit on the private server (DONE) | not started |

## Log

### 2026-06-25 — M0 kickoff
- Scaffolded repo: `pyproject.toml` (uv, base deps + optional `train` group for torch/pufferlib),
  `src/rotmg_rl/`, `scripts/smoke_wandb.py`, `.gitignore`.
- Next: provision the GPU box (`ripperred`) — install `uv`, clone repo, sync env, run the
  wandb smoke (offline first to prove the pipeline without blocking on an API key).
