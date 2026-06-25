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

#### M2 implementation plan (decided after introspecting pufferlib 2.0.6)
- pufferlib 2.0.6 ships NO packaged trainer (`pufferlib.pufferl` is 3.x only). So M2 = a
  compact, self-contained recurrent PPO loop we own, built on:
  - `pufferlib.emulation.GymnasiumPufferEnv` to wrap `SnakePitEnv` (flattens the Dict obs).
  - `pufferlib.vector.make(..., backend=Multiprocessing)` for parallel envs across 32 cores.
  - a torch policy: CNN over the 6x15x15 grid + MLP over scalars -> fuse -> `pufferlib.pytorch.LSTM`
    -> actor (MultiDiscrete 9x9) + critic heads.
- PPO with GAE, entropy bonus, minibatched epochs; wandb logging of reward/clear-rate/ep-len/
  entropy/KL; periodic greedy rollout video to wandb. Reward shaping + RND + curriculum + DR
  layer on after the bare loop is verified to learn the shrunk-boss case.
- OBS LAYOUT (verified on box): `GymnasiumPufferEnv` flattens the Dict to `Box((1356,) f4)`;
  first 1350 = grid `(6,15,15)`, last 6 = scalars. Policy reconstructs by slice+reshape
  (`x[:, :1350].view(B,6,15,15)`, `x[:, 1350:]`) — no `nativize` needed. Action space stays
  `MultiDiscrete([9,9])` (two categorical heads, summed logprob/entropy).

### Next iteration starts here
- Build `src/rotmg_rl/policy.py` (CNN over grid + MLP over scalars -> fuse -> LSTM -> 2 actor
  heads + critic) and a CleanRL-style recurrent-PPO `scripts/train.py` over `pufferlib.vector`
  envs. Verify a short run learns the shrunk-boss case and logs a curve + video to wandb.

### Blockers needing the user (non-stopping)
- `WANDB_API_KEY` on the box for ONLINE progress following (offline works meanwhile).

### Correction (2026-06-25): M5/M6 ARE fully doable headless on the Linux box
- Earlier note claimed real-game deploy needs the user's environment. Retracted. The real
  interface is the network protocol, not a GUI: a headless `nrelay` fork both reads state
  (incl. `EnemyShoot` packets) and sends actions (`Move`/`PlayerShoot`); the bullet field is
  reconstructed by locally simulating projectiles. NR-CORE runs headless on Linux. So the
  whole pipeline (M0-M6) runs on `baby-ai-ripper`. No GUI client, no Wine/Xvfb, no input
  injection into a window. Spec component 4 updated accordingly.
