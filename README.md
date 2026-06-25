# rotmg-rl

Cold-start reinforcement learning agent that clears one ROTMG (Realm of the Mad God)
dungeon — **Snake Pit** — trained entirely in a fast custom simulator, then deployed on the
real game.

- **No supervised data / no behavioral cloning.** Pure cold-start RL.
- **PufferLib** (recurrent CNN-LSTM PPO) over a custom C simulator of the dungeon.
- **Sim-to-real via a shared observation schema**: the policy only ever sees a world-agnostic
  egocentric tensor, produced identically from the sim's ground truth and from the real
  game's network packets, so a sim-trained policy can drive the real client.

See [`docs/specs/2026-06-25-rotmg-rl-design.md`](docs/specs/2026-06-25-rotmg-rl-design.md)
for the full design and [`GOAL.md`](GOAL.md) for the autonomous build loop.
