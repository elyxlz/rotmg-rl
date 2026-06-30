# Snake Pit live-deploy harness — handoff / state

Checkpoint of the no-root live-deploy effort: drive the trained policy through a **real** Snake Pit
on the live betterSkillys server and record a continuous, cheat-free POV clear (see `GOAL.md`).
This dir is the working harness, snapshotted from the GPU box (`ripbox`, user `audiogen`) where it
runs. Code here is the source of truth; the box is the runtime.

## Status (2026-06-30)

The full pipeline works end to end and is reliable/idempotent. The big deploy-time bug is **found
and fixed**. The policy now traverses the maze with proper T7 gear, but does **not** yet clear — it
dies ~20-25 s in to deeper enemies (Greater Pit Snake/Viper). Combat transfers well; survivability
in the deeper rooms is the open problem.

### What works
- **Server-side provisioning** (`provision_wizard.py`): re-arms the deliverable char `char.1.2`
  directly in the live server's redis (6379) — maxed 8/8 Wizard, T7 Staff of Destruction (`0xa9e`)
  + Burning Retribution Spell (`0x2055`) **equipped**, Snake Pit Key (`0x70b`) in the first
  inventory slot, alive, full HP. Idempotent, re-runnable. Removes the flaky UI setup entirely.
- **One-command attempts** (`attempt.sh`): release session -> re-arm char -> run harness
  (fresh obs pipeline -> client -> select char 2 -> open portal with the held key -> enter ->
  record + drive policy -> monitor to death/clear). Video at `~/rotmg-rl/videos/snakepit_run.mp4`.
- **Cheat-free entry**: the key is pre-provisioned (allowed setup); the run uses it as normal
  gameplay (shift-click). No `/give`/admin during the run.
- **Obs fidelity**: the live packet parser latches the right player and produces the 9807-float obs
  that matches training (verified the player tracks HP/gear/position correctly).

### THE FIX — 45° movement-frame rotation (root cause of "walks into a wall / dead-end")
The policy trained in the C# sim, where a move action is applied in **world space**. At deploy the
move is realized via **WASD**, but the real client's WASD axes are rotated **45°** from world (the
isometric screen frame). So every commanded direction came out 45° off and the policy drifted
diagonally into walls/dead-ends. Aim was unaffected (mouse is absolute) — which is why combat
transferred but navigation didn't.

Proof: `policy.py` traces commanded move-vector vs actual world displacement per tick to
`/dev/shm/rotmg_trace.txt`; `tools/trace_analysis.py` measures the rotation. It was a clean
**-45°**. Fix = rotate the intended move direction by +45° before mapping to WASD
(`config.MOVE_FRAME_ROT_DEG = 45.0`, applied in `input.py:apply_action`). After the fix the
rotation collapsed to **~0°** and the policy traverses the maze.

### Open problem (next session)
It survives longer and explores deeper but still dies (~25 s) to tougher snakes. Hypotheses, rough
priority:
1. **Survivability in deeper rooms** — may just need to dodge the harder snake patterns; check
   whether the obs enemy-bullet channel is faithful at deploy (timing/positions of `ENEMYSHOOT`).
2. **Residual aim offset** — combat works (earned the Sniper >50%-accuracy bonus), but aim *might*
   still carry a frame offset worth measuring the same way movement was (trace aim vs hit angle).
3. **Other sim-real deltas** — player speed (`0.55` t/tick assumed), tick rate / 10 Hz drive,
   projectile speeds. Worth a `verify_*` pass against the sim.
4. **Residual move spread** — post-fix rotation buckets still show some ±45° (knockback/collision
   sliding + discrete 8-way WASD). Probably fine; revisit only if pathing still looks off.

## How to run (on ripbox)

```bash
ssh ripbox
# one full attempt (re-arms char + runs + records):
bash ~/snakepit-harness/attempt.sh            # video -> ~/rotmg-rl/videos/snakepit_run.mp4
# re-measure the move-frame rotation after a run:
~/rotmg-rl/.venv/bin/python ~/snakepit-harness/tools/trace_analysis.py
# just re-arm the char (no run):
cd ~/rotmg-rl && .venv/bin/python ~/provision_wizard.py
```

Constants worth knowing live in `config.py`: `CHAR_SELECT=(450,314)` (char 2 is the sole alive
slot — `alive.1=[2]`; char 7 was SREM'd to make selection deterministic), `PORTAL_KEY_SLOT=(826,575)`
(first inventory slot where the provisioned key sits), `MOVE_FRAME_ROT_DEG=45.0` (the fix).

## Box layout
- `~/snakepit-harness/` — this harness (run from here).
- `~/rotmg-rl/` — the trainer + deploy package (`src/rotmg_rl/...`), `obs_reader.py`, checkpoint at
  `checkpoints/longrun/final.pt` (hidden=2048, layers=2), `.venv`, videos.
- Live game server: loopback `:2050`, redis `6379`. Training sim is separate (`:2060`, redis `6390`).
  GPU 0 = training (don't disturb); GPU 1 = inference.
- Client path: real Flash client under Xvfb `:99`, `LD_PRELOAD=~/redirect.so` reroutes `:2050`->`:2052`
  (tee proxy) -> obs consumer -> `/dev/shm/rotmg_obs.f32`; `miniwm` provides EWMH focus.

## Local instrumentation added this session (keep or revert intentionally)
- `policy.py`: writes the per-tick move/pos trace to `/dev/shm/rotmg_trace.txt` (cheap; keep for now).
- `obs.py`: publishes live player world-pos to `/dev/shm/rotmg_pos.txt` each tick (used by probes).
- `tools/probe.py`, `tools/enter_probe.py`, `tools/clk.py`: ad-hoc input/movement probes.

## Rebuilding the native bits (sources committed alongside the binaries)
```bash
cc -shared -fPIC -o redirect.so redirect.c -ldl      # :2050 -> :2052 connect() shim (LD_PRELOAD)
cc -o miniwm miniwm.c -lX11                            # minimal EWMH WM so the Flash client takes focus
```
