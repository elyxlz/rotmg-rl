# sim-server

A C# game engine repurposed as a deterministic, uncapped RL environment, shipped
as a **thin pinned-upstream overlay**. This package is only our ~3500 LOC; the
~1M-LOC game server itself is fetched verbatim from a pinned upstream commit and
the overlay is applied on top. Nothing upstream is vendored here.

## Pin & provenance

```
UPSTREAM_REPO   = https://github.com/Honeybee5151/betterSkillys_SERVER_BANNERSYSTEM
UPSTREAM_BRANCH = master
UPSTREAM_COMMIT = cf125dd8cfc39667e4f3bb4f8baab85ef345ca9d
```

Lineage: the server is a **betterSkillys**-network build, itself a fork of
`Slendergo/TKR-Source` (TK-Reborn), targeting **.NET 8 (`net8.0`)**. The exact
variant we pinned was originally published under a now-**deleted** repo (reported
`shilohdynasty7142/runes02`). **`Honeybee5151/betterSkillys_SERVER_BANNERSYSTEM`
is the closest surviving public mirror**, confirmed by blob-hash match against the
deleted original (962/984 files exact path+content) plus a clean `dotnet build`
on .NET 8.0.422.

The only structural difference between the original fork and this mirror is a
`source/` path prefix (fork: `source/WorldServer/...`, mirror/upstream:
`WorldServer/...`). The overlay is authored at the **upstream-relative** layout
(no `source/` prefix), so it applies to this mirror with `git apply -p1`.

## What the overlay adds (~3500 LOC)

- **15 new `Sim*.cs` files** (`Sim/WorldServer/core/{objects,worlds}/`) — the
  self-contained sim/RL harness: the fixed-dt uncapped loop, the in-process RL
  loop, the shared-memory + redis lockstep bridge, the obs/action/reward
  builders, the measurement probe, and the deterministic-RNG plumbing.
- **11 gated engine seams** (`overlay/seam.patch`) — minimal hooks into the
  stock engine (`GameServer.cs`, `Enemy.cs`, `Entity.cs`, `World.cs`,
  `RootWorldThread.cs`, `WorldBranch.cs`, `CollisionMap.cs`, `Behavior.cs`,
  `Grenade.cs`, `Transition.cs`, `EntityUtils.cs`) that call into the Sim code.
  **Every seam is gated behind a `SIM_*` env check** (see `SimMode.cs`).

> **With no `SIM_*` env set, the built server is byte-identical stock.** Each seam
> short-circuits to the original code path unless its flag is present, so this same
> binary serves both stock play and the RL sim.

## Package layout

```
sim-server/
├── pin.txt              # the upstream pin sourced by fetch.sh
├── fetch.sh             # fetch upstream @ pin, apply overlay, build (idempotent)
├── overlay/
│   └── seam.patch       # 11-file engine-seam patch (a/WorldServer/..., -p1)
├── Sim/                 # the 15 Sim*.cs at UPSTREAM-relative paths
│   └── WorldServer/core/
│       ├── objects/     # SimAgent.cs, SimProbe.cs
│       └── worlds/      # 13 Sim*.cs (Mode, Runner, Harness, RlLoop, ObsBuilder,
│                        #   ActionApply, Projectiles, EnemyShoots, Geodesic,
│                        #   StepGate, ShmBridge, ShmBarrier, ShmAsync)
└── upstream/            # (created by fetch.sh) the pinned checkout + overlay applied
```

## Fetch & build

```bash
bash fetch.sh
```

`fetch.sh` (idempotent):

1. sources `pin.txt`;
2. clones `$UPSTREAM_REPO` into `./upstream` if absent;
3. `git fetch` + `git checkout $UPSTREAM_COMMIT`;
4. resets the tree pristine (`git clean -fdq` + `git checkout -- .`) so the
   overlay always applies onto stock;
5. `git apply -p1 overlay/seam.patch`;
6. `cp -r Sim/. upstream/`;
7. `dotnet build -c Release WorldServer/WorldServer.csproj`.

It prepends `~/.dotnet` to `PATH` (the SDK location on this box) and prints a
success banner on green. Re-running re-pristines and re-applies, so it is safe to
run repeatedly.

The build artifact is the WorldServer under
`upstream/WorldServer/bin/Release/net8.0/`.

## Run (isolated sim mode)

The sim is enabled purely by environment — the same binary, gated by `SIM_*`:

- `wServer.sim.json` — a sim-only server config (its own ids/db indices) so the
  sim never collides with the live server's config.
- `run-server-sim.sh` — the isolation wrapper: launches the WorldServer with the
  sim config and the `SIM_*` env, pinned to its own resources:
  - **redis `127.0.0.1:6390`, db `5`** (the dedicated sim lockstep channel; see
    `SimStepGate.cs` defaults `SIM_STEP_REDIS_PORT=6390`, db 5),
  - **server port `2060`** (the live server stays on `:2050`).

Key `SIM_*` switches (full list read in `SimMode.cs` / `SimStepGate.cs` /
`SimShmBridge.cs`):

| env | effect |
|-----|--------|
| `SIM_HARNESS=1`   | spawn the Snake-Pit measurement harness (probe + CSV) |
| `SIM_UNCAPPED=1`  | uncapped fixed-dt loop instead of stock 10 TPS (implies harness) |
| `SIM_INPROC=1`    | in-process RL loop (obs/action/reward on the world thread, no nrelay) |
| `SIM_SHM=1`       | PufferLib C-shim drives ticks over shared memory + redis lockstep |
| `SIM_ASYNC=1`     | async-overlap free-run (GPU/worlds overlap) instead of strict barrier |
| `SIM_STEP_DRIVEN=1` | redis step-on-command lockstep gate |
| `SIM_RNG_SEED` / `SIM_RL_SEED` | seed the deterministic fight RNG for reproducible runs |
| `SIM_WORLDS=N`    | number of parallel Snake-Pit worlds |
| `SIM_FIXED_DT_MS` | synthetic logical delta per tick (default 100ms == 10 TPS) |

With **none** of these set, `SimMode` reports every switch off and the server runs
the stock real-time path unchanged.
