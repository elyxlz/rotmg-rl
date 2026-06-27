# nrelay deploy bridge

The headless RotMG client that drives a trained policy against the live **betterSkillys** private
server (vendored at `vendor/betterSkillys/`). It is a fork of
[`thomas-crane/nrelay`](https://github.com/thomas-crane/nrelay) with a small set of hand-written
additions and wire-format patches; this directory vendors **only** those custom pieces, not the
whole upstream framework.

## Upstream baseline

- Repo: `https://github.com/thomas-crane/nrelay.git`
- Commit: `4517f57a8350ad2bb3b51167c36b1c74cb2d0547` (`Merge branch 'Urantij-conerror'`)

To reconstruct the full working tree, clone that commit, drop in the `src/` files here, apply the
patches under `patches/`, then `npm install` and apply the `@realmlib/net` patches (see
`realmlib-patches/README.md`).

## What's vendored here

```
nrelay/
├── src/
│   ├── policy-bridge.ts      # NEW. The core bridge plugin (see below).
│   └── connect-logger.ts     # NEW. Decode/debug logger for MapInfo/Update/NewTick/Failure.
├── patches/
│   ├── core-client.ts.patch  # 2 surgical fixes to upstream src/core/client.ts
│   └── crypto-rsa.ts.patch    # swap RotMG prod RSA pubkey -> betterSkillys server pubkey
├── realmlib-patches/         # @realmlib/net@3.3.3 wire-format patches (see its README)
├── start-bot.js              # entry point: Runtime.run(), bounded 120s run window
├── package.json              # upstream nrelay manifest (pins @realmlib/net@3.3.3)
├── tsconfig.json
├── accounts.example.json     # bot login template (copy -> accounts.json, fill creds)
├── versions.json             # client/build version + clientToken for the betterSkillys build
├── servers.cache.json        # server list: Netherlands -> 127.0.0.1 (local betterSkillys)
└── packets.json              # packet-id <-> name map for this server build
```

Not vendored (regenerate on the box): `node_modules/` (run `npm install`), and
`resources/GroundTypes.json` + `resources/Objects.json` (~9 MB of game data exported from the
betterSkillys XML resources; the no-walk tile lookup and projectile speeds read these). Export them
from `vendor/betterSkillys/` resources, or copy from the server install.

## The custom code

### `patches/core-client.ts.patch` — two hard-won fixes

1. **int32 bullet ids.** `getBulletId()` wraps at `% 2000000000` instead of `% 128`. The
   public-client bullet id was a single byte; betterSkillys uses int32 (matching the
   `@realmlib/net` hit/shoot-packet patches). Without this, bullet ids collide and hits are dropped.
2. **no-walk tile collision.** `moveTo()` consults `runtime.resources.tiles[type].noWalk` so the
   bot can't walk through walls/obstacles the policy doesn't model.

### `patches/crypto-rsa.ts.patch` — server RSA key

Swaps the hard-coded RotMG production RSA public key for the betterSkillys server's public key, so
login credentials are encrypted with the key the local server can decrypt.

### `src/policy-bridge.ts` — the bridge plugin

An nrelay `@Library` that wires the live game to the trained policy:

- **Spawns the policy server** as a subprocess:
  `.venv/bin/python -m rotmg_rl.deploy.server --checkpoint checkpoints/full_dungeon_95.pt`
  (cwd `~/rotmg-rl`, under the `scripts/setup.sh`-provisioned venv that has torch + the vendored
  pufferlib), and talks to it over line-delimited JSON on stdin/stdout. The Python side lives in this
  repo at `src/rotmg_rl/deploy/`.
- **Per-tick loop** (`onNewTick`): reconstructs the policy's view from decoded packets — player
  state, enemies (with boss detection for Stheno/Snake Queen), player bullets, walkable tiles, and
  enemy-shot bursts forward-simulated from `EnemyShoot` (`onEnemyShoot`) — ships it to the policy,
  and applies the returned intent (move/aim/shoot/cast).
- **Autonomous nav into content**: in the Nexus it sends `/max` then `/spawn Snake Pit Portal`,
  finds the spawned portal object (type `0x0718`) and uses it; in the Snake Pit it `/tppos`-es to the
  boss to bridge the sim-to-real navigation gap (the real maze is ~80 tiles; the sim doesn't model
  it). These commands require the bot account to be **admin** (see the runbook).
- **Intent application** (`applyIntent`): greedy passability-checked stepping, `client.shoot(angle)`
  for the weapon, and a `UseItem` on inventory slot 1 (the Wizard ability, gated on MP/cooldown) for
  casts.

### `src/connect-logger.ts`

A passive `@Library` that logs decoded `MapInfo` / `CreateSuccess` / `Failure` / `Update` /
`NewTick` packets — the instrument used to verify the `@realmlib/net` patches decode the betterSkillys
stream correctly. Safe to leave enabled; remove from the plugin set for quiet runs.

## Build & run (on the box)

```bash
cd ~/rotmg-realgame/nrelay
npm install
( cd node_modules/@realmlib/net && git apply -p1 < <repo>/deploy/nrelay/realmlib-patches/realmlib-net-3.3.3.patch )
npx tsc -p .                       # compile src -> lib
cp accounts.example.json accounts.json   # then edit in the bot creds
node start-bot.js
```
