# deploy/ — real-server deploy bridge

Everything needed to run a trained `rotmg-rl` policy against a **live betterSkillys RotMG server**,
plus the Flash-client tooling used to watch it play. This is the hand-written, non-binary half of
what previously lived only on the box at `~/rotmg-realgame/`; it is vendored here so the deploy path
is tracked, auditable, and reproducible.

```
deploy/
├── nrelay/          # headless TS client + policy bridge (see nrelay/README.md)
│   └── realmlib-patches/   # @realmlib/net wire-format patches (see its README)
└── flash/           # Flash client build + capture scripts
```

The Python policy server the bridge spawns lives in the main tree at
`src/rotmg_rl/deploy/v3/` (`server.py`, `obs.py`, `policy.py`).

## Components on the box (for orientation)

`~/rotmg-realgame/` holds the full deploy stack. Most of it is upstream source already tracked
elsewhere or rebuildable, and is **not** vendored:

| Path | What it is | Vendored? |
|---|---|---|
| `nrelay/` | headless client fork | custom pieces only -> `deploy/nrelay/` |
| `betterSkillys/` | private server (mono/dotnet `WorldServer`) | source already at `vendor/betterSkillys/` |
| `NR-CORE/` | alternate server build harness | no (rebuildable) |
| `RealmShark/` | packet-inspection tool (Java/Gradle) | no (external upstream) |
| `*.swf`, `flash_client_build/` | compiled Flash client + SDK | no (binary; rebuilt by `flash/flash_build.sh`) |

## Checkpoints (NOT in git)

Trained policy checkpoints are large binaries and stay out of the repo. On the box:

- `~/rotmg-rl/checkpoints/` (~4.4 GB total) — e.g. `full_dungeon_95.pt` (the checkpoint the bridge
  loads by default), `confirm-gamma095.pt`, the `curr-c0*.pt` curriculum series, etc.

Back these up separately (e.g. `~/transfer_checkpoints.sh`). `git add` nothing under `checkpoints/`.

---

# Runbook: stand up a live policy run

Reproduces what was set up by hand on the box. Assumes the betterSkillys server source
(`vendor/betterSkillys/`) and this repo are present.

## 1. Run the betterSkillys server + redis

Redis must be up on `127.0.0.1:6379` (the WorldServer config `wServer.json` points there). Build and
launch the world server:

```bash
cd vendor/betterSkillys/source/WorldServer
dotnet build -c Release
( cd bin/Release/net8.0 && dotnet WorldServer.dll wServer.json )
```

The App/account server (`vendor/betterSkillys/source/App`) serves the account API on its configured
port (registration + login).

## 2. Create the accounts

Register via the account server's `POST /account/register` endpoint (form fields
`guid`/`newGuid`/`newPassword`/`name` — see `vendor/betterSkillys/source/App/Controllers/AccountController.cs`).
Two accounts are used:

- **Bot account** — drives the policy. Matches `deploy/nrelay/accounts.json`
  (template: `accounts.example.json`). Used by `nrelay`.
- **Spectator account** — `spectator@spec.com` / `specpass123`. Auto-logged-in by the Flash client
  (the autologin patch in `flash/flash_build.sh`) so you can watch the bot from a real client.

## 3. Make the bot account admin

The bridge issues admin-only commands (`/max`, `/spawn Snake Pit Portal`, `/tppos`). Admin is a
boolean field on the account's redis hash (`account.{id}`, field `admin` — see
`vendor/betterSkillys/source/Shared/database/account/DbAccount.cs`). For the first account:

```bash
redis-cli hset account.1 admin 1
```

(Substitute the actual account id. `rank` can likewise be raised for elevated command tiers.)

## 4. Build & run the nrelay bridge

See `deploy/nrelay/README.md` for the full sequence (npm install, apply the `@realmlib/net` patches,
`tsc`, `node start-bot.js`). On connect the bridge:

1. spawns the Python policy server (`rotmg_rl.deploy.v3.server`, checkpoint `full_dungeon_95.pt`);
2. in the Nexus, sends `/max` then `/spawn Snake Pit Portal` and uses the spawned portal;
3. in the Snake Pit, `/tppos`-es to the boss and lets the policy fight.

## 5. (Optional) Watch via the Flash client

`flash/flash_build.sh` compiles the betterSkillys Flash client headlessly and captures screenshots /
video; `flash/flash_play.sh` replays a built `client.swf` and drives the UI via `flash/flash_click.py`
(synthetic X events under Xvfb). The build script is self-contained and idempotent — it downloads the
Flex 4.16.1 SDK + AIR32 + the standalone projector, then applies the client source fixes inline:

- **Empty MiniDungeonHub** — overwrites `MiniDungeonHub.xml` with `<Objects></Objects>`.
- **Font substitution** — swaps the proprietary MyriadPro `.otf` fonts for Liberation Sans `.ttf`
  and flips `embedAsCFF` to false (the `.otf`s aren't shippable).
- **Dead-import removal** — an iterative compile loop strips `import`s for definitions the build
  reports as missing, up to 30 passes, until `WebMain.as` links.
- **Autologin** — injects the spectator creds into `WebLoadAccountTask.as` so the client logs in
  unattended.

These need a Linux box with `Xvfb`, `ffmpeg`, `unzip`, and the Liberation fonts installed.
