# @realmlib/net wire-format patches

The headless nrelay client talks to a **betterSkillys** private server (vendored at
`vendor/betterSkillys/`), whose packet wire format diverges from the public RotMG format that
upstream `@realmlib/net` was written for. Without these patches the client desyncs immediately
(garbage object/tile ids, a rejected `Hello`, mis-parsed `MapInfo`, ignored hits).

These are the highest-value, hardest-won part of the deploy bridge: each field had to be matched
byte-for-byte against the betterSkillys C# handlers.

## Provenance

- **Upstream package:** `@realmlib/net@3.3.3` (npm), the version pinned in `deploy/nrelay/package.json`.
- **Patch target:** the compiled `lib/` tree inside the installed package
  (`node_modules/@realmlib/net/lib/`). The package ships compiled JS only; the patches edit the
  emitted `.js` (the `.d.ts` declarations were left untouched, so types still resolve).
- **Patch:** `realmlib-net-3.3.3.patch` — a unified diff of the 14 changed `.js` files against a
  clean `npm pack @realmlib/net@3.3.3`. Verified to apply cleanly with `git apply -p1` from the
  package root.

## How to apply (reproducible rebuild)

```bash
cd deploy/nrelay
npm install                                  # pulls @realmlib/net@3.3.3 into node_modules
cd node_modules/@realmlib/net
git apply -p1 < ../../../realmlib-patches/realmlib-net-3.3.3.patch
#   or, if not a git tree:
#   patch -p1 < ../../../realmlib-patches/realmlib-net-3.3.3.patch
```

(`postinstall` automation is intentionally left out — apply explicitly so the patch step is
auditable.)

## Field-by-field changelog

The unifying theme: **betterSkillys widened many `byte`/`short` fields to `int32`** (bullet ids,
object types, tile types) and **dropped the public-client-only Hello/MapInfo fields**.

### Core int32 widenings

- **`data/object-data.js`** — `objectType` read as `int32` instead of `unsignedShort`. betterSkillys
  object ids exceed 65535, so a short read corrupted every entity type.
- **`data/ground-tile-data.js`** — tile `type` read as `int32` instead of `unsignedShort` (after the
  unchanged `x`/`y` shorts). Tile ids likewise exceed 16 bits; this is what makes the no-walk lookup
  in `client.ts`/`policy-bridge.ts` correct.
- **`data/stat-data.js`** — `isStringStat()` reduced to `statType === 31 || statType === 62`
  (Name, GuildName). betterSkillys `ObjectStats.Write` only emits those two as strings; the upstream
  list (account-id / pet-name / owner-account-id) over-read and desynced the stat stream.

### Bullet id: byte -> int32 (hit + shoot packets)

Upstream encodes bullet ids as a single `unsignedByte` (0-127, hence the matching
`% 128` wraparound in `client.ts`, also patched to `% 2000000000`). betterSkillys uses `int32`. All
five hit/shoot packets were widened on both `read` and `write`:

- **`packets/outgoing/player-hit-packet.js`** — `bulletId` int32.
- **`packets/outgoing/enemy-hit-packet.js`** — `bulletId` int32 (then `targetId` int32, `kill` bool).
- **`packets/outgoing/other-hit-packet.js`** — `bulletId` int32 (then `objectId`, `targetId` int32).
- **`packets/outgoing/square-hit-packet.js`** — `bulletId` int32 (then `objectId` int32).
- **`packets/outgoing/player-shoot-packet.js`** — layout `time(int32), bulletId(int32),
  containerType(int32), startPos, angle(float)`. The upstream `speedMult`/`lifeMult` shorts are
  **dropped** (betterSkillys `PlayerShootHandler` does not read them).

### Hello (outgoing handshake)

- **`packets/outgoing/hello-packet.js`** — trimmed to the betterSkillys layout:
  `buildVersion, gameId(int32), guid, password, keyTime(int32), key(bytes), mapJSON(UTF32)`.
  All the public-client trailer fields are **removed**: `random1`, `random2`, `secret`, `entryTag`,
  `gameNet`, `gameNetUserId`, `playPlatform`, `platformToken`, `userToken`, `trailer`,
  `previousConnectionGuid`. Sending them made the server reject the handshake.

### MapInfo (incoming)

- **`packets/incoming/mapinfo-packet.js`** — betterSkillys layout:
  `width(short), height(short)` (was int32), `name`, `displayName`, `fp(uint32)`,
  `background(byte)`, `difficulty(byte)` (were int32), `allowPlayerTeleport(bool)`,
  `showDisplays(bool)`, then **new** `music(string)`, `disableShooting(bool)`,
  `disableAbilities(bool)`. The public-client `realmName`, `maxPlayers`, `connectionGuid`,
  `gameOpenedTime`, and the `clientXML`/`extraXML` UTF32 arrays are **not sent** by betterSkillys and
  are defaulted to empty.

### Reconnect (incoming)

- **`packets/incoming/reconnect-packet.js`** — `name, host, port(int32), gameId(int32),
  keyTime(int32), key(int16 len + bytes)`. The upstream `stats(string)` and `isFromArena(bool)`
  fields are **not present** in betterSkillys `Reconnect.cs` and are defaulted.

### EnemyShoot (incoming)

- **`packets/incoming/enemy-shoot-packet.js`** — betterSkillys `EnemyShootMessage`:
  `bulletId(int32)`, `ownerId(int32)`, `bulletType(byte)`, `startPos`, `angle(float)`,
  `damage(int32)` (was short), `numShots(byte)`, `angleInc(float)`. Unlike upstream, `numShots`/
  `angleInc` are **always** present (upstream made them conditional on `numShots !== 1`). Drives the
  enemy-bullet burst forward-simulation in the policy obs.

### Failure (incoming)

- **`packets/incoming/failure-packet.js`** — the trailing `errorPlace` / `errorConnectionId`
  strings are read **only if bytes remain** (`reader.remaining > 0`). betterSkillys often sends a
  short failure packet; the unconditional reads threw on the buffer end.

### RC4 disabled

- **`crypto/rc4.js`** — `cipher()` early-returns (`/*RC4-DISABLED*/ return;`), leaving the byte
  stream in plaintext. This betterSkillys build runs with RC4 packet encryption turned off; running
  the cipher would scramble every packet.

## Re-deriving these patches

If the betterSkillys wire format changes, regenerate by diffing the patched install against a clean
package:

```bash
npm pack @realmlib/net@3.3.3 && tar xzf realmlib-net-3.3.3.tgz   # -> package/lib (upstream)
diff -ruN package/lib node_modules/@realmlib/net/lib > realmlib-net-3.3.3.patch
```

Cross-check every field against the betterSkillys C# handlers under
`vendor/betterSkillys/` (e.g. `Hello.cs`, `Reconnect.cs`, `EnemyShootMessage`, the `*Handler`
classes, and `ObjectStats.Write`).
