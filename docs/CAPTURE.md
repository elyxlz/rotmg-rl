# Capturing a real Snake Pit run (for the gap measurement)

To measure the sim-to-real gap and re-fit the sim, I need a recording of one real Snake Pit
fight. The cleanest source is **RealmShark** (passive packet sniffer, read-only, no injection,
no ban risk).

## What I need

Any dump that lets me reconstruct, per tick:
- the boss's `EnemyShoot` bursts,
- player and boss positions + HP.

**Don't worry about exact formatting** — send me whatever RealmShark produces and I write the
adapter. The richer the better, but the minimum useful signal is the `EnemyShoot` packets.

## The fields that matter (RealmShark `EnemyShootPacket`)

| field | meaning |
|-------|---------|
| `startingPos` (x, y) | burst origin |
| `angle` | direction of the first bullet |
| `numShots` | bullets in the burst |
| `angleInc` | angle step between bullets (fan = `angle + i*angleInc`) |
| `ownerId` | which enemy fired (to isolate the boss) |
| `bulletType` | projectile id -> speed/lifetime via `Objects.xml` |
| `time` | packet timestamp |

Bullet **speed/lifetime are not in the packet** — they come from the projectile asset. If you
can also grab the `Objects.xml` projectile entries for Stheno's bullets (or the resource pack),
I can reconstruct exact bullet positions; otherwise I infer speed from timing.

## The schema I ingest (if you can produce it directly)

JSONL, one object per `EnemyShoot` packet, plus occasional state lines:

```json
{"type":"shoot","x":20.1,"y":8.3,"angle":1.05,"numShots":8,"angleInc":0.39,"bulletType":12,"ownerId":345,"time_ms":1733251.0}
{"type":"state","player":[19.0,33.2],"playerHp":78,"boss":[20.1,8.3],"bossHp":2300,"time_ms":1733260.0}
```

`scripts`/`rotmg_rl.deploy.realmshark` already maps these fields into the sim's event model
(`to_event`), and `rotmg_rl.deploy.gap.measure_gap` turns a capture into a refit `SnakePitConfig`.

## Capture steps (Windows)

1. Install Java + Npcap, download the latest `Tomato-v*.jar` from RealmShark releases.
2. Start the sniffer, then play through one Snake Pit to the boss and (ideally) a kill.
3. Send me the resulting packet log. I handle the rest.
