# Snake Pit — faithful spec (extracted from betterSkillys source)

Ground truth for the M1 faithful sim. Source: `betterSkillys/source/WorldServer/logic/db/
BehaviorDb.SnakePit.cs` + `logic/behaviors/Shoot.cs`. This replaces the v1 radial-burst guess.

## Shoot semantics
`Shoot(radius, count=1, shootAngle=null, projectileIndex=0, fixedAngle=null, rotateAngle=null,
angleOffset=0, defaultAngle=null, predictive=0, coolDownOffset=0, coolDown, ...)`
- Fires `count` bullets aimed at the player, fanned `shootAngle` degrees apart.
- `rotateAngle`: the aim rotates by this each shot (a rotating spread).
- `projectileIndex`: which of the enemy's projectiles (speed/lifetime/damage from its XML).
- `coolDown`: ms between shots. `predictive`: lead the player's velocity.

## Stheno the Snake Queen (boss)
- Activate: a player within 20 tiles -> "Start" (Invulnerable + taunt, 1s) -> First Phase.
- **First Phase** (until HP < 66%):
  - `Wander(0.3)` slow drift.
  - `Reproduce("Stheno Swarm", 15, 5, 1500)`: spawn up to 5 swarm minions, every 1.5s.
  - `Grenade(3.5, 150, 11, null, 1500, Confused, 1000)`: AoE grenade r=3.5, dmg 150, range 11,
    every 1.5s, inflicts Confused.
  - `Shoot(2, 3, shootAngle:15, coolDown:1500)`: 3 aimed bullets, 15 deg apart, every 1.5s.
- **Second Phase Start**: Invulnerable, ReturnToSpawn, green flash, 1.5s.
- **Second Phase** (until HP < 33%):
  - `Grenade(3.5, 150, 11, null, 1000, Confused, 1000)`: grenade every 1s.
  - `Shoot(25, 4, projectileIndex:2, rotateAngle:15, coolDown:250)`: 4 bullets, ROTATING 15 deg
    per shot, every 0.25s (dense).
- **Third Phase Start**: Invulnerable, red flash, 1.5s.
- **Third Phase**:
  - `Shoot(30, 3, shootAngle:15, coolDown:1500)` + `Shoot(25, 4, rotateAngle:15, coolDown:500)`.
  - 8x `Grenade(1.5, 75, 6, <fixed angle>, Petrify)` at 0/90/180/270 (every 1.5s) and the 45s
    (every 3s): directional petrify grenades.

## Minions
- **Stheno Swarm**: Wander(0.3) + `Shoot(10, coolDown: 750-250)`; despawns over time.
- **Stheno Pet**: `Shoot(25, coolDown:1000)`; wanders when Stheno absent.
- **Pit Snake / Pit Viper / Yellow Python / Brown Python**: Wander(0.3) + `Shoot(20, coolDown:1000)`.
- **Fire Python / Greater Pit Snake**: `Shoot(15, count:3, shootAngle:5, coolDown:1000)`.

## Extracted hard numbers (betterSkillys EmbeddedData_SnakePitCXML.xml + Snake Pit.jm)
- Stheno base **MaxHitPoints 7500** (ScaleHP2(20) scales with player count; sim uses a fixed
  single-player value, tune for a ~feasible fight as in v1).
- Projectiles: **id0 Speed 70, LifetimeMS 1500** (~7 tiles/s, ~10.5 tile range);
  **id1 Speed 62, LifetimeMS 2000**. (projectileIndex 2 in phases 2/3 -> check for an id2 entry.)
  ROTMG: tiles/sec ~= Speed/10; match the server's projectile motion when calibrating.
- Dungeon map: `Shared/resources/worlds/Dungeons/Snake Pit.jm`, width 120, JSON `dict` of tiles
  (ground types + wall/object ids). Defines entrance, corridors, boss room -> the navigation map.

## Still to extract for M1
- Projectile properties (speed, lifetime, damage, size) per `projectileIndex` from Stheno's
  object XML in `Shared/resources/.../*.xml` (search the Stheno object entry).
- The dungeon MAP / setpiece: Snake Pit layout (entrance -> path -> boss room) for navigation.
- Status-effect mechanics: Confused (reversed movement), Petrify (no movement) durations.
- `ScaleHP2(20)` and base boss HP (fix a single-player value for the sim).

## Sim modeling plan (M1)
Model phases as an HP-gated state machine; aimed/rotating spreads; grenades as telegraphed AoE
circles applying Confused/Petrify; Stheno Swarm minions; Wander. Whole dungeon = navigate the
map from entrance to boss room (sparse-reward navigation) then the 3-phase fight. Game-faithful
renderer uses the real sprites/tiles from `Shared/resources`.
