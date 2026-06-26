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

## Player character (CHOSEN: Wizard) + full-fidelity plan
User directive: make the sim AS CLOSE to the real game as possible, done properly. Chosen
character = **Wizard** with **Staff** + **Spell** (the iconic default; ranged 2-shot + AoE nuke
that makes the 7500-HP boss killable). Real stats from betterSkillys resources:
- Wizard: base HP 100 (max 670), MP 100 (max 385); model the max-level character.
- Staff of Destruction: NumProjectiles 2, ArcGap 0 (parallel), dmg 45-85, Speed 180,
  LifetimeMS 475, RateOfFire 1.
- Spell of Galactic Creation: dmg 110-205/shot, Speed 160, LifetimeMS 1000 (classic Wizard
  spell = a burst of ~20 shots in an arc toward the cursor; MP cost ~100). The big nuke.

### Unit calibration (real -> sim)
- Tick dt = 0.1s (10 ticks/s). ROTMG projectile tiles/sec ~= XML Speed / 10:
  staff 180 -> 18 t/s -> 1.8 tiles/tick (life 475ms -> ~8.5 tiles);
  boss proj0 70 -> 7 t/s -> 0.7 tiles/tick (life 1500ms -> ~10.5 tiles);
  spell 160 -> 16 t/s -> 1.6 tiles/tick.
- Player move ~5 tiles/s -> 0.5 tiles/tick (Wizard SPD). Staff fire ~5 shots/s -> every ~2 ticks.
- Damage per shot random in [min,max]. MP regen ~ a few/sec; Spell costs ~100 MP.

### Full-fidelity checklist (the "done properly" build)
1. Wizard: HP/MP, 2-shot staff (real dmg/speed/life/rate), Spell ability (burst nuke, MP cost),
   MP regen. Action gains a CAST dimension.
2. Stheno full: 7500 HP, the 3-phase shoots (done), + grenades (telegraphed AoE -> Confused
   /Petrify status on player), + Stheno Swarm minions (Reproduce).
3. Path enemies: snakes along the corridors (Pit Snakes/Vipers/Pythons at the map snake tiles),
   wander + shoot.
4. Player status effects: Confused (reversed move), Petrify (no move), timers.
5. A simple DEBUG renderer (shapes/colors, not faithful) to follow the policy visually.

## FAITHFUL REDESIGN (user feedback: the arcade sim is NOT faithful)
The first sim cheated. Real game constraints to honor:
1. **Local viewport only.** The real client shows a small window around the character (~radius
   10-15 tiles). REMOVE the global geodesic field and the global direction-to-boss from the
   observation/reward. The agent sees only local surroundings and must EXPLORE.
2. **Exploration-based progression**, not path-to-known-boss. Reward = visiting NEW tiles +
   killing enemies + finding/damaging the boss + clearing. No global pathfinding breadcrumb.
3. **Fight through rooms.** The dungeon is full of snakes that shoot you; you fight/dodge
   through. Enemy stats: Pit Snake/Pit Viper HP 5, proj Speed 60; Fire Python HP 200, Speed 80,
   life 2000ms. Snakes: Wander + Shoot(~1s). Spawn them through the walkable rooms/corridors.
4. **Continuous mouse-aim.** Staff and Spell fire toward the MOUSE direction (continuous), not 8
   discrete dirs. Action -> continuous Box [move_x, move_y, aim_x, aim_y, shoot, cast], which
   also maps cleanly to driving the real client (WASD + cursor + click + spell key).
This makes the RL problem much harder (explore + fight with local vision only) but it is the
real game. Policy must become continuous (Box action). Viewport ~21x21 (radius 10).

## Sim modeling plan (M1)
Model phases as an HP-gated state machine; aimed/rotating spreads; grenades as telegraphed AoE
circles applying Confused/Petrify; Stheno Swarm minions; Wander. Whole dungeon = navigate the
map from entrance to boss room (sparse-reward navigation) then the 3-phase fight. Game-faithful
renderer uses the real sprites/tiles from `Shared/resources`.
