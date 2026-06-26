# Real game analysis — Snake Pit (from betterSkillys source, the ground truth)

Full mechanical analysis of the real game, read from `betterSkillys/source/WorldServer`. This is
the blueprint the faithful sim must match. Source citations in parentheses.

## 1. Time, units, projectiles (CONFIRMED — my calibration was right)
- Game is continuous, millisecond-based. Network/logic tick = 200ms (5 Hz) but movement and
  projectiles advance on elapsed ms; the client interpolates at high FPS. (`GameServer.cs`:
  `sleepTime = 200 - logicTime`).
- Projectile travel: `dist = elapsed_ms * Speed / 10000` (`ValidatedProjectile.cs`). So
  **tiles/sec = Speed/10**. With sim dt=100ms: tiles/tick = Speed/100.
  - Staff Speed 180 -> 18 t/s -> 1.8 t/tick; boss proj 70 -> 0.7 t/tick; spell 160 -> 1.6 t/tick.
  - Range = LifetimeMS * Speed/10000 tiles. Cooldowns: ms/100 = ticks (1500ms -> 15 ticks).

## 2. Player movement
- `speed = Stats.GetSpeed() * diff * 1.1` (`Player.Ground.cs`). GetSpeed() derives from the SPD
  stat; ROTMG range ~4 t/s (SPD 0) to ~9.6 t/s (SPD 75). Wizard is mid -> ~5-6 t/s.
- Movement is continuous in the aim/WASD sense; walls block.

## 3. Shooting + aiming (KEY: continuous, mouse-directed)
- Player shoots toward the MOUSE direction (continuous angle), auto-fires while held.
- Fire rate: `attackPeriod_ms = 1/AttackFrequency * 1/RateOfFire` (`Player.Shoot.cs`).
  AttackFrequency from DEX (~0.0015-0.008 /ms). Wizard high DEX -> ~5-8 shots/s.
- Staff of Destruction: NumProjectiles 2, ArcGap 0 (2 parallel shots), dmg 45-85, Speed 180,
  Life 475ms.
- Wizard Spell (ability): a burst of bolts toward the cursor direction, dmg 110-205 each,
  Speed 160, Life 1000ms, MP cost ~100. Fired toward the MOUSE.

## 4. Vision (KEY: local only, must explore)
- **`VISIBILITY_RADIUS = 15`** (`Player.Update.cs`). The player sees a 31x31 tile window around
  itself. NOT the whole map. The agent must EXPLORE to find the boss. -> sim observation = a
  local ~31x31 egocentric window; REMOVE the global geodesic field + global boss-direction.

## 5. Enemy behavior (the Shoot/Wander/Chase building blocks)
- `Shoot(radius, count, shootAngle, projectileIndex, rotateAngle, predictive, coolDown, ...)`
  (`Shoot.cs`): aims at the nearest/target player within `radius`; fires `count` bullets
  `shootAngle` deg apart (or 360/count); `rotateAngle` rotates the spread over time; `predictive`
  leads the target; halved by Dazed, blocked by Stunned.
- Snake Pit enemies (`EmbeddedData_SnakePitCXML.xml`): Pit Snake / Pit Viper HP **5**, proj
  Speed 60; Fire Python HP **200**, Speed 80, Life 2000ms. Snakes: Wander(0.3) + Shoot(~1s).
  They populate the rooms/corridors and shoot you -> you fight/dodge through.

## 6. Dungeon structure
- No hard internal gates (the "Snake Pit Key" is a realm-side unlock loot, not internal). The
  dungeon is open: navigate enemy-filled rooms to find the boss room, surviving their fire.
- Map: real `.jm`, 120x119, entrance Portal (110,21), boss Stheno (16,73). Snake objects mark
  enemy/decor tiles (159 of them).

## 7. Boss: Stheno the Snake Queen (BehaviorDb.SnakePit.cs)
- 7500 HP (ScaleHP2(20)). Activates when a player is within 20 tiles.
- Phase 1 (>66%): Wander(0.3); Reproduce "Stheno Swarm" (up to 5, every 1.5s); Grenade (r3.5,
  dmg150, every 1.5s, Confused); Shoot(3 bullets, 15deg, every 1.5s).
- Phase 2 (66-33%): invuln transition; Grenade (every 1s, Confused); Shoot(4, rotating 15deg,
  every 0.25s).
- Phase 3 (<33%): invuln transition; Shoot(3, 15deg, 1.5s) + Shoot(4, rotating 15deg, 0.5s);
  8 directional Grenades (Petrify) at 0/90/180/270 (1.5s) and 45s (3s).

## 8. Status effects (from boss grenades)
- Confused: reverses movement controls for a duration.
- Petrify/Petrified: cannot move for a duration.

## Faithful-sim implications (the rebuild)
1. Observation = local 31x31 egocentric window (walls, enemies, enemy bullets+vel, player
   bullets, items/boss-if-visible) + scalars (hp, mp, spell_ready). NO global geodesic/boss dir.
2. Action = continuous: move dir + aim dir (mouse) + shoot + cast. Maps to WASD + cursor + click
   + spell key on the real client.
3. Progression = exploration (visit new tiles) + kill enemies + find/clear boss. No path-to-boss
   breadcrumb.
4. Populate the dungeon with the real snakes (HP 5, Wander+Shoot). Fight/dodge through.
5. Boss: add grenades (AoE -> Confused/Petrify), Stheno Swarm minions, full 3-phase (have shoots).
6. Calibration (CONFIRMED): dt=100ms, tiles/tick = Speed/100, cooldown ticks = ms/100.
