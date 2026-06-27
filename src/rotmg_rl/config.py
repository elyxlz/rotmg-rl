"""Shared Snake Pit config + layout constants (pure Python: dataclass + numpy tables only).

This is the single Python home for the env's tunable config (`DungeonConfig`) and the obs-layout /
calibration constants that every consumer imports (the C env in `pufferlib/ocean/dungeon/dungeon.h` is the single source
of truth for the *dynamics*; this module mirrors its config + layout for the Python layers — policies,
the real-server obs bridge, the eval/render wrappers). It deliberately pulls in no gym / pufferlib /
torch so it can be imported anywhere (the cheap dev box, the deploy bridge, the scenario tests).

Calibration (source: vendor/betterSkillys/source): dt=100ms, tiles/tick = projectile Speed/100,
cooldown ticks = ms/100. The faithful loadout + boss/snake behavior are documented on the fields below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VIS_RADIUS = 15  # source VISIBILITY_RADIUS
BOSS_RETURN_RADIUS = 1.0  # ReturnToSpawn(0.7, 1): the boss anchors within 1 tile of spawn
GRID = 2 * VIS_RADIUS + 1  # 31x31 egocentric window
MOVE_DIRS = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1).astype(np.float32)
N_AIM = 32
AIM_DIRS = np.stack([np.cos(np.arange(N_AIM) * 2 * np.pi / N_AIM), np.sin(np.arange(N_AIM) * 2 * np.pi / N_AIM)], 1).astype(np.float32)

CH_WALL, CH_ENEMY, CH_EBULLET, CH_EBVX, CH_EBVY, CH_PBULLET, CH_GRENADE = range(7)
NUM_CH = 7
NUM_SCALARS = 8  # hp, mp, spell_ready, boss_visible, confused, petrified, boss_hp_frac, boss_invuln
# Fog-of-war minimap: a global downsampled view the player builds up by exploring (no cheats). MM x MM
# cells, each covering a block of the full dungeon. Channels: terrain (discovered walkable +1 / wall
# -1 / fog 0), player cell, boss cell (only once the boss has been seen).
MM = 32
NUM_MM_CH = 3
MM_CH_TERRAIN, MM_CH_PLAYER, MM_CH_BOSS = range(3)
BX, BY, BVX, BVY, BLIFE, BDMG = range(6)  # bullet columns
EX, EY, EHP, ETIMER, ETYPE = range(5)  # enemy columns (type indexes SNAKE_TYPES)
GX, GY, GFUSE, GRAD, GDMG, GSTATUS = range(6)  # grenade columns (status: 0 confused, 1 petrify)

# Real Snake Pit enemies (EmbeddedData_SnakePitCXML.xml + BehaviorDb.SnakePit.cs). Calibrated:
# bullet t/tick = Speed/100, life ticks = LifetimeMS/100, follow t/tick = followSpeed*0.5 (Wander/
# Follow advance speed*BehaviourTickTime per 200ms tick). Columns:
# hp, defense, dmg, bvspeed, blife(ticks), count, arc(rad), cooldown(ticks), follow, follow_speed,
# acquire_range, shoot_range.
ST_HP, ST_DEF, ST_DMG, ST_BVS, ST_BLIFE, ST_CNT, ST_ARC, ST_CD, ST_FOLLOW, ST_FSPD, ST_ACQ, ST_SRANGE = range(12)
SNAKE_TYPES = np.array(
    [
        [5.0, 0.0, 20.0, 0.6, 20.0, 1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 20.0],  # Pit Viper (HP5)
        [200.0, 5.0, 25.0, 0.8, 20.0, 3.0, np.radians(5.0), 10.0, 1.0, 0.5, 10.0, 15.0],  # Fire Python (3-shot, Follow)
        [200.0, 5.0, 25.0, 0.8, 30.0, 1.0, 0.0, 10.0, 1.0, 0.5, 10.0, 20.0],  # Yellow Python (Follow)
        [500.0, 10.0, 50.0, 0.8, 30.0, 3.0, np.radians(5.0), 10.0, 1.0, 0.5, 10.0, 15.0],  # Greater Pit Snake (3-shot)
        [500.0, 10.0, 50.0, 0.6, 30.0, 1.0, 0.0, 3.0, 1.0, 0.5, 10.0, 15.0],  # Greater Pit Viper (cd 300ms)
    ],
    np.float32,
)
SNAKE_WEIGHTS = np.array([0.40, 0.22, 0.15, 0.15, 0.08], np.float32)  # spawn mix: many fillers, few greaters
SNAKE_TIMER_JITTER = 10  # initial shoot-timer desync (ticks)


@dataclass
class DungeonConfig:
    player_speed: float = 0.773  # MoveSpeed at SPD 50: 0.004 + 50/75*(0.0096-0.004) = 0.00773 t/ms -> 0.773 t/tick
    player_radius: float = 0.4
    max_steps: int = 4000
    activation_range: float = 20.0
    spawn_in_room_prob: float = 0.0  # curriculum: prob of spawning near the boss (practice the fight)
    # ring distance (tiles) from the boss for the in-room spawn; ramp it up to teach
    # navigate-under-threats incrementally (6 = in-room, ~107 = entrance distance)
    spawn_in_room_radius: float = 6.0
    random_spawn_prob: float = 0.0  # spawn at a random walkable tile anywhere (coverage, less overfitting)
    # Wizard, realistic maxed T7 Snake-Pit loadout (gear stats are StatDataType bonuses resolved via
    # GetStatIndex, NOT raw core-stat ids): 670 base HP + 100 ring (stat0) + 40 spell (stat0) = 810;
    # 385 base MP + 30 robe (stat3=MaxMana) + 40 spell (stat3=MaxMana) = 455; DEF = 0 base + 8 robe
    # (stat21=Defense) = 8 (Wizard robes give little raw DEF); WIS = 60 + 3 robe + 7 spell = 70; VIT 40.
    # Regen (Player.cs HandleRegen): hp/s = 1 + 0.36*VIT = 15.4 -> 1.54/tick, mp/s = 1 + 0.24*WIS = 17.8
    # -> 1.78/tick. The HP regen is why the real fight is survivable -- the player recovers between hits.
    player_hp_max: float = 810.0
    player_mp_max: float = 455.0
    player_defense: float = 8.0  # robe stat21=Defense; incoming damage reduced by the real clamp
    damage_floor: float = 0.1  # DamageWithDefense floor: dealt = max(raw*floor, raw - defense)
    mp_regen: float = 1.78  # (1 + 0.24*WIS)/s at WIS 70, per 100ms tick
    hp_regen: float = 1.54  # (1 + 0.36*VIT)/s at VIT 40, per 100ms tick
    # Staff of Destruction (T7): 2 parallel shots (ArcGap 0), Speed 180 (1.8 t/tick), Life 475ms.
    # Damage is the raw item [45,85] * the maxed-ATT attack multiplier (0.5 + 75/75*1.5 = 2.0) =
    # [90,170] (the BulletNova spell does NOT take the multiplier; only the staff). Fire rate is the
    # real GetAttackFreq at DEX 75 = 0.008 = 125ms/shot = 1.25 ticks, carried fractionally (see step).
    staff_cooldown: float = 1.25  # ticks/shot, fractional accumulator -> ~8 shots/s
    staff_num: int = 2
    staff_dmg_lo: float = 90.0
    staff_dmg_hi: float = 170.0
    staff_speed: float = 1.8
    staff_life: float = 4.75  # Life 475ms = 4.75 ticks (range ~8.55 tiles)
    staff_radius: float = 0.5
    staff_offset: float = 0.5
    # Burning Retribution Spell (T7): 360-degree BulletNova from the player, 20 bullets, [95,185] dmg
    # (raw, NOT attack-scaled), Speed 160 (1.6 t/tick), Life 1000ms (10 ticks, ~16-tile range), MP 90.
    spell_cost: float = 90.0
    spell_cooldown: int = 0
    spell_num: int = 20
    spell_dmg_lo: float = 95.0
    spell_dmg_hi: float = 185.0
    spell_speed: float = 1.6
    spell_life: int = 10
    # snakes: real variety in SNAKE_TYPES (HP 5-500, dmg 20-50). snake_speed = wander drift std,
    # snake_radius = collision size; per-type combat stats live in SNAKE_TYPES, not config.
    n_snakes: int = 40
    n_snakes_jitter: int = 0  # per-episode +/- band around n_snakes (difficulty schedule spreads a batch around d)
    snake_speed: float = 0.15
    snake_radius: float = 0.5
    # boss (Stheno the Snake Queen): 7500 HP, DEF 19, phases at 66%/33%.
    boss_hp_max: float = 7500.0
    boss_radius: float = 2.0
    boss_defense: float = 19.0
    boss_wander_speed: float = 0.15  # Wander(0.3) in P1 (random drift); stationary in P2/P3
    boss_return_speed: float = 0.35  # ReturnToSpawn(0.7, 1): gentle pull toward spawn in P1 so it can't drift out of the room
    boss_shoots: bool = True  # curriculum stage 0 sets False: passive target, learn aim+kill first
    opening_invuln_ticks: int = 10  # Start state: 1.0s invuln taunt before P1
    invuln_ticks: int = 15  # P2/P3 transition invuln (1.5s)
    # Blade (boss projectile 0): Speed 70 (0.7 t/tick), Life 1500ms (15 ticks, ~10.5-tile range),
    # dmg 100. P1 acquires only point-blank (radius 2); P3 aims at range (radius 30).
    blade_cd: int = 15
    blade_radius_p1: float = 2.0
    blade_radius_p3: float = 30.0
    ebullet_speed: float = 0.7
    ebullet_life: int = 15
    ebullet_dmg: float = 100.0
    ebullet_radius: float = 0.4
    max_bullets: int = 8192
    # grenades (telegraphed AoE -> status, 1500ms fuse). Confused r3.5 dmg150 (P1/P2, acquire r11);
    # Petrify r1.5 dmg75 (P3, 8 in a radial fan thrown 6 tiles out, cardinals cd1.5s/diagonals cd3.0s).
    grenade_fuse: int = 15
    grenade_cd_p1: int = 15
    grenade_cd_p2: int = 10
    grenade_cd_p3_diag: int = 30
    grenade_range_confuse: float = 11.0
    grenade_petrify_dist: float = 6.0
    grenade_radius_confuse: float = 3.5
    grenade_dmg_confuse: float = 150.0
    grenade_radius_petrify: float = 1.5
    grenade_dmg_petrify: float = 75.0
    confused_ticks: int = 10
    petrify_ticks: int = 10
    # Stheno Swarm minions (Reproduce) -- OMITTED by default (optional, no-player-gated); kept tunable.
    minion_max: int = 5
    minion_cd: int = 15
    minion_hp: float = 30.0
    enable_grenades: bool = True  # curriculum can disable for early stages
    enable_minions: bool = False
    # rewards (exploration-based, no global pathfinding). Scaled to PufferLib's roughly -1..1 rule:
    # per-step signals tiny, a full clean clear totals ~1-2.5 total episode reward.
    rew_explore: float = 0.01  # per newly-visited tile
    rew_kill: float = 0.1  # per snake killed
    rew_boss_dmg: float = 1.0  # dominant signal, applied normalized by boss_hp_max (full boss ~= 1.0)
    rew_reach: float = 0.3
    rew_survive: float = 0.0  # NO reward for existing (paid the agent to flee -> cleared fell)
    rew_damage_taken: float = 0.5  # applied normalized by player_hp_max (full HP lost == 0.5)
    rew_clear: float = 1.0
    rew_death: float = 0.5  # small: don't make it terrified to engage the boss
    rew_step: float = -0.001  # net-negative existence: must make progress (kill the boss)
    # potential-based distance-to-boss shaping (privileged TRAINING signal, not in the obs): while
    # navigating (pre-fight), reward closing euclidean distance to the boss. Default 0 = off (the
    # explore reward alone never finds the boss across the ~107-tile fixed map). Turn on for the
    # navigation curriculum; the deployed policy uses no reward, only the fog-of-war minimap.
    rew_approach: float = 0.0
