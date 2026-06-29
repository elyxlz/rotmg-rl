"""The server-as-sim difficulty schedule: one d in [0,1] -> the FOUR real-engine knobs the C#
server applies per episode (no synthetic domain randomization -- the real game RNG is the only
variation). This is the server-as-sim analogue of schedule.py's difficulty_config(), but it drives
the REAL betterSkillys engine, so it touches only the legitimate training aids the server exposes:
the agent spawn distance, agent HP, agent DEF, and boss HP. The dungeon/enemies/AI/boss mechanics
stay UNMODIFIED; d=1 == the real deliverable conditions.

The curriculum-depth objective itself (CURRICULUM_RUNGS, curriculum_depth) is reused UNCHANGED from
schedule.py -- the ladder/depth math is engine-independent (it consumes only per-rung clear rates)."""

from __future__ import annotations

# Reuse the depth objective + rungs + the cosine d(t) ramp UNCHANGED from the C-sim schedule.
from rotmg_rl.schedule import (  # noqa: F401  (re-exported so the trainer/eval import one place)
    CURRICULUM_CLEAR_THRESHOLD,
    CURRICULUM_RUNGS,
    CURRICULUM_TIE_BREAK_SCALE,
    LADDER_EPISODES,
    curriculum_depth,
    difficulty_at,
)

# ---- the REAL deliverable conditions (d=1). These are the live-server values the proven loop must
# reach; every lever interpolates from an easy anchor (d=0) to exactly these at d=1. -----------------
REAL_SPAWN_GEO_DIST = -1  # -1 == the real pit ENTRANCE (the full maze; the C# server's sentinel)
# The DECIDED deploy loadout: a maxed Wizard with a T7 Staff of Destruction + a T7 Burning
# Retribution Spell and NOTHING else -- NO armor, NO ring. So d=1 is the real FRAGILE char:
#   HP  = base maxed Wizard 670 + the spell slot's ActivateOnEquip +40 = 710 (no ring).
#   DEF = 0 (no armor; the earlier 25 was a leather-armor char we no longer deploy).
#   MP  = base maxed Wizard 385 + the spell's +40 = 425 (env-only SIM_AGENT_MP, not ramped).
# The boss + spawn are unchanged (real Stheno HP, real entrance).
REAL_AGENT_HP = 710  # maxed Wizard base 670 + spell-slot +40 HP (no ring)
REAL_AGENT_DEF = 0   # no armor -> 0 flat reduction (the fragile no-armor Wizard)
REAL_BOSS_HP = 7500  # real solo Stheno HP on the live server (== schedule.BOSS_HP)

# ---- the EASY anchors (d=0): the gentlest honest config in which the proven loop already clears
# (~97%). Survivability (agent HP) is the dominant difficulty gradient the diagnosis established, so
# the agent starts very tanky and the boss very soft, near the boss, with extra DEF. -----------------
EASY_AGENT_HP = 5000  # very tanky: undodged bullets rarely kill (the d=0 proof config)
EASY_AGENT_DEF = 40  # extra flat reduction on top of the real 25
EASY_BOSS_HP = 1500  # soft boss: a handful of landed shots clear it
EASY_SPAWN_GEO_DIST = 12  # spawn a SHORT geodesic walk from the boss (a near-isolation fight)
# The geodesic distance the real entrance sits at (the full maze). The spawn ramps from the easy
# near-boss tile out to this, then SNAPS to -1 (the true entrance sentinel) at d=1, so the policy
# meets the entire authored maze + chokepoint exactly as the deliverable presents it. MEASURED from a
# real boot (SIM_SPAWN_GEO_DIST=-1 spawns at (110.5,21.5) geo_dist=150, max reachable 215): the Snake
# Pit entrance sits 150 geodesic tiles from the boss, so the spawn ramp climbs toward that before the
# d=1 snap, keeping the ramp continuous (d=0.99 ~= 148 tiles, then -1 == the same ~150-tile entrance).
ENTRANCE_GEO_DIST = 150


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def server_difficulty_config(d: float) -> dict:
    """Map d in [0,1] -> the four real-engine knobs the C# server reads per episode. Monotonic in d,
    every lever moving smoothly, and d=1 == the real deliverable conditions exactly. Returns ints
    (the shm config region + the server env vars are integer-typed).

    Lever rationale (survivability is the dominant gradient, per the diagnosis):
      - agent_hp:  5000 (very tanky) -> 710 (real, no-ring Wizard). The primary difficulty axis: as HP falls, an
                   undodged bullet is increasingly lethal, so the policy MUST learn to dodge.
      - boss_hp:   1500 (soft) -> 7500 (real). More HP == more landed shots == longer in the bullet
                   storm == more dodging required. Moves with d so ticks-to-kill rises smoothly.
      - agent_def: 40 (extra) -> 0 (real, no armor). Flat per-hit reduction; trims each hit while HP is high,
                   relaxes to the real value as the rest of the difficulty arrives.
      - spawn:     12 geodesic tiles (near-boss, near-isolation) -> the entrance maze. Ramps the
                   navigate-in path length, then SNAPS to the real entrance (-1) at d=1.

    No synthetic enemy/AI/boss randomization: the real engine's own RNG (snake spawn, boss wander,
    bullet patterns) is the variation. That is the deliberate departure from the C-sim schedule,
    which WIDENS synthetic dynamics ranges with d -- here the dynamics are the real game, untouched."""
    d = _clamp01(d)
    lerp = lambda a, b: a + (b - a) * d  # noqa: E731

    agent_hp = round(lerp(EASY_AGENT_HP, REAL_AGENT_HP))
    agent_def = round(lerp(EASY_AGENT_DEF, REAL_AGENT_DEF))
    boss_hp = round(lerp(EASY_BOSS_HP, REAL_BOSS_HP))

    # Spawn distance ramps out to the entrance, then snaps to the real entrance sentinel (-1) at d=1
    # so the deliverable is the literal real-entrance maze, not "a tile ~95 away". The snap happens
    # only at d==1.0; for every d<1 the spawn is a finite geodesic distance (a known navigate-in path).
    if d >= 1.0:
        spawn_geo_dist = REAL_SPAWN_GEO_DIST  # -1: the real entrance
    else:
        spawn_geo_dist = round(lerp(EASY_SPAWN_GEO_DIST, ENTRANCE_GEO_DIST))

    return {
        "spawn_geo_dist": spawn_geo_dist,
        "agent_hp": agent_hp,
        "agent_def": agent_def,
        "boss_hp": boss_hp,
    }
