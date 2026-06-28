"""The continuous-difficulty schedule: one d(t) in [0,1] that drives every env lever jointly so the
policy always faces slightly-harder-than-mastered (no phase cliffs). Pure functions over plain dicts,
shared by training, eval, and the sweep."""

from __future__ import annotations

import math

# The full authored roster: every enemy the real .jm places (== len(snakepit_map.authored_snakes),
# pinned by test_snakepit_map). The curriculum ramps n_snakes from 0 up to this, so d=1 activates the
# ENTIRE authored map -- the real ~405-enemy layout incl. the entrance chokepoint -- not a random scatter.
N_SNAKES_MAX = 405
BOSS_HP = 7500.0  # real solo Stheno HP on the live server (no-spectator runs show 7500)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def difficulty_at(step: int, total: int, ramp_frac: float) -> float:
    """d(t) in [0,1]: cosine ease-in-out over the first ramp_frac of training, then hold at 1.0."""
    ramp_steps = max(1.0, ramp_frac * total)
    x = _clamp01(step / ramp_steps)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def difficulty_config(d: float, n_snakes_max: int = N_SNAKES_MAX) -> dict:
    """Map a single difficulty d in [0,1] to the env-config levers. Monotonic in d, and every lever
    moves a little for each step of d so there is no cliff. The boolean toggles (boss_shoots /
    grenades / minions) fire at LOW d, while the rest of the difficulty is still gentle, so each new
    threat is met in near-isolation and then grows with d."""
    d = _clamp01(d)
    n = round(n_snakes_max * d)
    lerp = lambda a, b: a + (b - a) * d  # noqa: E731
    return {
        # No-cheat deploy: the policy enters at the real portal and must NAVIGATE to the boss itself
        # (no /tppos). Ramp the spawn from in-room (low d: learn the fight in isolation) to far-from-boss
        # (high d: spawn at the entrance and navigate the maze into the swarm wall on its own).
        "spawn_in_room_prob": lerp(1.0, 0.05),
        "spawn_in_room_radius": lerp(6.0, 14.0),
        "n_snakes": n,
        "n_snakes_jitter": round(0.35 * n),
        "enable_grenades": 1 if d > 0.15 else 0,
        "enable_minions": 0,  # superseded by the protective swarm
        # the boss's protective, replenishing, bullet-blocking swarm: the defining mechanic the real
        # Stheno is walled by. Ramped in at high d (like the boss/grenade threats) so it is met after
        # the policy has the basics, then becomes the hard wall the retrain must learn to penetrate.
        "enable_swarm": 1 if d > 0.45 else 0,
        "boss_shoots": 1 if d > 0.05 else 0,
        "blade_cd": round(15 + (40 - 15) * (1.0 - _clamp01(d / 0.4))),  # 40 (gentle) -> 15 (real) by d=0.4
    }


def apply_difficulty(args: dict, d: float, rew_approach: float, n_snakes_max: int) -> None:
    """Write the d-derived env config (+ the swept reward levers) into args['env'] in place."""
    for k, v in difficulty_config(d, n_snakes_max).items():
        args["env"][k] = v
    args["env"]["boss_hp_max"] = args["env"].get("boss_hp_max", BOSS_HP)
    # navigation shaping is only useful once the spawn is far from the boss; off while the fight is
    # still in-room (low d), scaled by the swept base rate otherwise.
    args["env"]["rew_approach"] = rew_approach if d > 0.2 else 0.0
