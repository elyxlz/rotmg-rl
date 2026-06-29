"""The continuous-difficulty schedule: one d(t) in [0,1] that drives every env lever jointly so the
policy always faces slightly-harder-than-mastered (no phase cliffs). Pure functions over plain dicts,
shared by training, eval, and the sweep."""

from __future__ import annotations

import math
import statistics

# The full authored roster: every enemy the real .jm places (== len(snakepit_map.authored_snakes),
# pinned by test_snakepit_map). The curriculum ramps n_snakes from 0 up to this, so d=1 activates the
# ENTIRE authored map -- the real ~405-enemy layout incl. the entrance chokepoint -- not a random scatter.
N_SNAKES_MAX = 405
BOSS_HP = 7500.0  # real solo Stheno HP on the live server (no-spectator runs show 7500)

# Curriculum-depth objective: the sweep maximizes how FAR UP the difficulty ladder a policy can clear,
# not the d=1 clear rate (which is 0 for every config on the authored map at a sweep budget -> no
# gradient). Purely clear-based -- no geodesic/survival/boss-damage terms. The per-rung clear rate is
# the only input; the reward cocktail stays a SEARCH variable that Protein tunes to maximize this depth.
CURRICULUM_RUNGS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CURRICULUM_CLEAR_THRESHOLD = 0.5  # "can clear" = >= 50% of episodes at that rung
LADDER_EPISODES = 12  # eval episodes PER rung (modest -> the 10-rung ladder stays cheap vs. training)
# tiny clear-based tie-break so two configs that clear NOTHING above threshold still differ by their
# easiest-rung clear rate (a config 40%-clearing d=0.1 beats one that clears 0%). Stays well below the
# smallest real depth (~0.1), so it never collides with a config that actually passes a rung.
CURRICULUM_TIE_BREAK_SCALE = 0.05


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
        # Snake Grate replenishment: the pit's continuous snake source that keeps the authored clusters
        # (incl. the converging Greater pack at the boss-approach chokepoint) stocked instead of thinning
        # by attrition. Ramped in with the swarm at high d, so the early curriculum still faces the gentle
        # static roster and the hard, sustained-density chokepoint is the d~1 wall the retrain must learn.
        "enable_grates": 1 if d > 0.45 else 0,
        "boss_shoots": 1 if d > 0.05 else 0,
        "blade_cd": round(15 + (40 - 15) * (1.0 - _clamp01(d / 0.4))),  # 40 (gentle) -> 15 (real) by d=0.4
    }


def curriculum_depth(
    rates: dict[float, float],
    threshold: float = CURRICULUM_CLEAR_THRESHOLD,
    tie_break_scale: float = CURRICULUM_TIE_BREAK_SCALE,
) -> float:
    """Curriculum depth in [0,1]: the highest difficulty d at which the policy still clears the dungeon
    at >= `threshold`, linearly interpolated between the bracketing rungs for a smooth continuous score.
    Input is ONLY the per-rung clear rate (rates[d] in [0,1]) -- no reward-shaping terms.

    Robustness to small-sample non-monotone noise: the rung curve is median-of-3 smoothed first, so a
    lone lucky clear at a hard rung can't inflate the depth and a lone dip at an easy rung can't sink it.
    Depth is then the threshold crossing on the smoothed curve.

    Edge cases:
      - clears >= threshold through the hardest rung -> depth = 1.0
      - never reaches threshold anywhere -> depth = tie_break_scale * (easiest-rung clear rate), so two
        all-zero configs read 0.0 and a config that at least clears the easy rung scores slightly higher.
    """
    rungs = sorted(rates)
    vals = [rates[d] for d in rungs]
    # median-of-3 with edge replication so every window is odd-length (a 2-wide edge median averages,
    # which would let a lone spike at the top rung leak into its neighbor). Now a fluke clear is dropped.
    padded = [vals[0], *vals, vals[-1]]
    smoothed = [statistics.median(padded[i : i + 3]) for i in range(len(vals))]

    passing = [i for i, v in enumerate(smoothed) if v >= threshold]
    if not passing:
        return tie_break_scale * vals[0]
    top = max(passing)
    if top == len(rungs) - 1:  # clears the hardest rung -> full depth
        return rungs[-1]
    # interpolate the threshold crossing between the deepest passing rung and the next (failing) rung.
    d_lo, d_hi = rungs[top], rungs[top + 1]
    r_lo, r_hi = smoothed[top], smoothed[top + 1]
    frac = (r_lo - threshold) / (r_lo - r_hi)  # r_lo >= threshold > r_hi, so frac in [0,1]
    return d_lo + frac * (d_hi - d_lo)


def apply_difficulty(args: dict, d: float, rew_approach: float, n_snakes_max: int) -> None:
    """Write the d-derived env config (+ the swept reward levers) into args['env'] in place."""
    for k, v in difficulty_config(d, n_snakes_max).items():
        args["env"][k] = v
    args["env"]["boss_hp_max"] = args["env"].get("boss_hp_max", BOSS_HP)
    # navigation shaping is only useful once the spawn is far from the boss; off while the fight is
    # still in-room (low d), scaled by the swept base rate otherwise.
    args["env"]["rew_approach"] = rew_approach if d > 0.2 else 0.0
