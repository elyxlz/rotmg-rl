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
    threat is met in near-isolation and then grows with d.

    DOMAIN RANDOMIZATION: the enemy-DYNAMICS ranges below WIDEN with d (narrow/easy at low d, full at
    d=1). The env samples a fresh value in each range every episode, so the policy faces a DISTRIBUTION
    of enemy configurations -- the real server one in-distribution sample -- and learns a robust transfer
    skill instead of memorizing one exact layout. The FIXED map (walls/floor/boss/entrance/geodesic) and
    the per-type archetype identities are NOT randomized. At d=0 every range collapses to its identity
    (density 1..1, no hot bias, no jitter, fixed boss/player HP), so the easy curriculum is deterministic.

    The old enable_grates anchor-restock (which sustained the whole-map density and pushed a hold-and-
    clear deadlock) is dropped: the sustained-vs-thinning variation is now just one end of the density
    distribution (density_hi up to ~1.4 + a few dense hot regions), never a forced hold strategy."""
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
        # 1) density / count: per-episode multiplier on the active count, widening from 1.0 (exact) to
        #    ~0.5..1.5 at d=1, so the swarm the policy crosses is sometimes thinner, sometimes denser.
        "density_lo": lerp(1.0, 0.5),
        "density_hi": lerp(1.0, 1.5),
        # 1b) WHICH regions are dense: a few random hot anchors per episode boost their neighborhood and
        #     suppress the rest (cold), so a chokepoint like (38,43) is sometimes packed (~9+) and
        #     sometimes light (~3). The radius/bias widen with d into a strong per-episode density field.
        "n_hot_regions": round(lerp(0.0, 4.0)),
        "hot_radius": lerp(12.0, 18.0),
        "hot_bias": lerp(0.0, 4.0),
        # 2) spawn-position jitter: perturb authored tiles by up to a few tiles (still on floor) so
        #    exact-position memorization fails.
        "spawn_jitter": round(lerp(0.0, 3.0)),
        # 3) bullet fire timing/phase: per-enemy phase desync grows, and a per-episode +/-40% cadence
        #    scale makes the bullet patterns vary (robust dodging, not a memorized rhythm).
        "fire_phase_jitter": round(lerp(10.0, 20.0)),
        "fire_cd_jitter": lerp(0.0, 0.4),
        # 4) aggression / Follow: jitter acquireRange + move speed so convergence density varies.
        "acq_jitter": lerp(0.0, 0.4),
        "fspd_jitter": lerp(0.0, 0.3),
        # 5) boss HP: sample the real ScaleHP2 band (solo 7500 .. populated 16500) so it never overfits
        #    one HP. At d=0 lo>=hi collapses to the fixed BOSS_HP; the band opens up with d.
        "boss_hp_lo": lerp(BOSS_HP, 7500.0),
        "boss_hp_hi": lerp(BOSS_HP, 16500.0),
        # 6) player HP/DEF: small +/- jitter around the real 670 / 25 for robustness.
        "player_hp_jitter": lerp(0.0, 0.1),
        "player_def_jitter": lerp(0.0, 0.1),
        "enable_grenades": 1 if d > 0.15 else 0,
        "enable_minions": 0,  # superseded by the protective swarm
        # the boss's protective, replenishing, bullet-blocking swarm: the defining mechanic the real
        # Stheno is walled by. Ramped in at high d (like the boss/grenade threats) so it is met after
        # the policy has the basics, then becomes the hard wall the retrain must learn to penetrate.
        "enable_swarm": 1 if d > 0.45 else 0,
        # Grate replenishment folded into the density distribution as ONE per-episode sample (never the
        # old always-on forcing): with prob grate_prob the chokepoint is SUSTAINED dense (~9 converging
        # Greaters, the real wall), otherwise it THINS by attrition (~3). Sometimes-on at ~0.4 by d=1, so
        # the policy must handle both ends and can't settle into the hold-and-clear deadlock.
        "enable_grates": 0,
        "grate_prob": lerp(0.0, 0.4),
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
