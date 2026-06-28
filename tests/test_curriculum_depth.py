"""The curriculum-depth objective (schedule.curriculum_depth) is a pure function of the per-rung clear
rate -- no torch, no env. These tests pin its interpolation, robustness, and edge cases, and prove the
signal it exists for: two configs that BOTH clear 0% at d=1 still get distinct depths, so the sweep has
gradient on the authored map where the d=1 clear rate is identically zero.
"""

import pytest

from rotmg_rl.schedule import CURRICULUM_CLEAR_THRESHOLD, CURRICULUM_RUNGS, curriculum_depth


def _ladder(values: list[float]) -> dict[float, float]:
    """Pair a list of per-rung clear rates with CURRICULUM_RUNGS (ascending d)."""
    assert len(values) == len(CURRICULUM_RUNGS)
    return dict(zip(CURRICULUM_RUNGS, values, strict=True))


def test_clears_every_rung_is_full_depth():
    assert curriculum_depth(_ladder([1.0] * 10)) == 1.0


def test_clears_nothing_is_zero_depth():
    assert curriculum_depth(_ladder([0.0] * 10)) == 0.0


def test_clean_crossing_interpolates_between_bracketing_rungs():
    # clears fully through d=0.5, then nothing from d=0.6 -> crossing the 0.5 threshold halfway between.
    rates = _ladder([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert curriculum_depth(rates) == pytest.approx(0.55, abs=1e-9)


def test_partial_decline_crossing():
    # a realistic decaying ladder; clears cross 0.5 between d=0.4 (0.7) and d=0.5 (0.4).
    rates = _ladder([1.0, 1.0, 0.9, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0])
    depth = curriculum_depth(rates)
    assert 0.4 < depth < 0.5


def test_lone_lucky_hard_clear_does_not_inflate_depth():
    # a single fluke clear at d=0.9 surrounded by zeros must NOT grant deep credit (median-of-3 kills it).
    rates = _ladder([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert curriculum_depth(rates) == 0.0


def test_lone_easy_dip_does_not_sink_depth():
    # one noisy dip at d=0.3 inside a passing region must not collapse the depth (median-of-3 fills it).
    rates = _ladder([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert curriculum_depth(rates) == pytest.approx(0.55, abs=1e-9)


def test_monotone_depth_in_difficulty_reached():
    # a policy that reaches deeper rungs scores strictly higher.
    shallow = curriculum_depth(_ladder([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    deep = curriculum_depth(_ladder([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert deep > shallow


def test_tie_break_separates_two_all_zero_at_d1_configs():
    # THE POINT: on the authored map every config clears 0% at d=1, so clear_d1 gives no gradient.
    # Curriculum depth still separates a config that climbs the ladder from one that barely starts.
    good = _ladder([1.0, 1.0, 0.9, 0.7, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0])  # clears well up the ladder
    bad = _ladder([0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # only scratches the easiest rung
    # both read identically zero on the old objective ...
    assert good[1.0] == 0.0 and bad[1.0] == 0.0
    # ... but the curriculum-depth objective spreads them.
    assert curriculum_depth(good) > curriculum_depth(bad) > 0.0


def test_below_threshold_everywhere_still_orders_by_easiest_rung():
    # two configs that never clear >= threshold at ANY rung still differ by their easiest-rung clear rate.
    a = curriculum_depth(_ladder([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    b = curriculum_depth(_ladder([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert a > b > 0.0
    # the tie-break stays well below any real passed-rung depth (>= ~0.1).
    assert a < 0.1


def test_threshold_constant_is_one_half():
    assert CURRICULUM_CLEAR_THRESHOLD == 0.5
