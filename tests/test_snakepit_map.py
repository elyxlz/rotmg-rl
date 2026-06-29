"""The real Snake Pit map parses into a sane navigable grid + authors the real enemy roster."""

import collections

from rotmg_rl.schedule import N_SNAKES_MAX
from rotmg_rl.sim.snakepit_map import authored_snakes, load_jm, snake_grates


def test_map_loads_with_expected_dims_and_floor():
    m = load_jm()
    assert (m.width, m.height) == (120, 119)
    assert m.tile_index.shape == (119, 120)
    assert int(m.tile_index.max()) < len(m.entries)  # all indices valid
    floor = int(m.walkable.sum())
    total = m.width * m.height
    assert 0 < floor < total  # some floor, some wall/void
    # walkable region is connected enough to be a real dungeon, not noise
    assert floor > 200


# The exact per-archetype enemy population the real .jm authors (exact id counts, NOT find_objects'
# substring counts). These are the fixed positions/types the env reproduces; if the map or the id->type
# mapping changes, this is the canonical count to re-pin.
EXPECTED_BY_TYPE = {0: 156, 1: 22, 2: 24, 3: 22, 4: 36, 5: 134, 6: 11}  # types index SNAKE_TYPES
EXPECTED_TOTAL = 405


def test_authored_snakes_match_real_population():
    """authored_snakes() returns every real Snake Pit enemy at its fixed tile, typed by archetype, all
    on walkable floor -- the layout the env spawns from instead of scattering snakes at random."""
    m = load_jm()
    snakes = authored_snakes(m)
    assert len(snakes) == EXPECTED_TOTAL
    assert collections.Counter(t for _, _, t in snakes) == EXPECTED_BY_TYPE
    assert all(m.walkable[y, x] for x, y, _ in snakes)  # every authored enemy stands on real floor
    assert len({(x, y) for x, y, _ in snakes}) == EXPECTED_TOTAL  # one enemy per tile (no overlaps)


def test_authored_roster_drives_curriculum_max():
    """The schedule's N_SNAKES_MAX is the full authored roster, so d=1 activates the entire real map."""
    assert len(authored_snakes()) == N_SNAKES_MAX


def test_authored_chokepoint_cluster_reproduced():
    """The hard entrance chokepoint the live policy dies in: a tight pack of 200-HP Fire Pythons + a
    500-HP Greater Pit Viper (+ a tanky Brown Python) around tiles (28-31, 41-46). Reproducing this exact
    cluster -- not a uniform scatter -- is the point of the authored map."""
    cluster = {(x, y, t) for x, y, t in authored_snakes() if 26 <= x <= 32 and 40 <= y <= 47}
    assert (31, 43, 4) in cluster  # Greater Pit Viper (500 HP)
    fire_pythons = {(x, y) for x, y, t in cluster if t == 1}  # Fire Python (200 HP, 3-shot)
    assert {(30, 43), (30, 44), (26, 46)} <= fire_pythons
    assert (28, 41, 6) in cluster  # Brown Python (DEF 20)


def test_snake_grates_match_real_placement():
    """The real .jm places exactly two Snake Grates (the pit's continuous Pit Snake/Viper source), both
    on walkable floor. These are the anchor tiles the env replenishes from to sustain the snake density."""
    m = load_jm()
    grates = snake_grates(m)
    assert set(grates) == {(61, 101), (47, 114)}  # the two authored Snake Grate tiles
    for x, y in grates:
        assert m.walkable[y, x]  # a grate sits on floor so its spawned children are reachable
