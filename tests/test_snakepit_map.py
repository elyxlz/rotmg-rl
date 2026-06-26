"""The real Snake Pit map parses into a sane navigable grid."""

from rotmg_rl.sim.snakepit_map import load_jm


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
