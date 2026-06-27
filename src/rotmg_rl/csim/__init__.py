"""The Snake Pit dungeon env — fast C (PufferLib Ocean) implementation.

`dungeon.h` is the single source of truth for the env dynamics. `single.py` wraps one env (numpy-only)
for eval + POV rendering; `render.py` paints the C render-state. Scenario tests live in
`tests/test_dungeon_scenarios.py`.
"""
