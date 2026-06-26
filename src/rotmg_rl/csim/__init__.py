"""Fast C (PufferLib Ocean) port of the faithful Snake Pit dungeon sim.

The Python `rotmg_rl.sim.dungeon.DungeonEnv` is the oracle; this package is a bit-faithful C
reimplementation for blazing-fast training. See `tests/test_csim_parity.py` for the acceptance
parity test (same seed + actions -> matching obs and rewards).
"""
