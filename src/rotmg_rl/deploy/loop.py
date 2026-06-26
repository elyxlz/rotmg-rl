"""Drive the sim entirely through the deploy bridge (the end-to-end deployment test).

Each tick: sim state -> RealmState dict (EnemyShoot events, as the real client receives) ->
PolicyRunner (reconstructs bullets, runs policy) -> action intent -> back into the sim. If the
champion clears via this path, the whole software deploy loop is proven; only the real-server
protocol-parsing layer remains untested.
"""

from __future__ import annotations

import numpy as np

from rotmg_rl.deploy.policy_server import PolicyRunner
from rotmg_rl.deploy.realm_state import ActionIntent, intent_to_action
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv


def build_realm_dict(env: SnakePitEnv, now: int) -> dict:
    shoots = [
        {
            "origin": [float(o[0]), float(o[1])],
            "base_angle": float(ba),
            "count": int(cnt),
            "arc_gap": float(g),
            "speed": float(sp),
            "spawn_time": float(st),
            "lifetime": float(lt),
        }
        for (o, ba, cnt, g, sp, st, lt) in env.shoot_log
        if st >= now - lt  # only bursts whose bullets are still alive
    ]
    return {
        "arena_size": env.cfg.arena_size,
        "player_pos": [float(env.player_pos[0]), float(env.player_pos[1])],
        "player_hp": float(env.player_hp),
        "player_hp_max": env.cfg.player_hp_max,
        "boss_pos": [float(env.boss_pos[0]), float(env.boss_pos[1])],
        "boss_hp": float(max(env.boss_hp, 0.0)),
        "boss_hp_max": env.cfg.boss_hp_max,
        "now": float(now),
        "enemy_shoots": shoots,
        "player_bullets": env.player_bullets.tolist(),  # the real client tracks its own shots
    }


def run_episode(runner: PolicyRunner, cfg: SnakePitConfig, seed: int) -> bool:
    env = SnakePitEnv(cfg)
    env.reset(seed=seed)
    runner.reset()
    for tick in range(cfg.max_steps):
        intent_d = runner.step(build_realm_dict(env, tick))
        intent = ActionIntent(
            move=np.array(intent_d["move"], np.float32),
            shoot=intent_d["shoot"],
            aim=np.array(intent_d["aim"], np.float32),
        )
        _, _, term, trunc, info = env.step(intent_to_action(intent))
        if term or trunc:
            return bool(info["cleared"])
    return False
