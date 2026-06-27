"""policy_server v3: line-delimited JSON bridge between the nrelay real-game client (TypeScript) and
the trained CDungeonPolicy. Read one tick-state JSON object per line on stdin -> reconstruct the
9807-float obs (RealObsBuilder) -> run the recurrent policy -> write one action JSON line on stdout.

Protocol (one JSON object per line):
  in : {"reset": true}                                  -> {"reset_ok": true}   (new dungeon: clears LSTM + map/fog)
  in : {"map": {"w":W,"h":H}}                            (set once at MapInfo; may ride along a tick)
  in : {"tiles": [{"x","y","walkable"}], ...}            (incremental discovered tiles; may ride along a tick)
  in : {"enemy_shots": [{"origin_x","origin_y","angle","count","angle_inc","speed","lifetime","spawn_ms"}], ...}
  in : <tick> {"player":{x,y,hp,hp_max,mp,mp_max,confused,petrified},
               "enemies":[{x,y,hp,hp_max,is_boss,invuln}], "player_bullets":[{x,y}], "now_ms"}
  out: {"action": {"move","aim","shoot","cast"}, "intent": {"dx","dy","aim_x","aim_y","shoot","cast"}}

The stale v1 (deploy/policy_server.py, grid+scalars only, 8-dir action) is reference, not reused.
"""

from __future__ import annotations

import argparse
import json
import sys

from rotmg_rl.deploy.v3.obs import RealObsBuilder, action_to_intent
from rotmg_rl.deploy.v3.policy import PolicyRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/full_dungeon_95.pt")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    runner = PolicyRunner(args.checkpoint)
    obs = RealObsBuilder()
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)

        if "reset" in msg and msg["reset"]:
            runner.reset()
            obs = RealObsBuilder()
            print(json.dumps({"reset_ok": True}), flush=True)
            continue

        if "map" in msg:
            obs.set_map(msg["map"]["w"], msg["map"]["h"])
        if "tiles" in msg:
            obs.update_tiles([(t["x"], t["y"], t["walkable"]) for t in msg["tiles"]])
        if "enemy_shots" in msg:
            obs.add_shots(msg["enemy_shots"])

        if "player" not in msg:  # a map/tiles/shots-only frame -> ack, no action
            print(json.dumps({"ok": True}), flush=True)
            continue

        flat = obs.build(msg)
        action = runner.act(flat, greedy=args.greedy)
        print(json.dumps({"action": action, "intent": action_to_intent(action)}), flush=True)


if __name__ == "__main__":
    main()
