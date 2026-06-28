"""policy_server: line-delimited JSON bridge between the nrelay real-game client (TypeScript) and
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

--record <path.mp4> renders a POV frame per tick (the milestone-5 fallback recording) and writes the
mp4 on stdin EOF / SIGTERM. The stale v1 (deploy/policy_server.py) is reference, not reused.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys

from rotmg_rl.deploy.obs import RealObsBuilder, action_to_intent
from rotmg_rl.deploy.policy import PolicyRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/curriculum/finish.pt")
    ap.add_argument("--hidden", type=int, default=256, help="policy hidden size (must match the checkpoint's arch)")
    ap.add_argument("--num-layers", type=int, default=2, help="policy LSTM layers (must match the checkpoint's arch)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--record", default=None, help="render a POV mp4 of the real-server rollout to this path")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    runner = PolicyRunner(args.checkpoint, hidden=args.hidden, num_layers=args.num_layers)
    obs = RealObsBuilder()
    frames: list = []

    def save_and_exit(*_):
        if args.record and frames:
            import imageio.v2 as imageio

            imageio.mimsave(args.record, frames, fps=args.fps)
            print(json.dumps({"saved": args.record, "frames": len(frames)}), flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_and_exit)

    if args.record:
        from rotmg_rl.deploy.render import render_frame

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
        if args.record:
            frames.append(render_frame(obs, msg))
        print(json.dumps({"action": action, "intent": action_to_intent(action)}), flush=True)

    save_and_exit()


if __name__ == "__main__":
    main()
