"""Measure the rotation between the policy's COMMANDED move direction and the player's ACTUAL
world displacement, from the trace policy.py writes to /dev/shm/rotmg_trace.txt.

This is the diagnostic that found the deploy-time navigation bug: the trained policy moves in the
sim's WORLD frame, but the real client realizes WASD in its ISO/SCREEN frame, rotated 45 deg. The
mean commanded->actual rotation was a clean -45 deg; after setting config.MOVE_FRAME_ROT_DEG=45 it
collapsed to ~0. Re-run this after any input/aim change to confirm intent matches motion.

  trace columns (per tick): tick  move_idx  cmd_dx  cmd_dy  pos_x  pos_y

  python trace_analysis.py [/dev/shm/rotmg_trace.txt]
"""

from __future__ import annotations

import collections
import math
import sys


def main(path: str) -> None:
    rows = []
    for ln in open(path):
        p = ln.split()
        if len(p) >= 6:
            try:
                rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
            except ValueError:
                pass
    diffs = []
    for i in range(len(rows) - 1):
        mv, mdx, mdy, px, py = rows[i]
        _, _, _, px2, py2 = rows[i + 1]
        adx, ady = px2 - px, py2 - py
        if mv == 0 or math.hypot(mdx, mdy) < 1e-6 or math.hypot(adx, ady) < 0.10:
            continue
        cmd_ang = math.atan2(mdy, mdx)
        act_ang = math.atan2(ady, adx)
        diffs.append(math.degrees((act_ang - cmd_ang + math.pi) % (2 * math.pi) - math.pi))
    if not diffs:
        print("no movement samples (player never moved / wrong-player latch / died at entry)")
        return
    sx = sum(math.cos(math.radians(d)) for d in diffs)
    sy = sum(math.sin(math.radians(d)) for d in diffs)
    print("n_moved=%d  circular-mean rotation (cmd->actual) = %.1f deg  (target ~0)" % (len(diffs), math.degrees(math.atan2(sy, sx))))
    print("buckets(deg):", dict(sorted(collections.Counter(round(d / 45.0) * 45 for d in diffs).items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/rotmg_trace.txt")
