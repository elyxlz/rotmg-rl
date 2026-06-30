#!/usr/bin/env python3
"""Render the server-as-sim learning curve as a compact text plot + summary."""
from __future__ import annotations

import csv
import sys


def sparkline(vals: list[float], lo: float = 0.0, hi: float = 1.0, width: int = 60) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not vals:
        return ""
    # resample to width buckets
    step = max(1, len(vals) // width)
    out = []
    for i in range(0, len(vals), step):
        chunk = vals[i:i + step]
        v = sum(chunk) / len(chunk)
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
        t = max(0.0, min(1.0, t))
        out.append(blocks[int(t * (len(blocks) - 1))])
    return "".join(out)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/server_proof_curve.csv"
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("no data")
        return
    steps = [int(r["step"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]
    done_rate = [float(r["done_rate"]) for r in rows]
    clear_rate = [float(r["clear_rate"]) for r in rows]
    n_clear = [int(r["n_clear"]) for r in rows]
    n_death = [int(r["n_death"]) for r in rows]
    n_timeout = [int(r["n_timeout"]) for r in rows]

    print(f"== server-as-sim learning curve ({len(rows)} log points, {steps[-1]:,} steps) ==\n")
    print(f"done_rate  [0..1]   {sparkline(done_rate, 0, 1)}")
    print(f"clear_rate [0..1]   {sparkline(clear_rate, 0, 1)}")
    rmax = max(reward) or 1.0
    print(f"reward     [0..{rmax:.2f}] {sparkline(reward, 0, rmax)}")
    print()
    # tabulated milestones
    print(f"{'step':>9} {'reward':>8} {'done_rt':>8} {'clear_rt':>9} {'clears':>7} {'deaths':>7} {'timeouts':>9}")
    idxs = sorted(set([0] + [int(len(rows) * f) for f in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)] + [len(rows) - 1]))
    for i in idxs:
        r = rows[i]
        print(f"{int(r['step']):>9} {float(r['reward']):>8.4f} {float(r['done_rate']):>8.3f} "
              f"{float(r['clear_rate']):>9.3f} {r['n_clear']:>7} {r['n_death']:>7} {r['n_timeout']:>9}")
    tot_clear = sum(n_clear)
    print()
    print(f"final clear_rate (last 5 windows): "
          f"{sum(clear_rate[-5:]) / max(1, len(clear_rate[-5:])):.3f}")
    print(f"final done_rate  (last 5 windows): "
          f"{sum(done_rate[-5:]) / max(1, len(done_rate[-5:])):.3f}")
    print(f"deaths across whole curve windows: {sum(n_death)}   timeouts: {sum(n_timeout)}")


if __name__ == "__main__":
    main()
