"""Read wandb metrics via the API to inform the next experiment (reliable; never log-grep).

    uv run --extra train python scripts/wandb_metrics.py            # latest run: summary + trends
    uv run --extra train python scripts/wandb_metrics.py <run_id>
    uv run --extra train python scripts/wandb_metrics.py --cleanup  # delete runs not in the latest group
"""

from __future__ import annotations

import sys

import wandb

PROJECT = "elyx/rotmg-dungeon"
TREND_KEYS = ("clear", "entropy", "policy_loss", "value_loss", "boss_hp", "episodic", "reward", "approx_kl")


def fmt(v):
    return round(v, 4) if isinstance(v, float) else v


def main() -> None:
    api = wandb.Api()
    a = sys.argv[1:]
    runs = list(api.runs(PROJECT, order="-created_at"))
    if a and a[0] == "--cleanup":
        keep = runs[0].group if runs else None
        for r in runs:
            if r.group != keep:
                print(f"delete {r.name} ({r.id}, {r.state}, group={r.group})")
                r.delete()
        print(f"kept group {keep}")
        return

    target = api.run(f"{PROJECT}/{a[0]}") if a else runs[0]
    print(f"=== {target.name} ({target.id}) state={target.state} group={target.group} ===")
    summary = {k: target.summary[k] for k in target.summary.keys() if not k.startswith("_")}
    for k in sorted(summary):
        print(f"  {k:28s} = {fmt(summary[k])}")
    keys = [k for k in summary if any(t in k.lower() for t in TREND_KEYS)]
    hist = target.history(keys=keys, samples=400, pandas=False) if keys else []
    print("--- trends (early -> latest) ---")
    for k in keys:
        vals = [row[k] for row in hist if k in row and row[k] is not None]
        if len(vals) > 1:
            step = max(1, len(vals) // 10)
            print(f"  {k:28s}: {[fmt(v) for v in vals[::step]]}")


if __name__ == "__main__":
    main()
