#!/usr/bin/env bash
# Manage dungeon training on the box reliably (avoid fragile inline ssh).
# Usage: ./scripts/box.sh {kill | train <train_dungeon args...> | follow | status}
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
mkdir -p logs checkpoints/run

cmd="${1:-status}"
shift || true
case "$cmd" in
  kill)
    pkill -9 -f scripts/train_dungeon.py 2>/dev/null
    pkill -9 -f scripts/curriculum_dungeon.py 2>/dev/null
    pkill -9 -f scripts/follow_along.py 2>/dev/null
    sleep 5
    echo "remaining: $(pgrep -cf 'scripts/(train_dungeon|curriculum_dungeon|follow_along)')  load:$(uptime | grep -oE 'average.*')"
    ;;
  train)  # launch ONE clean run; kills any existing first
    pkill -9 -f scripts/train_dungeon.py 2>/dev/null
    pkill -9 -f scripts/curriculum_dungeon.py 2>/dev/null
    sleep 4
    rm -rf checkpoints/run
    mkdir -p checkpoints/run
    group="dungeon-$(date +%Y%m%d-%H%M%S)"  # wandb group: bundles train + rollouts runs together
    echo "$group" > logs/group.txt
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid uv run python scripts/train_dungeon.py "$@" --wandb --data-dir checkpoints/run --save-path checkpoints/dungeon.pt </dev/null >logs/train.log 2>&1 &
    echo "train launched pid $! (group $group)"
    ;;
  follow)
    pkill -9 -f scripts/follow_along.py 2>/dev/null
    sleep 2
    rid=""  # resume the training run so POV videos share its wandb dashboard
    for _ in $(seq 1 30); do
      rid=$(grep -aoE 'runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1 | cut -d/ -f2)
      [ -n "$rid" ] && break
      sleep 2
    done
    group=$(cat logs/group.txt 2>/dev/null || echo "")  # same group as the training run -> bundled in wandb
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid uv run --extra train python scripts/follow_along.py --watch checkpoints/run --wandb --run-id "$rid" --interval 180 </dev/null >logs/follow.log 2>&1 &
    echo "follow launched pid $! (group $group, run $rid)"
    ;;
  status)
    echo "procs: train=$(pgrep -cf scripts/train_dungeon.py) follow=$(pgrep -cf scripts/follow_along.py)  load:$(uptime | grep -oE 'average.*')"
    echo "wandb:   $(grep -aoE 'https://wandb.ai/[^ ]+/runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1)"
    echo "config:  $(grep -a CONFIG logs/train.log 2>/dev/null | head -1)"
    echo "cleared: $(grep -aoE 'cleared +[0-9.]+' logs/train.log 2>/dev/null | tail -1)   steps: $(grep -aoE 'Steps +[0-9.]+ *[KM]?' logs/train.log 2>/dev/null | tail -1)"
    echo "follow:  $(grep -aoE 'rendered .*|skip .*' logs/follow.log 2>/dev/null | tail -1)"
    ;;
  *)
    echo "usage: box.sh {kill|train <args>|follow|status}" ;;
esac
