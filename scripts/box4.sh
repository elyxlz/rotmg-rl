#!/usr/bin/env bash
# Manage PufferLib 4.0 dungeon training on the box, parallel to box.sh (3.0). Runs on GPU 1 / .venv4
# so it never touches a live 3.0 run on GPU 0 / .venv. Use --slowly runs (our CNN) for renderable
# videos; the native path trains faster but isn't renderable (see docs/pufferlib4-migration.md).
#
# Usage: ./scripts/box4.sh {kill | train <train_dungeon4 args...> | follow | wait | status}
#   ./scripts/box4.sh train --slowly --boss-hp 300 --no-snakes ... --total-timesteps 50000000
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1   # GPU 1; GPU 0 is reserved for any live 3.0 run
mkdir -p logs4 checkpoints4
PY=.venv4/bin/python

cmd="${1:-status}"
shift || true
case "$cmd" in
  kill)
    pkill -f scripts/train_dungeon4.py 2>/dev/null
    pkill -f scripts/follow_along4.py 2>/dev/null
    pkill -f 'puffer train dungeon' 2>/dev/null
    sleep 8
    pkill -9 -f 'scripts/(train_dungeon4|follow_along4)|puffer train dungeon' 2>/dev/null
    sleep 2
    echo "remaining: $(pgrep -cf 'scripts/(train_dungeon4|follow_along4)|puffer train dungeon')  load:$(uptime | grep -oE 'average.*')"
    ;;
  wait)  # block until the 4.0 run finishes/dies, then print final metrics (run in background -> notifies)
    misses=0
    while [ "$misses" -lt 2 ]; do
      if pgrep -f 'puffer train dungeon' >/dev/null; then misses=0; else misses=$((misses + 1)); fi
      sleep 60
    done
    rid=$(grep -aoE 'runs/[a-z0-9]+' logs4/train.log 2>/dev/null | tail -1 | cut -d/ -f2)
    echo "=== 4.0 TRAINING RUN FINISHED (wandb run $rid) ==="
    grep -aE 'boss_hp_frac|cleared |perf |Steps |SPS ' logs4/train.log 2>/dev/null | tail -8
    ;;
  train)  # launch ONE clean run; kills any existing 4.0 train first. Defaults to --slowly (our CNN,
          # renderable, the recommended path); pass --native for the flat native encoder instead.
    pkill -9 -f scripts/train_dungeon4.py 2>/dev/null
    pkill -9 -f 'puffer train dungeon' 2>/dev/null
    sleep 4
    rm -rf checkpoints4/dungeon
    group="dungeon4-$(date +%Y%m%d-%H%M%S)"   # wandb group: bundles train + rollouts4 runs
    echo "$group" > logs4/group.txt
    set -- "$@"  # default to the CNN (--slowly) unless --native was explicitly requested
    case " $* " in
      *" --native "*) args=$(printf '%s ' "$@" | sed 's/ --native / /'); set -- $args; rm -f logs4/slowly.flag ;;
      *" --slowly "*) touch logs4/slowly.flag ;;
      *) set -- "$@" --slowly; touch logs4/slowly.flag ;;
    esac
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid "$PY" scripts/train_dungeon4.py "$@" --wandb --checkpoint-dir checkpoints4 </dev/null >logs4/train.log 2>&1 &
    echo "4.0 train launched pid $! (group $group, GPU $CUDA_VISIBLE_DEVICES, $* )"
    ;;
  follow)  # render the latest checkpoint to POV mp4 + wandb (only meaningful for --slowly runs)
    if [ ! -f logs4/slowly.flag ]; then
      echo "refusing: the live 4.0 run is native (flat encoder) -> checkpoints aren't renderable. Train with --slowly."
      exit 1
    fi
    pkill -9 -f scripts/follow_along4.py 2>/dev/null
    sleep 2
    rid=""
    for _ in $(seq 1 30); do
      rid=$(grep -aoE 'runs/[a-z0-9]+' logs4/train.log 2>/dev/null | tail -1 | cut -d/ -f2)
      [ -n "$rid" ] && break
      sleep 2
    done
    group=$(cat logs4/group.txt 2>/dev/null || echo "")
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid "$PY" scripts/follow_along4.py --watch checkpoints4/dungeon --wandb --run-id "$rid" --interval 120 </dev/null >logs4/follow.log 2>&1 &
    echo "4.0 follow launched pid $! (group $group, run $rid)"
    ;;
  status)
    echo "procs: train=$(pgrep -cf scripts/train_dungeon4.py) follow=$(pgrep -cf scripts/follow_along4.py)  load:$(uptime | grep -oE 'average.*')"
    echo "wandb:   $(grep -aoE 'https://wandb.ai/[^ ]+/runs/[a-z0-9]+' logs4/train.log 2>/dev/null | tail -1)"
    echo "SPS:     $(grep -aoE 'SPS +[0-9.]+ *[KM]?' logs4/train.log 2>/dev/null | tail -1)   steps: $(grep -aoE 'Steps +[0-9.]+ *[KM]?' logs4/train.log 2>/dev/null | tail -1)"
    echo "cleared: $(grep -aoE 'cleared +[0-9.]+' logs4/train.log 2>/dev/null | tail -1)   boss_hp: $(grep -aoE 'boss_hp_frac +[0-9.]+' logs4/train.log 2>/dev/null | tail -1)"
    echo "follow:  $(grep -aoE 'rendered .*|skip .*' logs4/follow.log 2>/dev/null | tail -1)"
    ;;
  metrics)  # latest per-episode + per-step metrics from the dashboard log (score = per-episode mean)
    echo "wandb: $(grep -aoE 'https://wandb.ai/[^ ]+/runs/[a-z0-9]+' logs4/train.log 2>/dev/null | tail -1)"
    for k in score episodes cleared boss_hp_frac perf reward entropy SPS Steps; do
      printf '%-13s %s\n' "$k:" "$(grep -aoE "$k +[-0-9.]+ *[KM]?" logs4/train.log 2>/dev/null | tail -1)"
    done
    ;;
  *)
    echo "usage: box4.sh {kill|train [--native] <args>|follow|wait|status|metrics}" ;;
esac
