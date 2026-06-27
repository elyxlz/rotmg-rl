#!/usr/bin/env bash
# Manage a single configurable dungeon training run on the box. Use --slowly runs (our CNN) for
# renderable POV videos; the native path trains faster but isn't renderable.
#
# Usage: ./scripts/box.sh {kill | train <puffer_cli args...> | follow | wait | status | metrics}
#   ./scripts/box.sh train --slowly --boss-hp 300 --no-snakes ... --total-timesteps 50000000
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # GPU 1 by default; leave GPU 0 free
mkdir -p logs checkpoints
PY=.venv/bin/python

cmd="${1:-status}"
shift || true
case "$cmd" in
  kill)
    pkill -f 'rotmg_rl.puffer_cli' 2>/dev/null
    pkill -f 'rotmg_rl.video' 2>/dev/null
    pkill -f 'puffer train dungeon' 2>/dev/null
    sleep 8
    pkill -9 -f 'rotmg_rl.(puffer_cli|video)|puffer train dungeon' 2>/dev/null
    sleep 2
    echo "remaining: $(pgrep -cf 'rotmg_rl.(puffer_cli|video)|puffer train dungeon')  load:$(uptime | grep -oE 'average.*')"
    ;;
  wait)  # block until the run finishes/dies, then print final metrics (run in background -> notifies)
    misses=0
    while [ "$misses" -lt 2 ]; do
      if pgrep -f 'puffer train dungeon' >/dev/null; then misses=0; else misses=$((misses + 1)); fi
      sleep 60
    done
    rid=$(grep -aoE 'runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1 | cut -d/ -f2)
    echo "=== TRAINING RUN FINISHED (wandb run $rid) ==="
    grep -aE 'boss_hp_frac|cleared |perf |Steps |SPS ' logs/train.log 2>/dev/null | tail -8
    ;;
  train)  # launch ONE clean run; kills any existing train first. Defaults to --slowly (our CNN,
          # renderable, the recommended path); pass --native for the flat native encoder instead.
    pkill -9 -f 'rotmg_rl.puffer_cli' 2>/dev/null
    pkill -9 -f 'puffer train dungeon' 2>/dev/null
    sleep 4
    rm -rf checkpoints/dungeon
    group="dungeon-$(date +%Y%m%d-%H%M%S)"   # wandb group: bundles train + rollouts runs
    echo "$group" > logs/group.txt
    set -- "$@"  # default to the CNN (--slowly) unless --native was explicitly requested
    case " $* " in
      *" --native "*) args=$(printf '%s ' "$@" | sed 's/ --native / /'); set -- $args; rm -f logs/slowly.flag ;;
      *" --slowly "*) touch logs/slowly.flag ;;
      *) set -- "$@" --slowly; touch logs/slowly.flag ;;
    esac
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid "$PY" -m rotmg_rl.puffer_cli "$@" --wandb --checkpoint-dir checkpoints </dev/null >logs/train.log 2>&1 &
    echo "train launched pid $! (group $group, GPU $CUDA_VISIBLE_DEVICES, $* )"
    ;;
  follow)  # render the latest checkpoint to POV mp4 + wandb (only meaningful for --slowly runs)
    if [ ! -f logs/slowly.flag ]; then
      echo "refusing: the live run is native (flat encoder) -> checkpoints aren't renderable. Train with --slowly."
      exit 1
    fi
    pkill -9 -f 'rotmg_rl.video' 2>/dev/null
    sleep 2
    rid=""
    for _ in $(seq 1 30); do
      rid=$(grep -aoE 'runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1 | cut -d/ -f2)
      [ -n "$rid" ] && break
      sleep 2
    done
    group=$(cat logs/group.txt 2>/dev/null || echo "")
    WANDB_RUN_GROUP="$group" PYTHONUNBUFFERED=1 setsid "$PY" -m rotmg_rl.video --watch checkpoints/dungeon --wandb --run-id "$rid" --interval 120 </dev/null >logs/follow.log 2>&1 &
    echo "follow launched pid $! (group $group, run $rid)"
    ;;
  status)
    echo "procs: train=$(pgrep -cf 'rotmg_rl.puffer_cli') follow=$(pgrep -cf 'rotmg_rl.video')  load:$(uptime | grep -oE 'average.*')"
    echo "wandb:   $(grep -aoE 'https://wandb.ai/[^ ]+/runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1)"
    echo "SPS:     $(grep -aoE 'SPS +[0-9.]+ *[KM]?' logs/train.log 2>/dev/null | tail -1)   steps: $(grep -aoE 'Steps +[0-9.]+ *[KM]?' logs/train.log 2>/dev/null | tail -1)"
    echo "cleared: $(grep -aoE 'cleared +[0-9.]+' logs/train.log 2>/dev/null | tail -1)   boss_hp: $(grep -aoE 'boss_hp_frac +[0-9.]+' logs/train.log 2>/dev/null | tail -1)"
    echo "follow:  $(grep -aoE 'rendered .*|skip .*' logs/follow.log 2>/dev/null | tail -1)"
    ;;
  metrics)  # latest per-episode + per-step metrics from the dashboard log (score = per-episode mean)
    echo "wandb: $(grep -aoE 'https://wandb.ai/[^ ]+/runs/[a-z0-9]+' logs/train.log 2>/dev/null | tail -1)"
    for k in score episodes cleared boss_hp_frac perf reward entropy SPS Steps; do
      printf '%-13s %s\n' "$k:" "$(grep -aoE "$k +[-0-9.]+ *[KM]?" logs/train.log 2>/dev/null | tail -1)"
    done
    ;;
  *)
    echo "usage: box.sh {kill|train [--native] <args>|follow|wait|status|metrics}" ;;
esac
