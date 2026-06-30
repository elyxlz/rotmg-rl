#!/bin/bash
# ONE long server-as-sim curriculum run -> a clearing policy at d=1. Boots the throwaway server in
# server-as-sim mode (SIM_ASYNC=1, N=48, CPU-partitioned worlds 0-27 / trainer 28-31, GPU 0), then
# runs rotmg_rl.trainer.longrun (curriculum d 0->1, warm-start hparams, periodic eval ladder + wandb). The
# two WIRED reward knobs go in as server env vars. Isolated: sim redis 6390 db5, shm rotmg_longrun.
# NEVER touches live :2050 / redis 6379.
set -u
N="${N:-48}"
STEPS="${STEPS:-1450000}"      # steps PER AGENT; global = STEPS*N
HIDDEN="${HIDDEN:-2048}"
RAMP="${RAMP:-0.55}"
EVAL_EVERY="${EVAL_EVERY:-15000000}"
GPU="${GPU:-0}"
APPROACH="${APPROACH:-0.07}"
STEP_PEN="${STEP_PEN:-0.000586}"
SHM=/dev/shm/rotmg_longrun
SRV=~/rotmg-sim-server; RL=~/rotmg-rl
SRVLOG="$RL/logs/longrun_server.log"; TRNLOG="$RL/logs/longrun_trainer.log"
mkdir -p "$RL/logs" "$RL/checkpoints/longrun"
rm -f "$SHM"

export SIM_SHM=1 SIM_ASYNC=1 SIM_SHM_BARRIER=0
export SIM_SHM_PATH="$SHM"
export SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS=0-27
export SIM_RL_INVULN=0
export SIM_APPROACH_SCALE="$APPROACH" SIM_STEP_PENALTY="$STEP_PEN" SIM_EP_TIMEOUT=1500
# fallback (pre-first-config) easy anchor; the curriculum drives the live d over shm.
export SIM_AGENT_HP=5000 SIM_AGENT_DEF=40 SIM_SPAWN_GEO_DIST=12 SIM_BOSS_HP=1500

setsid bash "$SRV/run-server-sim.sh" "$N" > "$SRVLOG" 2>&1 < /dev/null &
SRV_SETSID=$!
for i in $(seq 1 120); do [ -f "$SHM" ] && grep -q "region " "$SRVLOG" && break; sleep 0.5; done
sleep 3

cd "$RL"; source .venv/bin/activate; source buildenv.sh
SIM_SHM_PATH="$SHM" SIM_ASYNC=1 SIM_SHM_BARRIER=0 \
  CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 \
  taskset -c 28-31 python -m rotmg_rl.trainer.longrun \
    --agents "$N" --steps "$STEPS" --hidden "$HIDDEN" --ramp-frac "$RAMP" \
    --eval-every "$EVAL_EVERY" --server-log "$SRVLOG" \
    --redis-port 6390 --redis-db 5 --out-dir checkpoints/longrun \
    --wandb-project rotmg-server-as-sim --wandb-name "longrun-h${HIDDEN}-$(date +%m%d_%H%M)" \
  2>&1 | tee "$TRNLOG"

# teardown
pkill -9 -f WorldServer.dll 2>/dev/null; sleep 1; rm -f "$SHM"
echo "=== longrun script exit ==="
