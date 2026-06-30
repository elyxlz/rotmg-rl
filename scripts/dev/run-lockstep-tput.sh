#!/bin/bash
# Lockstep baseline probe under the SAME end-to-end harness (server + trainer over shm),
# SIM_ASYNC unset -> pure-shm futex barrier. Same isolation + config as the async probe so
# the SPS/GPU comparison is apples-to-apples.
set -u
N="${1:-24}"
SECONDS_RUN="${2:-40}"
HIDDEN="${3:-1024}"
SRV_CPUS="${SRV_CPUS:-0-27}"
TRAINER_CPUS="${TRAINER_CPUS:-28-31}"
TORCH_THREADS="${TORCH_THREADS:-4}"
SHM=/dev/shm/rotmg_sim_lock
SRV=~/rotmg-sim-server; RL=~/rotmg-rl
LOG=/tmp/lock_tput_srv_${N}.log; TLOG=/tmp/lock_tput_trn_${N}.log
rm -f "$SHM" "$LOG" "$TLOG"
export SIM_SHM=1 SIM_SHM_BARRIER=1
unset SIM_ASYNC
export SIM_SHM_PATH="$SHM" SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS="$SRV_CPUS"
export SIM_AGENT_HP=700 SIM_AGENT_DEF=0 SIM_SPAWN_GEO_DIST=0
export SIM_BOSS_HP=99999999 SIM_RL_INVULN=1 SIM_APPROACH_SCALE=0.0 SIM_STEP_PENALTY=0.0 SIM_EP_TIMEOUT=0
setsid bash "$SRV/run-server-sim.sh" "$N" > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do [ -f "$SHM" ] && grep -q "region '" "$LOG" && break; sleep 0.5; done
sleep 2
cd "$RL"; source .venv/bin/activate; source buildenv.sh
SIM_SHM_PATH="$SHM" SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS="$TORCH_THREADS" \
  taskset -c "$TRAINER_CPUS" python -m rotmg_rl.trainer.train --agents "$N" --steps 100000000 --hidden "$HIDDEN" \
  --redis-port 6390 --redis-db 5 --out /tmp/lock_tput.pt > "$TLOG" 2>&1 &
TRN=$!
sleep 12
gs=""
for s in $(seq 1 $((SECONDS_RUN/3))); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1); gs="$gs $u"; sleep 3; done
kill -9 "$TRN" 2>/dev/null; pkill -9 -f WorldServer.dll 2>/dev/null; sleep 1; rm -f "$SHM"
sps=$(grep -oE 'SPS=[0-9]+' "$TLOG" | tail -15 | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
gm=$(echo $gs | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
echo "RESULT(lockstep) N=$N hidden=$HIDDEN median_SPS=$sps median_GPU=${gm}%"
