#!/bin/bash
# Boot the async server-as-sim + run verify_obs_async.py (bit-identical, torn-free obs proof).
set -u
N="${1:-24}"
SHM=/dev/shm/rotmg_sim_vobs
SRV=~/rotmg-sim-server; RL=~/rotmg-rl
LOG=/tmp/vobs_srv_${N}.log
rm -f "$SHM" "$LOG"
export SIM_SHM=1 SIM_ASYNC=1 SIM_SHM_BARRIER=0
export SIM_SHM_PATH="$SHM" SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS=0-27
export SIM_AGENT_HP=700 SIM_AGENT_DEF=0 SIM_SPAWN_GEO_DIST=0
export SIM_BOSS_HP=99999999 SIM_RL_INVULN=1 SIM_APPROACH_SCALE=0.0 SIM_STEP_PENALTY=0.0 SIM_EP_TIMEOUT=0
setsid bash "$SRV/run-server-sim.sh" "$N" > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do [ -f "$SHM" ] && grep -q "region '" "$LOG" && break; sleep 0.5; done
sleep 2
cd "$RL"; source .venv/bin/activate; source buildenv.sh
SIM_SHM_PATH="$SHM" SIM_ASYNC=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
  taskset -c 28-31 python verify_obs_async.py --agents "$N" --warmup 150 2>&1 | grep -vE "^\s*$"
RC=${PIPESTATUS[0]:-$?}
pkill -9 -f WorldServer.dll 2>/dev/null; sleep 1; rm -f "$SHM"
echo "verify_obs_async exit: $RC"
exit $RC
