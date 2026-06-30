#!/bin/bash
# Throwaway async smoke: boot the C# server-as-sim in SIM_ASYNC=1, run a short trainer,
# assert no segfault/hang. Isolated from live (:2050 / redis 6379): sim redis 6390 db5,
# server port 2060, shm /dev/shm/rotmg_sim_smoke. GPU 1.
set -u
N="${1:-24}"
STEPS="${2:-3000}"
HIDDEN="${3:-1024}"
VERIFY="${SIM_ASYNC_VERIFY:-0}"
SHM=/dev/shm/rotmg_sim_smoke
SRV=~/rotmg-sim-server
RL=~/rotmg-rl
LOG=/tmp/async_smoke_srv_${N}.log
rm -f "$SHM" "$LOG"

# --- boot the server (async, isolated) ---
export SIM_SHM=1 SIM_ASYNC=1 SIM_SHM_BARRIER=0
export SIM_SHM_PATH="$SHM"
export SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS="${SIM_SERVER_CPUS:-0-27}"
export SIM_ASYNC_VERIFY="$VERIFY"
# easy fixed proof config so episodes turn over (learning smoke); throughput probes override.
export SIM_AGENT_HP="${SIM_AGENT_HP:-5000}" SIM_AGENT_DEF="${SIM_AGENT_DEF:-40}"
export SIM_SPAWN_GEO_DIST="${SIM_SPAWN_GEO_DIST:-25}" SIM_BOSS_HP="${SIM_BOSS_HP:-1500}"
export SIM_APPROACH_SCALE="${SIM_APPROACH_SCALE:-0.02}" SIM_EP_TIMEOUT="${SIM_EP_TIMEOUT:-1500}"
setsid bash "$SRV/run-server-sim.sh" "$N" > "$LOG" 2>&1 < /dev/null &
SRV_PID=$!

# wait for the shm region + magic
for i in $(seq 1 120); do
  [ -f "$SHM" ] && grep -q "region '" "$LOG" && break
  sleep 0.5
done
echo "[smoke] server booted (log $LOG):"
grep -E "SIM-SHM|async" "$LOG" | head

# --- run the trainer (async) ---
cd "$RL" || exit 1
source .venv/bin/activate
source buildenv.sh
RC=0
SIM_SHM_PATH="$SHM" SIM_ASYNC=1 SIM_ASYNC_VERIFY="$VERIFY" CUDA_VISIBLE_DEVICES=1 \
  python -m rotmg_rl.trainer.train --agents "$N" --steps "$STEPS" --hidden "$HIDDEN" \
  --redis-port 6390 --redis-db 5 --out /tmp/async_smoke.pt 2>&1
RC=$?
echo "[smoke] trainer exit code: $RC"

# --- teardown ---
pkill -9 -f "WorldServer.dll" 2>/dev/null
sleep 1
rm -f "$SHM"
echo "[smoke] server tail:"; tail -5 "$LOG"
exit $RC
