#!/bin/bash
# Short learning run on the easy-clear config. MODE=async|lockstep. Prints the done_rate
# (clear-rate proxy) curve so async vs lockstep learning can be compared. N=24, ~60k steps/agent.
set -u
MODE="${1:-async}"
N="${2:-24}"
STEPS="${3:-60000}"
SHM=/dev/shm/rotmg_sim_learn_${MODE}
SRV=~/rotmg-sim-server; RL=~/rotmg-rl
LOG=/tmp/learn_srv_${MODE}.log; TLOG=/tmp/learn_trn_${MODE}.log
rm -f "$SHM" "$LOG" "$TLOG"
export SIM_SHM=1 SIM_SHM_PATH="$SHM" SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS=0-27
# easy fixed proof config (a policy clears in tens-of-K steps)
export SIM_AGENT_HP=5000 SIM_AGENT_DEF=40 SIM_SPAWN_GEO_DIST=25 SIM_BOSS_HP=1500
export SIM_APPROACH_SCALE=0.02 SIM_STEP_PENALTY=0.0005 SIM_EP_TIMEOUT=1500 SIM_RL_INVULN=0
if [ "$MODE" = "async" ]; then export SIM_ASYNC=1 SIM_SHM_BARRIER=0; else unset SIM_ASYNC; export SIM_SHM_BARRIER=1; fi
setsid bash "$SRV/run-server-sim.sh" "$N" > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do [ -f "$SHM" ] && grep -q "region '" "$LOG" && break; sleep 0.5; done
sleep 2
cd "$RL"; source .venv/bin/activate; source buildenv.sh
ASYNC_ENV=""; [ "$MODE" = "async" ] && ASYNC_ENV="SIM_ASYNC=1"
env $ASYNC_ENV SIM_SHM_PATH="$SHM" SIM_SHM_BARRIER=$([ "$MODE" = lockstep ] && echo 1 || echo 0) \
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 SIM_RL_SEED=0 \
  taskset -c 28-31 python -m rotmg_rl.trainer.train --agents "$N" --steps "$STEPS" --hidden 1024 \
  --redis-port 6390 --redis-db 5 --out /tmp/learn_${MODE}.pt > "$TLOG" 2>&1
pkill -9 -f WorldServer.dll 2>/dev/null; sleep 1; rm -f "$SHM"
echo "=== $MODE done_rate curve (every ~10k steps) ==="
grep -oE "step=[0-9]+ updates=[0-9]+ SPS=[0-9]+ reward=[-0-9.]+ done_rate=[0-9.]+" "$TLOG" | awk 'NR%6==1{print}'
echo "=== $MODE final ==="; grep -oE "done_rate=[0-9.]+" "$TLOG" | tail -3
