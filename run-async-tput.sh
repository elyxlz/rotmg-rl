#!/bin/bash
# Async THROUGHPUT probe: boot server-as-sim (SIM_ASYNC=1), run a fixed-duration trainer,
# report steady-state SPS + GPU util. CPU-partitioned: worlds get most cores, the GPU-bound
# trainer a few (PyTorch is GPU-bound; handing it 28 cores starves the worlds, which caps SPS).
# Isolated: sim redis 6390 db5, server port 2060, shm /dev/shm/rotmg_sim_tput. GPU 1.
set -u
N="${1:-24}"
SECONDS_RUN="${2:-40}"
HIDDEN="${3:-1024}"
SRV_CPUS="${SRV_CPUS:-0-27}"
TRAINER_CPUS="${TRAINER_CPUS:-28-31}"
TORCH_THREADS="${TORCH_THREADS:-4}"
SHM=/dev/shm/rotmg_sim_tput
SRV=~/rotmg-sim-server
RL=~/rotmg-rl
LOG=/tmp/async_tput_srv_${N}.log
TLOG=/tmp/async_tput_trn_${N}.log
rm -f "$SHM" "$LOG" "$TLOG"

export SIM_SHM=1 SIM_ASYNC=1 SIM_SHM_BARRIER=0
export SIM_SHM_PATH="$SHM"
export SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
export SIM_SERVER_NICE=0 SIM_SERVER_CPUS="$SRV_CPUS"
# throughput config: invuln + huge boss so episodes never end (pure per-tick throughput, no reset noise)
export SIM_AGENT_HP=700 SIM_AGENT_DEF=0 SIM_SPAWN_GEO_DIST=0
export SIM_BOSS_HP=99999999 SIM_RL_INVULN=1
export SIM_APPROACH_SCALE=0.0 SIM_STEP_PENALTY=0.0 SIM_EP_TIMEOUT=0
setsid bash "$SRV/run-server-sim.sh" "$N" > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 120); do
  [ -f "$SHM" ] && grep -q "region '" "$LOG" && break
  sleep 0.5
done
sleep 2  # let worlds spawn + bind settle

cd "$RL" || exit 1
source .venv/bin/activate
source buildenv.sh
export OMP_NUM_THREADS="$TORCH_THREADS"
# run a big step budget, kill after SECONDS_RUN; SPS is read from the steady-state tail
SIM_SHM_PATH="$SHM" SIM_ASYNC=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS="$TORCH_THREADS" \
  taskset -c "$TRAINER_CPUS" python server_train.py --agents "$N" --steps 100000000 --hidden "$HIDDEN" \
  --redis-port 6390 --redis-db 5 --out /tmp/async_tput.pt > "$TLOG" 2>&1 &
TRN_PID=$!

# sample GPU util across the run
sleep 12  # warmup
gpu_samples=""
for s in $(seq 1 $((SECONDS_RUN/3))); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
  gpu_samples="$gpu_samples $u"
  sleep 3
done
kill -9 "$TRN_PID" 2>/dev/null
pkill -9 -f WorldServer.dll 2>/dev/null
sleep 1
rm -f "$SHM"

# steady-state SPS: median of the last several samples
sps=$(grep -oE 'SPS=[0-9]+' "$TLOG" | tail -15 | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{if(NR>0)print a[int((NR+1)/2)]; else print "NA"}')
gpu_med=$(echo $gpu_samples | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -n | awk '{a[NR]=$1} END{if(NR>0)print a[int((NR+1)/2)]; else print "NA"}')
echo "RESULT N=$N hidden=$HIDDEN  median_SPS=$sps  median_GPU_util=${gpu_med}%  srv_cpus=$SRV_CPUS trainer_cpus=$TRAINER_CPUS torch_threads=$TORCH_THREADS"
echo "  gpu_samples:$gpu_samples"
echo "  last SPS lines:"; grep -oE 'step=[0-9]+ updates=[0-9]+ SPS=[0-9]+' "$TLOG" | tail -4
