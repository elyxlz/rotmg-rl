#!/bin/bash
# Async throughput sweep across N (and a CPU-split for the first N to confirm partition helps).
set -u
RL=~/rotmg-rl
HIDDEN="${HIDDEN:-1024}"
echo "### async throughput sweep (hidden=$HIDDEN) ###"
for N in 24 48 64 96; do
  # give the trainer 4 cores, worlds the other 28 (worlds are the bottleneck)
  SRV_CPUS="0-27" TRAINER_CPUS="28-31" TORCH_THREADS=4 \
    bash "$RL/run-async-tput.sh" "$N" 36 "$HIDDEN" 2>&1 | grep -E "RESULT|gpu_samples|last SPS" 
  sleep 3
done
