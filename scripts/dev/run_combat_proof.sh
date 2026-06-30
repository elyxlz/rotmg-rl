#!/bin/bash
# Combat-faithful learning re-proof: start the server-as-sim with REAL projectile
# collision (no proximity shortcut) at an easy-but-honest config where the boss can
# ONLY be killed by the agent landing real shots, then run a real pufferl PPO loop
# and tail the server's ground-truth EPISODE DONE lines into a learning curve.
set -u
N=32
STEPS=${STEPS:-800000}            # per-agent env steps
SRV_LOG=/tmp/combat_proof_server.log
CURVE=/home/audiogen/rotmg-rl/logs/combat_proof_curve.csv

# --- honest easy config: real collision is the ONLY damage path ---
export SIM_SPAWN_GEO_DIST=18      # short navigate-in (known clear path)
export SIM_BOSS_HP=800            # killable with sustained real shots
export SIM_AGENT_HP=700           # moderate: undodged bullets CAN kill, but a clear is finishable
export SIM_AGENT_DEF=30           # flat reduction; sustained fire is still lethal
export SIM_APPROACH_SCALE=0.02
export SIM_STEP_PENALTY=0.0005
export SIM_EP_TIMEOUT=1500
export SIM_SHM_BARRIER=1

echo "[proof] starting server N=$N (boss_hp=$SIM_BOSS_HP agent_hp=$SIM_AGENT_HP def=$SIM_AGENT_DEF spawn_geo=$SIM_SPAWN_GEO_DIST)"
tmux kill-session -t combatsrv 2>/dev/null
cd /home/audiogen/rotmg-sim-server
tmux new-session -d -s combatsrv \
  "SIM_SPAWN_GEO_DIST=$SIM_SPAWN_GEO_DIST SIM_BOSS_HP=$SIM_BOSS_HP SIM_AGENT_HP=$SIM_AGENT_HP \
   SIM_AGENT_DEF=$SIM_AGENT_DEF SIM_APPROACH_SCALE=$SIM_APPROACH_SCALE SIM_STEP_PENALTY=$SIM_STEP_PENALTY \
   SIM_EP_TIMEOUT=$SIM_EP_TIMEOUT SIM_SHM_BARRIER=$SIM_SHM_BARRIER ./run-server-sim.sh $N 2>&1 | tee $SRV_LOG"

# wait for the shm region + both worlds to be up (barrier waiting)
for i in $(seq 1 60); do
  if grep -q 'pure-shm futex lockstep ON' "$SRV_LOG" 2>/dev/null; then break; fi
  sleep 1
done
echo "[proof] server ready:"; grep -E 'region|barrier|created Snake Pit|difficulty' "$SRV_LOG" | tail -5

# run the proof harness (trainer + server-log tailer -> learning curve)
cd /home/audiogen/rotmg-rl
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy
CUDA_VISIBLE_DEVICES=1 SIM_SHM_BARRIER=1 .venv/bin/python -m rotmg_rl.trainer.proof \
  --agents $N --steps $STEPS --server-log $SRV_LOG --out $CURVE 2>&1 | tee /tmp/combat_proof_train.log

echo "[proof] DONE; curve -> $CURVE"
tmux kill-session -t combatsrv 2>/dev/null
