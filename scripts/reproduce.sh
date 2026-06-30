#!/usr/bin/env bash
# reproduce.sh — the one-command path from a clean checkout to a clearing policy + a recorded clear.
#
#   provision -> build the C# server -> boot N server-as-sim worlds -> train the curriculum (d 0->1)
#   -> eval ladder (curriculum depth) -> deploy on the live :2050 server -> record the POV clear.
#
# Each stage echoes a banner and is independently runnable (see the per-stage commands it prints).
# Tunables (env-overridable): N worlds, STEPS per agent, GPU, the deploy mode. The training stages run
# on the GPU box (CUDA + the built _C); the deploy stages need the live betterSkillys server + a built
# Flash/nrelay client (see deploy/README.md). Nothing here touches GPU-0 of an already-running job
# unless you point GPU at it.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

N="${N:-32}"                 # parallel Snake Pit worlds == trainer --agents N == SIM_WORLDS
STEPS="${STEPS:-200000}"     # steps PER AGENT for the long run (global = STEPS*N)
HIDDEN="${HIDDEN:-2048}"
GPU="${GPU:-1}"              # default GPU 1; GPU 0 is reserved for the long-running sweep on the box
SHM="${SHM:-/dev/shm/rotmg_reproduce}"
SRVLOG="$REPO_ROOT/logs/reproduce_server.log"
CKPT_DIR="$REPO_ROOT/checkpoints/reproduce"
DEPLOY_MODE="${DEPLOY_MODE:-nrelay}"   # nrelay = headless working path; proxy = POV-recording path
mkdir -p "$REPO_ROOT/logs" "$CKPT_DIR"

banner() { echo; echo "======================================================================"; echo "  $*"; echo "======================================================================"; }

banner "1/6  Provision the venv + build the server_env C-shim into PufferLib _C"
scripts/setup.sh

banner "2/6  Fetch the pinned betterSkillys upstream, apply the Sim overlay, build the C# WorldServer"
sim-server/fetch.sh

banner "3/6  Boot $N server-as-sim Snake Pit worlds (shm $SHM, futex barrier; isolated :2060 / redis 6390)"
rm -f "$SHM"
export SIM_SHM=1 SIM_SHM_BARRIER=1 SIM_SHM_PATH="$SHM"
export SIM_STEP_REDIS_PORT=6390 SIM_STEP_REDIS_DB=5
setsid bash sim-server/run-server-sim.sh "$N" > "$SRVLOG" 2>&1 < /dev/null &
for _ in $(seq 1 120); do [ -f "$SHM" ] && grep -q "region" "$SRVLOG" && break; sleep 0.5; done
sleep 3
echo "server up (log: $SRVLOG)"

# shellcheck disable=SC1091
source .venv/bin/activate
source buildenv.sh

banner "4/6  Train the d-ramp curriculum 0->1 (longrun) -> a clearing policy at the real difficulty"
SIM_SHM_PATH="$SHM" SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 \
  python -m rotmg_rl.trainer.longrun \
    --agents "$N" --steps "$STEPS" --hidden "$HIDDEN" \
    --server-log "$SRVLOG" --redis-port 6390 --redis-db 5 --out-dir "$CKPT_DIR"
# the longrun writes best.pt (best curriculum depth), latest.pt, and final.pt; prefer best.
CKPT="$CKPT_DIR/best.pt"
[ -f "$CKPT" ] || CKPT="$CKPT_DIR/final.pt"
[ -f "$CKPT" ] || CKPT="$CKPT_DIR/latest.pt"

banner "5/6  Eval ladder -> curriculum depth (per-rung clear rate, the Protein objective)"
SIM_SHM_PATH="$SHM" SIM_SHM_BARRIER=1 CUDA_VISIBLE_DEVICES="$GPU" \
  python -m rotmg_rl.trainer.eval --checkpoint "$CKPT" --agents "$N" --server-log "$SRVLOG"

# Tear down the throwaway sim server before the live deploy (frees the box; never touched :2050).
pkill -9 -f WorldServer.dll 2>/dev/null || true; sleep 1; rm -f "$SHM"

banner "6/6  Deploy the trained policy on the LIVE :2050 server, then record the clear"
# Two ways to drive the live game (see deploy/README.md + deploy/nrelay/README.md):
#   nrelay (default, the working path): the headless TS client + policy-bridge spawns
#     rotmg_rl.deploy.server with the checkpoint, navigates into a Snake Pit, and fights.
#   proxy (the POV-recording path): a no-root packet proxy + input injection plays the REAL
#     Flash client so the recorded video is an authentic first-person POV.
case "$DEPLOY_MODE" in
  nrelay)
    echo "deploy via deploy/nrelay (checkpoint: $CKPT)"
    ( cd deploy/nrelay && npm install && npx tsc -p . && CHECKPOINT="$CKPT" node start-bot.js )
    ;;
  proxy)
    echo "deploy via deploy/proxy (POV recording). Boot the proxy + input-injection bridge:"
    python deploy/proxy/obs_proxy.py &
    echo "then run the policy against the proxied stream (see deploy/proxy/ + deploy/README.md)."
    ;;
  *)
    echo "unknown DEPLOY_MODE=$DEPLOY_MODE (use 'nrelay' or 'proxy')" >&2; exit 1
    ;;
esac

banner "Record the POV clear (real Flash spectator client, full entrance->boss-death window)"
deploy/record.sh

echo
echo "DONE. checkpoint: $CKPT  |  recorded clear: see videos/ (deploy/record.sh output)."
