#!/bin/bash
# Server-as-sim launcher: boots the throwaway WorldServer in the COMBINED mode the
# PufferLib server_env C-shim drives — N Snake Pit worlds, each an in-process RL loop
# whose actions/obs/reward/done cross SHARED MEMORY, gated by the REDIS LOCKSTEP gate.
#
#   ./run-server-sim.sh [N]      # N = agent count (default 4); == trainer --agents N
#
# Sets:
#   SIM_SHM=1          -> the in-proc loop is shm-driven (the C-shim is the policy)
#   SIM_STEP_DRIVEN=1  -> the redis lockstep gate advances all N worlds per tick command
#   SIM_WORLDS=N       -> N Snake Pit worlds == N shm agent slots
#   SIM_INVULN=0       -> agents can die (real episodes); flip to 1 to never die
#
# Isolation (NEVER touches live :2050 / redis 6379): port 2060, sim redis on 6390 db5.
#
# Runs the WorldServer that sim-server/fetch.sh builds under ./upstream/. Paths are relative
# to this script so the package is self-contained (no ~/rotmg-sim-server checkout needed).
set -u
N="${1:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/upstream/WorldServer/bin/Release/net8.0"
# dotnet on PATH wins; else the per-user SDK fetch.sh also assumes at ~/.dotnet.
export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
DOTNET="$(command -v dotnet || echo "$DOTNET_ROOT/dotnet")"
export IS_DOCKER=1
if [ ! -f "$BIN/WorldServer.dll" ]; then
  echo "ERROR: $BIN/WorldServer.dll missing -- run sim-server/fetch.sh first." >&2
  exit 1
fi
# The WorldServer reads its config from the working dir; drop the sim config in beside the dll.
cp "$HERE/wServer.sim.json" "$BIN/wServer.sim.json"

export SIM_SHM=1
# Sync primitive: SIM_SHM_BARRIER=1 == pure-shm futex barrier (no redis on the hot
# path); default. Set SIM_SHM_BARRIER=0 to fall back to the redis LPUSH/BLPOP gate.
export SIM_SHM_BARRIER="${SIM_SHM_BARRIER:-1}"
# Redis gate is still started as the fallback path (cheap; idle unless the barrier is off).
export SIM_STEP_DRIVEN=1
export SIM_WORLDS="$N"
export SIM_FIXED_DT_MS="${SIM_FIXED_DT_MS:-100}"
export SIM_BOSS_HP="${SIM_BOSS_HP:-7500}"
export SIM_PROBE_DPS_TICK="${SIM_PROBE_DPS_TICK:-200}"
export SIM_RL_INVULN="${SIM_RL_INVULN:-0}"

# ---- RL difficulty knobs (the curriculum drives these later; here they are env
# overridable). The real dungeon enemies + AI + boss mechanics stay UNMODIFIED;
# only the agent spawn/stats + boss HP are controllable training aids. d=1 (the
# real conditions) is SIM_SPAWN_GEO_DIST=-1, SIM_AGENT_DEF=0, full SIM_BOSS_HP.
#
#   EASY FIXED PROOF CONFIG (a policy clears within ~50-300K steps):
#     SIM_SPAWN_GEO_DIST=25 SIM_AGENT_HP=5000 SIM_AGENT_DEF=40 \
#     SIM_BOSS_HP=1500 SIM_APPROACH_SCALE=0.02 ./run-server-sim.sh 32
export SIM_AGENT_HP="${SIM_AGENT_HP:-700}"             # agent max HP (survivability knob)
export SIM_AGENT_DEF="${SIM_AGENT_DEF:-0}"             # flat per-hit damage reduction
export SIM_SPAWN_GEO_DIST="${SIM_SPAWN_GEO_DIST:--1}"  # geodesic tiles from boss; -1 == real entrance
export SIM_APPROACH_SCALE="${SIM_APPROACH_SCALE:-0.02}" # geodesic navigate-in reward scale
export SIM_STEP_PENALTY="${SIM_STEP_PENALTY:-0.0005}"  # per-tick time penalty
export SIM_EP_TIMEOUT="${SIM_EP_TIMEOUT:-1500}"        # hard episode-step cap (reason=timeout)

# ---- Deploy loadout (the live no-cheat test char): a maxed Wizard with a T7 Staff of
# Destruction (0xa9e, the continuous-fire weapon, 2-shot pattern) + a T7 Burning Retribution
# Spell (0x2055, the MP-limited BulletNova burst), NO armor, NO ring. The agent's DPS,
# bullet geometry, hitboxes + survivability match what it has on the live game. The staff/
# spell stats default to the real item XML in SimMode.cs; only the MP pool is set here (it is
# not a curriculum knob -- HP/DEF/boss/spawn ramp via the d-flow config, MP is the fixed
# real-Wizard pool). d=1 == HP 710 / DEF 0 / MP 425 / boss 7500 / entrance.
export SIM_AGENT_MP="${SIM_AGENT_MP:-425}"             # maxed Wizard 385 + spell-slot +40 (no ring)

export SIM_SHM_PATH="${SIM_SHM_PATH:-/dev/shm/rotmg_sim_shm}"
export SIM_STEP_REDIS_PORT="${SIM_STEP_REDIS_PORT:-6390}"
export SIM_STEP_REDIS_DB="${SIM_STEP_REDIS_DB:-5}"
export SIM_LOG_PATH="${SIM_LOG_PATH:-$BIN/logs/server_sim_timeline.csv}"

echo "[run-server-sim] N=$N shm=$SIM_SHM_PATH redis=127.0.0.1:$SIM_STEP_REDIS_PORT db$SIM_STEP_REDIS_DB"
echo "[run-server-sim] difficulty: spawn_geo=$SIM_SPAWN_GEO_DIST agent_hp=$SIM_AGENT_HP agent_def=$SIM_AGENT_DEF boss_hp=$SIM_BOSS_HP approach=$SIM_APPROACH_SCALE ep_timeout=$SIM_EP_TIMEOUT"
cd "$BIN" || exit 1
# CPU placement (env-overridable). Async overlap (SIM_ASYNC) needs the worlds to tick in
# PARALLEL across many cores, so the default nice -19 (which starves them to ~1 core under load)
# is overridable: SIM_SERVER_NICE sets the nice level and SIM_SERVER_CPUS pins the worlds to a
# core set (e.g. 4-31), leaving a few cores for the GPU-bound trainer so the two do not fight.
NICE_LEVEL="${SIM_SERVER_NICE:-19}"
if [ -n "${SIM_SERVER_CPUS:-}" ]; then
  exec taskset -c "$SIM_SERVER_CPUS" nice -n "$NICE_LEVEL" "$DOTNET" WorldServer.dll wServer.sim.json
fi
exec nice -n "$NICE_LEVEL" "$DOTNET" WorldServer.dll wServer.sim.json
