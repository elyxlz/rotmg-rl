#!/usr/bin/env bash
# One idempotent Snake Pit attempt: release any live session, re-arm the deliverable T7 Wizard
# (char.1.2) server-side, then run the harness (fresh pipeline -> client -> char2 -> portal ->
# record + policy -> monitor). Re-runnable after every death. Video: ~/rotmg-rl/videos/snakepit_run.mp4
set -u
cd ~/snakepit-harness
echo "[attempt] release client + stale pipeline"
pkill -x flashplayer 2>/dev/null
fuser -k 2052/tcp 2>/dev/null
fuser -k /dev/shm/rotmg_obs.f32 2>/dev/null
sleep 3
echo "[attempt] re-arm char.1.2 (T7 staff+spell equipped, key, alive)"
( cd ~/rotmg-rl && .venv/bin/python ~/provision_wizard.py ) | tail -4
echo "[attempt] run harness"
export DISPLAY=:99
exec ~/rotmg-rl/.venv/bin/python run.py
