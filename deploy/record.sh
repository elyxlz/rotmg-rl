#!/bin/bash
# ============================================================================
# record_snakepit.sh - record the bot fighting in a Snake Pit, as rendered by
# the REAL Flash spectator client, for a FULL entrance->fight window (>60s).
# Policy-independent (no-policy bot harness; swap in the trained policy later,
# see bottom).
#
# PROVEN MECHANISM
#  1. Bot (Wizardbot, nrelay plugin verify-shoot.ts): connects -> /max ->
#     /spawn Snake Pit Portal -> enters its OWN Snake Pit instance -> applies
#     /eff Invincible (server-side, via the game protocol = 100% reliable, the
#     bot never dies) -> /spawn-s the boss (Stheno the Snake Queen) at its own
#     entrance position and fights it (moving + shooting). All via packets.
#  2. Spectator-into-bot-instance: the Watcher Flash client joins the bot world
#     with admin  /visit Wizardbot  (Command.Visit -> Reconnect into the target
#     player World). Typed into the real Flash projector under Xvfb via a tiny
#     EWMH WM (~/miniwm) + an xtest focus CLICK + xtest keystrokes
#     (~/flash_type.py). Keyboard works in the NEXUS (pre-reconnect); that is
#     enough to send /visit.
#
#  SPECTATOR SURVIVAL (the >60s fix) -- METHOD 3, server-side persistent flag:
#     RotMG is permadeath and the projector drops keyboard focus AFTER the
#     /visit world-reconnect, so we cannot type /eff Invincible in-pit. Instead
#     we set the Watcher account's `hidden` flag in Redis BEFORE login
#     (HSET account.2 hidden \x01). betterSkillys reconstructs the Player on
#     EVERY world entry, and Player.cs does:
#         if (account.Hidden) { IsHidden = true;
#                               ApplyPermanentConditionEffect(Invincible); }
#     so the spectator is granted permanent Invincible on the nexus spawn AND
#     again on the /visit pit reconnect -- it survives the snake swarm for the
#     whole fight with ZERO post-reconnect keyboard. (Hidden also makes the
#     spectator invisible to others, which is irrelevant: we record the
#     spectator's own camera, which still renders the bot + boss fight.)
#     DbAccount.Hidden is GetValue<bool>("hidden") => a single redis byte,
#     0x01 = true (Shared/database/RedisObject.cs).
#  3. ffmpeg records :99 for the full window while the bot fights; a multi-point
#     survival check rejects runs where the spectator died (death -> char-select
#     = frozen/dark tail).
#
# POLICY RUN: re-enable policy-bridge (mv nrelay/lib/policy-bridge.js.off
# policy-bridge.js) and disable verify-shoot (mv verify-shoot.js .off); the
# spectator visit + survival + record half here is identical and unchanged.
# The real policy bot will navigate to the real boss room; the spectator
# /visit + hidden-invincibility carries it there alive.
# ============================================================================
set -u
W=~/flash_client_build; VID=~/rotmg-rl/videos
NODE=/home/audiogen/.nvm/versions/node/v18.16.1/bin/node
BOT=~/rotmg-realgame/nrelay
REC_SECS=${1:-70}
MAX_TRIES=${2:-6}
OUT=${3:-$VID/snakepit_survive.mp4}
export DISPLAY=:99

# Set/clear the Watcher account (account.2) Redis `hidden` byte. $1 = 1|0.
set_hidden() {
  python3 - "$1" << "PY"
import socket, sys, time
def rcmd(*args):
    s = socket.socket(); s.connect(("127.0.0.1", 6379)); s.settimeout(3)
    out = b"*%d\r\n" % len(args)
    for a in args:
        ab = a if isinstance(a, bytes) else str(a).encode()
        out += b"$%d\r\n" % len(ab) + ab + b"\r\n"
    s.sendall(out); time.sleep(0.15)
    d = b""
    try:
        while True:
            ch = s.recv(65536)
            if not ch: break
            d += ch
            if len(ch) < 65536: break
    except Exception: pass
    return d
val = b"\x01" if sys.argv[1] == "1" else b"\x00"
rcmd("HSET", "account.2", "hidden", val)
print("account.2 hidden =", "1" if sys.argv[1] == "1" else "0")
PY
}

# Read the Watcher char's (char.2.1) Redis `dead` byte: "1" if dead, "0" alive.
watcher_dead() {
  python3 - << "PY"
import socket, time
def rcmd(*args):
    s = socket.socket(); s.connect(("127.0.0.1", 6379)); s.settimeout(3)
    out = b"*%d\r\n" % len(args)
    for a in args:
        ab = a if isinstance(a, bytes) else str(a).encode()
        out += b"$%d\r\n" % len(ab) + ab + b"\r\n"
    s.sendall(out); time.sleep(0.15)
    d = b""
    try:
        while True:
            ch = s.recv(65536)
            if not ch: break
            d += ch
            if len(ch) < 65536: break
    except Exception: pass
    return d
r = rcmd("HGET", "char.2.1", "dead")
# bulk reply "$1\r\n<byte>\r\n"; the payload byte is 0x01 when dead.
print("1" if (b"\x01" in r.split(b"\r\n", 1)[-1][:1]) else "0")
PY
}

# Survival gate. AUTHORITATIVE signal = the server's own `dead` flag on the
# Watcher char (it cannot lie: RotMG death persists it). betterSkillys death
# drops the client to a LIT char-select screen, so luma alone can't see it --
# only the server knows. Visual checks are sanity only: every sampled frame is
# lit (the world actually rendered, not a black/failed reconnect) and the clip
# carries motion overall (the bot is fighting, not a frozen tail). Sampling
# starts at t=10 to skip the /visit reconnect-load.
survived() {
  local clip="$1" dur="$2"
  local dead; dead=$(watcher_dead)
  local pts=(10 $((dur/4)) $((dur/2)) $((dur*3/4)) $((dur-3)))
  rm -f "$VID"/_sv_*.png
  local i=0
  for t in "${pts[@]}"; do
    ffmpeg -y -ss "$t" -i "$clip" -frames:v 1 "$VID/_sv_$i.png" >/dev/null 2>&1
    i=$((i+1))
  done
  python3 - "$VID" "${#pts[@]}" "$dead" << "PY"
import sys, os
from PIL import Image, ImageChops
base, n, dead = sys.argv[1], int(sys.argv[2]), sys.argv[3]
imgs = []
for i in range(n):
    p = f"{base}/_sv_{i}.png"
    if not os.path.exists(p):
        print("0|no_frame"); sys.exit()
    imgs.append(Image.open(p).convert("RGB"))
lums = [sum(list(im.convert("L").getdata())) / (im.size[0] * im.size[1]) for im in imgs]
diffs = [sum(list(ImageChops.difference(a, b).convert("L").getdata()))/(a.size[0]*a.size[1]) for a, b in zip(imgs, imgs[1:])]
avg_diff = sum(diffs)/len(diffs)
alive = (dead == "0")            # authoritative: server says the spectator never died
lit = all(l > 45 for l in lums)  # world rendered in every sample (not black)
moving = avg_diff > 1.0          # the fight is live overall (loose, not per-pair)
ok = alive and lit and moving
print("%d|dead=%s|min_luma=%.1f|avg_diff=%.2f" % (1 if ok else 0, dead, min(lums), avg_diff))
PY
}

cleanup_all(){ pkill -9 -f flashplayer 2>/dev/null; pkill -9 -f miniwm 2>/dev/null; pkill -9 -f "Xvfb :99" 2>/dev/null; pkill -9 -f start-bot.js 2>/dev/null; }

SUCCESS=0
for try in $(seq 1 $MAX_TRIES); do
  echo "===== attempt $try/$MAX_TRIES (rec ${REC_SECS}s) ====="
  cleanup_all; sleep 2
  python3 ~/revive_watcher.py
  set_hidden 1                       # <-- spectator invincibility (survives the /visit reconnect)
  Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & sleep 2
  [ -x ~/miniwm ] || gcc ~/miniwm.c -o ~/miniwm -lX11
  ~/miniwm >/tmp/miniwm.log 2>&1 & sleep 1
  LIBGL_ALWAYS_SOFTWARE=1 HOME=$HOME "$W/proj/flashplayer" "$W/client.swf" >/tmp/fp.log 2>&1 & sleep 15
  python3 ~/flash_click.py play; sleep 4
  python3 ~/flash_click.py click 592 314; sleep 6
  python3 ~/flash_type.py focusclick; sleep 1

  rm -f /tmp/bot.log
  ( cd "$BOT" && nohup $NODE start-bot.js > /tmp/bot.log 2>&1 & )
  for i in $(seq 1 40); do grep -q "BOSS bossHP" /tmp/bot.log && { echo "  bot at boss (${i}s)"; break; }; sleep 1; done

  python3 ~/flash_type.py chat "/visit Wizardbot"; sleep 3
  ffmpeg -y -f x11grab -video_size 1024x768 -i :99 -frames:v 1 "$VID/sp_2_after_visit.png" >/dev/null 2>&1

  echo "  recording ${REC_SECS}s..."
  ffmpeg -y -f x11grab -video_size 1024x768 -framerate 20 -i :99 -t $REC_SECS -pix_fmt yuv420p -an "$OUT" >/tmp/ffrec.log 2>&1

  SV=$(survived "$OUT" "$REC_SECS")
  echo "  survival=$SV (leading 1 = alive whole clip)"
  if [ "${SV%%|*}" = "1" ]; then echo "  SUCCESS on attempt $try"; SUCCESS=1; break; fi
  echo "  spectator did not survive the full clip; retrying"
done

echo "[verify]"
ffmpeg -y -ss $((REC_SECS/2)) -i "$OUT" -frames:v 1 "$VID/snakepit_frame.png" >/dev/null 2>&1
ffprobe -v error -show_entries format=duration:stream=width,height -of default=noprint_wrappers=1 "$OUT" 2>&1 | head
python3 - << "PY"
import os
from PIL import Image
g=Image.open(os.path.expanduser("~/rotmg-rl/videos/snakepit_frame.png")).convert("L"); px=list(g.getdata())
print("mid-frame luma mean=%.1f nonblack_frac=%.3f"%(sum(px)/len(px), sum(1 for p in px if p>20)/len(px)))
PY
grep "BOSS bossHP" /tmp/bot.log | tail -1
cleanup_all
set_hidden 0                         # leave the Watcher account clean (script re-sets it each run)
python3 ~/revive_watcher.py
echo "RESULT: success=$SUCCESS out=$OUT"
echo "DONE_RECORD_SNAKEPIT"
