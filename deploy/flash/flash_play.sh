#!/bin/bash
set -u
W=~/flash_client_build; VID=~/rotmg-rl/videos
[ -f "$W/client.swf" ] || { echo "no client.swf, rebuilding"; ~/flash_build.sh >/dev/null 2>&1; }
export DISPLAY=:99
pkill -f "Xvfb :99" 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &
sleep 2
LIBGL_ALWAYS_SOFTWARE=1 HOME=$HOME "$W/proj/flashplayer" "$W/client.swf" >/tmp/fp.log 2>&1 & FP=$!
shot(){ ffmpeg -y -f x11grab -video_size 1024x768 -i :99 -frames:v 1 "$VID/$1" >/dev/null 2>&1; echo "shot $1"; }
sleep 15; shot pp_1_loggedin.png
python3 ~/flash_click.py play; sleep 5; shot pp_2_afterplay.png
python3 ~/flash_click.py center; sleep 8; shot pp_3_aftercenter.png
# record 15s of whatever world state we are in
ffmpeg -y -f x11grab -video_size 1024x768 -framerate 20 -i :99 -t 15 -pix_fmt yuv420p "$VID/real_client_world.mp4" >/dev/null 2>&1
shot pp_4_final.png
echo "=== fp.log tail ==="; tail -15 /tmp/fp.log
kill $FP 2>/dev/null; pkill -f "Xvfb :99" 2>/dev/null
