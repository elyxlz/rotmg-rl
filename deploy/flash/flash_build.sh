#!/bin/bash
set -u
W=~/flash_client_build
REPO=~/rotmg-realgame/betterSkillys/client
VID=~/rotmg-rl/videos
mkdir -p "$W" "$VID"; cd "$W"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
# --- SDK setup (idempotent) ---
if [ ! -x "$W/flexsdk/bin/mxmlc" ]; then
  log "download flex 4.16.1"; curl -s -L -o flex.tgz "https://archive.apache.org/dist/flex/4.16.1/binaries/apache-flex-sdk-4.16.1-bin.tar.gz"
  rm -rf flexsdk; mkdir flexsdk; tar xzf flex.tgz -C flexsdk --strip-components=1
fi
if [ ! -f "$W/flexsdk/frameworks/libs/player/32.0/playerglobal.swc" ]; then
  log "download AIR32 (airglobal)"; curl -s -o air.zip "https://fpdownload.macromedia.com/air/win/download/32.0/AdobeAIRSDK.zip"
  mkdir -p airsdk && (cd airsdk && unzip -q -o ../air.zip frameworks/libs/air/airglobal.swc)
  cp -rn airsdk/frameworks/libs/air flexsdk/frameworks/libs/ 2>/dev/null
  mkdir -p flexsdk/frameworks/libs/player/32.0
  cp airsdk/frameworks/libs/air/airglobal.swc flexsdk/frameworks/libs/player/32.0/playerglobal.swc
  sed -i "s#{playerglobalHome}#$W/flexsdk/frameworks/libs/player#g; s#<target-player>[0-9.]*</target-player>#<target-player>32.0</target-player>#" flexsdk/frameworks/flex-config.xml
fi
if [ ! -x "$W/proj/flashplayer" ]; then
  log "download projector"; curl -s -o flash_sa.tgz "https://fpdownload.macromedia.com/pub/flashplayer/updaters/32/flash_player_sa_linux.x86_64.tar.gz"
  mkdir -p proj; tar xzf flash_sa.tgz -C proj
fi
SDK=$W/flexsdk
# --- build copy + edits ---
log "prep source"; rm -rf clientbuild; cp -r "$REPO" clientbuild; cd clientbuild
printf "%s" "<Objects></Objects>" > src/kabam/rotmg/assets/prod/xmls/MiniDungeonHub.xml
( cd src/com/company/ui/fonts
  cp /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf MyriadPro.ttf
  cp /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf MyriadProBold.ttf
  for f in MyriadPro MyriadProBold MyriadProCFF MyriadProBoldCFF; do
    sed -i "s/MyriadPro\.otf/MyriadPro.ttf/g; s/MyriadProBold\.otf/MyriadProBold.ttf/g; s/embedAsCFF=\"true\"/embedAsCFF=\"false\"/g" $f.as; done )
LT=src/kabam/rotmg/account/web/services/WebLoadAccountTask.as
grep -q AUTOLOGIN $LT || sed -i "s#this.getAccountData();#this.getAccountData();\n         this.data.username=\"spectator@spec.com\";this.data.password=\"specpass123\";//AUTOLOGIN#" $LT
# --- iterative compile ---
compile(){ $SDK/bin/mxmlc -source-path+=src -library-path+=libs -locale=en_US \
  -default-size 800 600 -default-frame-rate=60 -default-background-color=0x000000 \
  -swf-version=15 -static-link-runtime-shared-libraries=true \
  -keep-as3-metadata+=Inject -keep-as3-metadata+=Embed -keep-as3-metadata+=PostConstruct -keep-as3-metadata+=ArrayElementType \
  -warnings=false -strict=true src/WebMain.as -output "$W/client.swf" > "$W/build.log" 2>&1; echo $?; }
log "compile"
for i in $(seq 1 30); do
  ec=$(compile); [ "$ec" = "0" ] && { log "BUILD OK ($i)"; break; }
  miss=$(grep "could not be found" "$W/build.log" | sed -E "s/.*Definition ([^ ]+):([^ ]+) could.*/\1.\2/" | sort -u)
  [ -z "$miss" ] && { log "BUILD FAIL"; grep -m5 Error: "$W/build.log"; exit 1; }
  for q in $miss; do for fl in $(grep -rl "import ${q};" src); do sed -i "\#import ${q//./\\.};#d" "$fl"; done; done
done
ls -lh "$W/client.swf" || exit 1
# --- run + capture ---
log "run+capture"
export DISPLAY=:99
mkdir -p ~/.macromedia/Flash_Player/\#Security/FlashPlayerTrust; echo "$W" > ~/.macromedia/Flash_Player/\#Security/FlashPlayerTrust/local.cfg
pkill -f "Xvfb :99" 2>/dev/null; sleep 1
Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & XP=$!
sleep 2
LIBGL_ALWAYS_SOFTWARE=1 HOME=$HOME "$W/proj/flashplayer" "$W/client.swf" >/tmp/fp.log 2>&1 & FP=$!
for t in 8 16 26 38; do
  sleep_to=$t
  while [ $(date +%s) -lt $((START+sleep_to)) ]; do sleep 0.5; done 2>/dev/null
  sleep 1
  ffmpeg -y -f x11grab -video_size 1024x768 -i :99 -frames:v 1 "$VID/rc_login_${t}s.png" >/dev/null 2>&1
done
# also a 12s clip
ffmpeg -y -f x11grab -video_size 1024x768 -framerate 20 -i :99 -t 12 -pix_fmt yuv420p "$VID/real_client_live.mp4" >/dev/null 2>&1
log "fp log tail:"; tail -8 /tmp/fp.log
kill $FP 2>/dev/null; pkill -f "Xvfb :99" 2>/dev/null
ls -lh "$VID"/rc_login_*.png "$VID/real_client_live.mp4" 2>&1
