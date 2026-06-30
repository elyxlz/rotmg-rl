import sys, time
sys.path.insert(0,"/home/audiogen/snakepit-harness")
sys.path.insert(0,"/home/audiogen/rotmg-rl")
sys.path.insert(0,"/home/audiogen/rotmg-rl/src")
import config, dungeon, input as inp
from pathlib import Path
from Xlib import display, X, XK
from Xlib.ext import xtest
ok = dungeon.enter()
print("entered:", ok)
time.sleep(1.0)
d=display.Display(":99"); POS=Path("/dev/shm/rotmg_pos.txt")
def pos():
    try:
        a,b=POS.read_text().split(); return float(a),float(b)
    except: return None
def hold(k,s):
    kc=d.keysym_to_keycode(XK.string_to_keysym(k))
    xtest.fake_input(d,X.KeyPress,kc); d.sync(); time.sleep(s)
    xtest.fake_input(d,X.KeyRelease,kc); d.sync()
inp.focus_stage(); time.sleep(0.3)
print("EXPECTED world: w(0,-) d(+,0) s(0,+) a(-,0)")
for key in ["w","d","s","a"]:
    p0=pos()
    if p0 is None: print(key,"nopos"); continue
    hold(key,1.0); time.sleep(0.3); p1=pos()
    print("%s: world d=(%+.2f,%+.2f)  (%.1f,%.1f)->(%.1f,%.1f)"%(key,p1[0]-p0[0],p1[1]-p0[1],p0[0],p0[1],p1[0],p1[1]))
    time.sleep(0.3)
