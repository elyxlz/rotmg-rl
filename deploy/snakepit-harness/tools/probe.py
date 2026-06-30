import time
from pathlib import Path
from Xlib import display, X, XK
from Xlib.ext import xtest
d = display.Display(":99")
POS = Path("/dev/shm/rotmg_pos.txt")
def pos():
    try:
        a,b = POS.read_text().split(); return float(a), float(b)
    except Exception: return None
def click(x,y):
    xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(.05)
    xtest.fake_input(d, X.ButtonPress,1); d.sync(); time.sleep(.05)
    xtest.fake_input(d, X.ButtonRelease,1); d.sync(); time.sleep(.2)
def hold(keychar, secs):
    kc = d.keysym_to_keycode(XK.string_to_keysym(keychar))
    xtest.fake_input(d, X.KeyPress, kc); d.sync(); time.sleep(secs)
    xtest.fake_input(d, X.KeyRelease, kc); d.sync()
click(490,400)  # grab stage focus
time.sleep(0.3)
print("EXPECTED (world): w->(0,-) a->(-,0) s->(0,+) d->(+,0)")
for key in ["w","a","s","d"]:
    p0=pos()
    if p0 is None: print(key,"NO POS"); continue
    hold(key,1.5); time.sleep(0.4); p1=pos()
    print("%s: world d=(%+.2f,%+.2f)   (%.1f,%.1f)->(%.1f,%.1f)" % (key,p1[0]-p0[0],p1[1]-p0[1],p0[0],p0[1],p1[0],p1[1]))
    time.sleep(0.6)
