import time, sys
from Xlib import display, X
from Xlib.ext import xtest
d = display.Display(":99")
root = d.screen().root
def move_click(x,y):
    xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(0.3)
    xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.08)
    xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.3)
cmd=sys.argv[1]
if cmd=="play": move_click(592,718)
elif cmd=="center": move_click(592,440)
elif cmd=="click":
    move_click(int(sys.argv[2]),int(sys.argv[3]))
