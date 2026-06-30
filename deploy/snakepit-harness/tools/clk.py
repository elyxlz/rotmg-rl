import sys, time
from Xlib import display, X
from Xlib.ext import xtest
d=display.Display(":99")
x,y=int(sys.argv[1]),int(sys.argv[2])
root=d.screen().root
root.warp_pointer(x,y); d.sync(); time.sleep(.2)
xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(.1)
xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(.08)
xtest.fake_input(d, X.ButtonRelease, 1); d.sync()
print("clicked",x,y)
