import sys, time, math
from Xlib import display, X, XK
from Xlib.ext import xtest
d = display.Display(":99")
CX, CY = 490, 400         # player is camera-centered on the stage
TILE = 36; R = 4*TILE     # cast lands ~4 tiles out along the aim
MOVE = [(math.cos(k*math.pi/4), math.sin(k*math.pi/4)) for k in range(8)]  # move 1..8 -> k 0..7
state = {"keys": set(), "shoot": False}
def kc(ch): return d.keysym_to_keycode(XK.string_to_keysym(ch))
def setkeys(nk):
    for k in state["keys"]-nk: xtest.fake_input(d,X.KeyRelease,kc(k))
    for k in nk-state["keys"]: xtest.fake_input(d,X.KeyPress,kc(k))
    d.sync(); state["keys"]=nk
def wasd(dx,dy):
    k=set()
    if dy<-0.3:k.add("w")
    if dy>0.3:k.add("s")
    if dx>0.3:k.add("d")
    if dx<-0.3:k.add("a")
    return k
def apply(move,aim,shoot,cast):
    if move==0: setkeys(set())
    else:
        dx,dy=MOVE[move-1]; setkeys(wasd(dx,dy))
    ang=aim*2*math.pi/32
    xtest.fake_input(d,X.MotionNotify,x=int(CX+R*math.cos(ang)),y=int(CY+R*math.sin(ang))); d.sync()
    if shoot and not state["shoot"]: xtest.fake_input(d,X.ButtonPress,1); state["shoot"]=True
    elif not shoot and state["shoot"]: xtest.fake_input(d,X.ButtonRelease,1); state["shoot"]=False
    d.sync()
    if cast:
        sp=d.keysym_to_keycode(XK.XK_space)
        xtest.fake_input(d,X.KeyPress,sp); d.sync(); time.sleep(0.02); xtest.fake_input(d,X.KeyRelease,sp); d.sync()
if __name__=="__main__":
    # focus stage
    xtest.fake_input(d,X.MotionNotify,x=490,y=400); d.sync(); time.sleep(0.05)
    xtest.fake_input(d,X.ButtonPress,1); d.sync(); time.sleep(0.05); xtest.fake_input(d,X.ButtonRelease,1); d.sync(); time.sleep(0.2)
    mv=int(sys.argv[1])  # test: hold a move dir 1.5s
    for _ in range(15): apply(mv,0,False,False); time.sleep(0.1)
    setkeys(set()); print("applied move",mv)
