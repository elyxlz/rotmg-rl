import time
from Xlib import display, X, XK
from Xlib.ext import xtest
d = display.Display(":99")
def click(x,y):
    xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(0.25)
    xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.08)
    xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.35)
def tap(ks, shift=False):
    kc=d.keysym_to_keycode(ks)
    if kc==0: return
    sk=d.keysym_to_keycode(XK.XK_Shift_L)
    if shift: xtest.fake_input(d,X.KeyPress,sk); d.sync()
    xtest.fake_input(d,X.KeyPress,kc); d.sync(); time.sleep(0.03)
    xtest.fake_input(d,X.KeyRelease,kc); d.sync(); time.sleep(0.03)
    if shift: xtest.fake_input(d,X.KeyRelease,sk); d.sync()
def typ(s):
    for c in s:
        if c=="@": tap(XK.XK_2, True)
        elif c==".": tap(XK.XK_period)
        elif c.isalpha(): tap(XK.string_to_keysym(c.lower()), c.isupper())
        elif c.isdigit(): tap(XK.string_to_keysym(c))
        else: tap(XK.string_to_keysym(c))
        time.sleep(0.02)
click(587,403); time.sleep(0.3); typ("botwiz@bot.com")
click(587,490); time.sleep(0.3); typ("botpass123")
time.sleep(0.4); click(681,591)   # Sign in
print("typed creds + signed in")
