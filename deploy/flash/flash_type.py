import time, sys
from Xlib import display, X, XK
from Xlib.ext import xtest

d = display.Display(":99")
root = d.screen().root

def find_stage():
    cands = []
    def walk(w):
        try: kids = w.query_tree().children
        except Exception: return
        for c in kids:
            try:
                a = c.get_attributes(); g = c.get_geometry()
                if a.map_state == X.IsViewable and g.width >= 400 and g.height >= 400:
                    cands.append((c, g.width * g.height))
            except Exception: pass
            walk(c)
    walk(root)
    if not cands: return None
    return sorted(cands, key=lambda t: t[1])[0][0]  # smallest viewable >=400 = the stage

def focus(do_click=True):
    st = find_stage()
    if st is None: return
    try:
        d.set_input_focus(st, X.RevertToParent, X.CurrentTime); d.sync()
        if do_click:
            # A real click inside the stage establishes the Flash projector keyboard focus (needs miniwm running).
            xtest.fake_input(d, X.MotionNotify, x=450, y=400); d.sync(); time.sleep(0.05)
            xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.05)
            xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.15)
    except Exception: pass

def tap(keysym, shift=False):
    kc = d.keysym_to_keycode(keysym)
    if kc == 0: return
    sk = d.keysym_to_keycode(XK.XK_Shift_L)
    if shift: xtest.fake_input(d, X.KeyPress, sk); d.sync(); time.sleep(0.01)
    xtest.fake_input(d, X.KeyPress, kc); d.sync(); time.sleep(0.03)
    xtest.fake_input(d, X.KeyRelease, kc); d.sync(); time.sleep(0.03)
    if shift: xtest.fake_input(d, X.KeyRelease, sk); d.sync(); time.sleep(0.01)

def char_keysym(c):
    if c == " ": return (XK.XK_space, False)
    if c == "/": return (XK.XK_slash, False)
    if c.isalpha(): return (XK.string_to_keysym(c.lower()), c.isupper())
    return (XK.string_to_keysym(c), False)

def type_str(s):
    for c in s:
        ks, sh = char_keysym(c); tap(ks, sh)

cmd = sys.argv[1]
if cmd == "chat":
    focus(do_click=True)
    tap(XK.XK_Escape); time.sleep(0.4)   # toggle in-client console (chat)
    type_str(sys.argv[2]); time.sleep(0.2)
    tap(XK.XK_Return); time.sleep(0.2)
    tap(XK.XK_Escape); time.sleep(0.1)   # close console
elif cmd == "focusclick":
    focus(do_click=True)
elif cmd == "esc":
    focus(do_click=True); tap(XK.XK_Escape)
elif cmd == "type":
    focus(do_click=True); type_str(sys.argv[2])
elif cmd == "key":
    focus(do_click=True); tap(XK.string_to_keysym(sys.argv[2]))
