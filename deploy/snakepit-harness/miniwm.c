/* Minimal WM: registers SubstructureRedirect + EWMH supporting-wm-check so GTK/Flash
   treat their window as active (accept keyboard), and focuses/activates mapped windows. */
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <stdio.h>
#include <string.h>

static Display *dpy;
static Window root, check;
static Atom A_SUPPORTING, A_ACTIVE, A_SUPPORTED, A_WMNAME, A_UTF8, A_NETWMSTATE;

static void activate(Window w) {
    if (w == None || w == root) return;
    XSetInputFocus(dpy, w, RevertToParent, CurrentTime);
    XRaiseWindow(dpy, w);
    XChangeProperty(dpy, root, A_ACTIVE, XA_WINDOW, 32, PropModeReplace, (unsigned char*)&w, 1);
}

int main(void) {
    if (!(dpy = XOpenDisplay(NULL))) { fprintf(stderr, "no display\n"); return 1; }
    root = DefaultRootWindow(dpy);
    A_SUPPORTING = XInternAtom(dpy, "_NET_SUPPORTING_WM_CHECK", False);
    A_ACTIVE     = XInternAtom(dpy, "_NET_ACTIVE_WINDOW", False);
    A_SUPPORTED  = XInternAtom(dpy, "_NET_SUPPORTED", False);
    A_WMNAME     = XInternAtom(dpy, "_NET_WM_NAME", False);
    A_UTF8       = XInternAtom(dpy, "UTF8_STRING", False);
    A_NETWMSTATE = XInternAtom(dpy, "_NET_WM_STATE", False);

    /* EWMH supporting-wm-check so toolkits believe a compliant WM is present */
    check = XCreateSimpleWindow(dpy, root, -10, -10, 1, 1, 0, 0, 0);
    XChangeProperty(dpy, check, A_SUPPORTING, XA_WINDOW, 32, PropModeReplace, (unsigned char*)&check, 1);
    XChangeProperty(dpy, root,  A_SUPPORTING, XA_WINDOW, 32, PropModeReplace, (unsigned char*)&check, 1);
    XChangeProperty(dpy, check, A_WMNAME, A_UTF8, 8, PropModeReplace, (unsigned char*)"miniwm", 6);
    Atom supported[] = { A_ACTIVE, A_SUPPORTING, A_NETWMSTATE };
    XChangeProperty(dpy, root, A_SUPPORTED, XA_ATOM, 32, PropModeReplace, (unsigned char*)supported, 3);

    XSelectInput(dpy, root, SubstructureRedirectMask | SubstructureNotifyMask);
    XSync(dpy, False);

    XEvent ev;
    for (;;) {
        XNextEvent(dpy, &ev);
        if (ev.type == MapRequest) {
            XMapWindow(dpy, ev.xmaprequest.window);
            activate(ev.xmaprequest.window);
        } else if (ev.type == ConfigureRequest) {
            XConfigureRequestEvent *c = &ev.xconfigurerequest;
            XWindowChanges wc; wc.x=c->x; wc.y=c->y; wc.width=c->width; wc.height=c->height;
            wc.border_width=c->border_width; wc.sibling=c->above; wc.stack_mode=c->detail;
            XConfigureWindow(dpy, c->window, c->value_mask, &wc);
        } else if (ev.type == MapNotify) {
            activate(ev.xmap.window);
        }
    }
    return 0;
}
