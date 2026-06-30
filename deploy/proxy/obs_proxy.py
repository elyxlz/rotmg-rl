import socket, threading, sys
def relay(s,d,tee=False):
    try:
        while True:
            b=s.recv(65536)
            if not b: break
            d.sendall(b)
            if tee:
                sys.stdout.buffer.write(b); sys.stdout.buffer.flush()
    except: pass
    for x in (s,d):
        try: x.close()
        except: pass
def handle(c):
    print("CONN",file=sys.stderr,flush=True)
    try: srv=socket.create_connection(("127.0.0.1",2050))
    except Exception as e: print("upstream fail",e,file=sys.stderr); c.close(); return
    threading.Thread(target=relay,args=(c,srv,False),daemon=True).start()
    relay(srv,c,True)
ls=socket.socket(); ls.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
ls.bind(("127.0.0.1",2052)); ls.listen(8); print("proxy up :2052->:2050",file=sys.stderr,flush=True)
while True:
    cc,_=ls.accept(); threading.Thread(target=handle,args=(cc,),daemon=True).start()
