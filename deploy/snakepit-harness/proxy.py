"""The tee proxy. Run as a standalone subprocess (its stdout is piped into the obs consumer).

It listens on PROXY_PORT, and for each client connection (the redirected Flash client) opens an
upstream connection to the real server on GAME_PORT and relays both directions. The server->client
direction is also teed to stdout so the obs consumer can frame packets and build observations.

It alters nothing on the wire -- a transparent byte relay plus a read-only tee. Distinct from
obs_reader's tcpdump path: here we are already in the data path (the client connects to us), so we
copy bytes through and duplicate the inbound stream to stdout.

Run:  python proxy.py        (writes status to stderr; tee'd bytes to stdout)
"""

from __future__ import annotations

import socket
import sys
import threading

import config


def _relay(src: socket.socket, dst: socket.socket, tee: bool) -> None:
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
            if tee:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
    except OSError as exc:
        print("[proxy] relay closed: %r" % exc, file=sys.stderr, flush=True)
    for sock in (src, dst):
        try:
            sock.close()
        except OSError:
            pass


def _handle(client: socket.socket) -> None:
    print("[proxy] client connected", file=sys.stderr, flush=True)
    try:
        upstream = socket.create_connection(("127.0.0.1", config.GAME_PORT))
    except OSError as exc:
        print("[proxy] upstream connect failed: %r" % exc, file=sys.stderr, flush=True)
        client.close()
        return
    # client->server: relay only. server->client: relay AND tee to stdout (the obs feed).
    threading.Thread(target=_relay, args=(client, upstream, False), daemon=True).start()
    _relay(upstream, client, True)


def main() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", config.PROXY_PORT))
    listener.listen(8)
    print("[proxy] listening on 127.0.0.1:%d -> 127.0.0.1:%d" % (config.PROXY_PORT, config.GAME_PORT), file=sys.stderr, flush=True)
    while True:
        client, _ = listener.accept()
        threading.Thread(target=_handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
