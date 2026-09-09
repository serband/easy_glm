"""Readiness checks must identify the child, not an unrelated local server."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("fastapi")
from easy_glm.desktop import launch


def test_occupied_port_never_returns_unrelated_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - standard library handler contract
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok", "launch_id":"other-process"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="did not become ready"):
            launch(port=server.server_address[1], open_browser=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
