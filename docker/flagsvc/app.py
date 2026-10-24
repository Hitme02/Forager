"""A deliberately-vulnerable research fixture service for the Tier-B isolated range.

This is a CTF-style target: a tiny self-contained HTTP service with an INTENTIONAL default
credential and a planted crown-jewel flag. It exists only so the Tier-B agent harness has a
real service to drive over localhost on an isolated, no-egress Docker network -- the standard
"cyber range / vulnerable-by-design" pattern used for security education and research. It
holds no real data and talks to nothing external.

It exposes exactly the endpoints the bounded action menu in `ffs/realism/agent_harness.py`
uses, and nothing else:

    GET  /                 liveness (probe)                 -> host_discovered
    GET  /version          banner/version (enumerate)       -> service_enumerated
    POST /login            {"user","password"} default creds -> foothold_shell (token)
    GET  /artifact/<n>     an artifact, decreasing in size   -> collection value
                           the flag service's deep artifact contains FLAG{...} (objective)

Difficulty and whether this instance holds the flag are set by environment variables so the
compose file can stand up a heterogeneous range of several instances with one holding the
crown jewel. Only standard library is used, so the image is tiny and auditable.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVICE_ID = os.environ.get("SVC_ID", "0")
HOLDS_FLAG = os.environ.get("HOLDS_FLAG", "false").lower() == "true"
FLAG = os.environ.get("FLAG", "FLAG{isolated-research-range-crown-jewel}")
# Depth (artifact index) at which the crown jewel is readable on the flag service.
FLAG_DEPTH = int(os.environ.get("FLAG_DEPTH", "3"))
DEFAULT_USER = os.environ.get("DEFAULT_USER", "admin")
DEFAULT_PASS = os.environ.get("DEFAULT_PASS", "admin")

_TOKEN = "session-" + SERVICE_ID


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # keep the fixture quiet
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, {"service": SERVICE_ID, "status": "up"})
        elif path == "/version":
            self._send(200, {"service": SERVICE_ID, "version": "demo-fixture/1.0"})
        elif path.startswith("/artifact/"):
            try:
                n = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._send(404, {"error": "bad index"})
            if self.headers.get("X-Session") != _TOKEN:
                return self._send(401, {"error": "not authenticated"})
            # artifacts shrink with index: a concave collection return
            size = max(1, 4096 // (n + 1))
            payload = {"artifact": n, "data": "A" * size}
            if HOLDS_FLAG and n >= FLAG_DEPTH:
                payload["flag"] = FLAG
            self._send(200, payload)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/login":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if body.get("user") == DEFAULT_USER and body.get("password") == DEFAULT_PASS:
            self._send(200, {"token": _TOKEN})
        else:
            self._send(403, {"error": "invalid credentials"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "80"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
