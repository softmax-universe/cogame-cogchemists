#!/usr/bin/env python3
"""Deterministic local Anthropic Messages fixture for offline Docker smoke and local certify --use-llm."""

import argparse
import json
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_handler(count_file: Path | None):
    lock = threading.Lock()
    state = {"count": 0}

    class FixtureHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            user = body["messages"][0]["content"]
            match = re.search(r"Round (\d+) of \d+, ([A-Z]+) phase", user)
            phase = match[2] if match else "MARKET"
            action = {"action": "forage"} if phase == "LAB" else {"action": "pass"}
            response = json.dumps(
                {
                    "model": body.get("model", "local-cert-fixture"),
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": json.dumps(action)}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
            ).encode("utf-8")
            with lock:
                state["count"] += 1
                if count_file is not None:
                    count_file.write_text(str(state["count"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("X-Softmax-Llm-Call-Id", str(uuid.uuid5(uuid.NAMESPACE_URL, raw.decode("utf-8"))))
            self.end_headers()
            self.wfile.write(response)

    return FixtureHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path)
    parser.add_argument("--count-file", type=Path)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.count_file))
    if args.port_file is not None:
        args.port_file.write_text(str(server.server_port))
    if args.count_file is not None:
        args.count_file.write_text("0")
    print(f"llm_fixture listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
