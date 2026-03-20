#!/usr/bin/env python3
"""Simple web server for the Poodle demo config UI.

Usage:
    python server.py [--port 8080]

Endpoints:
  GET  /             → index.html
  POST /api/cost     → compute cost series from a demo_config JSON body
  POST /api/save     → save demo_config JSON to ../demo/saved_config.json
"""
import argparse
import http.server
import json
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable so demo.* modules work
ROOT      = Path(__file__).parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.demo_config import (
    DemoScenario, ModelConfig, RequestConfig,
    TokenConfig, ModelDevConfig, ValidationConfig,
    Model, TaskDetectionMethod,
)
from cost_break_even.price_estimation import single_model_price, poodle_price

SAVE_PATH = REPO_ROOT / "demo" / "saved_config.json"


# ── Config deserialization ─────────────────────────────────────────────────

def scenario_from_dict(data: dict) -> DemoScenario:
    """Build a DemoScenario from a plain dict (as sent by the browser)."""
    m = data.get("models", {})
    t = data.get("tokens", {})
    return DemoScenario(
        models=ModelConfig(
            large_model=Model(m.get("large_model", "gpt-4.1")),
            small_model=Model(m.get("small_model", "bert-80M")),
        ),
        requests=RequestConfig(**data.get("requests", {})),
        tokens=TokenConfig(
            input=t.get("input", ""),
            prompt=t.get("prompt", ""),
            task_detection_method=TaskDetectionMethod(
                t.get("task_detection_method", "wrapper_prompt")
            ),
            wrapper_prompt=t.get("wrapper_prompt", ""),
            wrapped_requests_percent=t.get("wrapped_requests_percent", 1.0),
            output=t.get("output", ""),
            wrapped_output=t.get("wrapped_output", ""),
        ),
        dev=ModelDevConfig(**data.get("dev", {})),
        validation=ValidationConfig(**data.get("validation", {})),
    )


# ── HTTP handler ───────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}]  {fmt % args}")

    # ── GET ────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(ROOT / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/example":
            scenario = DemoScenario.get_example_scenario()
            self._send(200, "application/json",
                       json.dumps(scenario.to_dict(), indent=2,
                                  default=lambda o: o.value if hasattr(o, 'value') else str(o)).encode())
        else:
            self._send(404, "text/plain", b"Not found")

    # ── POST ───────────────────────────────────────────────────────────

    def do_POST(self):
        body = self._read_body()
        if body is None:
            return

        if self.path == "/api/cost":
            self._handle_cost(body)
        elif self.path == "/api/save":
            self._handle_save(body)
        else:
            self._send(404, "application/json",
                       json.dumps({"error": "not found"}).encode())

    # ── Route handlers ─────────────────────────────────────────────────

    def _handle_cost(self, data: dict):
        try:
            scenario = scenario_from_dict(data)
            # 100 linearly spaced points from 1 to expected_requests
            n = 100
            max_r = scenario.requests.expected_requests
            request_values = [max(1, int(max_r * i / (n - 1))) for i in range(n)]
            reqs, base_list, poodle_list, savings_list = [], [], [], []
            for r in request_values:
                scenario.requests.expected_requests = r
                b = single_model_price(scenario.models.large_model, scenario)
                p = poodle_price(scenario)
                reqs.append(r)
                base_list.append(b)
                poodle_list.append(p)
                savings_list.append(b - p)

            result = {
                "requests":       reqs,
                "base_prices":    base_list,
                "poodle_prices":  poodle_list,
                "poodle_savings": savings_list,
            }
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as e:
            self._send(500, "application/json",
                       json.dumps({"error": str(e)}).encode())

    def _handle_save(self, data: dict):
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAVE_PATH.write_text(json.dumps(data, indent=2))
        print(f"  Config saved → {SAVE_PATH.relative_to(REPO_ROOT)}")
        self._send(200, "application/json",
                   json.dumps({"ok": True, "path": str(SAVE_PATH)}).encode())

    # ── Helpers ────────────────────────────────────────────────────────

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._send(400, "application/json",
                       json.dumps({"error": str(e)}).encode())
            return None

    def _serve_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self._send(404, "text/plain", b"File not found")
            return
        self._send(200, content_type, data)

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poodle config UI server")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = http.server.HTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  🐩  Poodle Config UI")
    print(f"  Running at  {url}")
    print(f"  Saves to    {SAVE_PATH.relative_to(REPO_ROOT)}")
    print(f"\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
