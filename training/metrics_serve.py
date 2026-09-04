"""Serve the training metrics chart, re-rendered on every request.

Point a browser at http://127.0.0.1:8001 and leave it open while training runs.
train.py rewrites its run history after every logging step, so the page shows the
current state of the run each time it refreshes - no tracking service involved.

    python -m training.metrics_serve [--port 8001] [--refresh 5]
"""

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from training.config import RUNS
from training.plot import collect, render

HOST = "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--refresh", type=int, default=5,
                        help="Auto-refresh interval in seconds; 0 disables")
    parser.add_argument("--smoothing", type=float, default=0.8)
    return parser.parse_args()


def build_handler(args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            # Loading and rendering are handled separately: json.JSONDecodeError is a
            # ValueError, so catching them together reports a corrupt run file as
            # "no runs yet" and quietly hides the real problem.
            try:
                runs = collect()
            except Exception as exc:  # a half-written or corrupt history is not fatal
                body = f"<title>Training Metrics</title><p>Could not render: {exc}</p>"
            else:
                try:
                    body = render(runs, args.smoothing)
                except ValueError:
                    body = ("<title>Training Metrics</title>"
                            "<p style='font:15px sans-serif'>No runs with metrics yet. "
                            "Start a run with <code>make train</code>.</p>")
                except Exception as exc:
                    body = f"<title>Training Metrics</title><p>Could not render: {exc}</p>"

            if args.refresh > 0:
                body = f'<meta http-equiv="refresh" content="{args.refresh}">' + body

            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            pass  # a page refreshing every few seconds would flood the terminal

    return Handler


def main() -> int:
    args = parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)

    try:
        server = ThreadingHTTPServer((HOST, args.port), build_handler(args))
    except OSError as exc:
        print(f"ERROR: cannot bind {HOST}:{args.port} - {exc}", file=sys.stderr)
        print("Another server may already be running; try --port 8002.", file=sys.stderr)
        return 1

    print(f"Training metrics on http://{HOST}:{args.port}")
    print(f"Reading {RUNS} | refreshing every {args.refresh}s | Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
