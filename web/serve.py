"""Serve the chat app and the exported model over HTTP.

Static files come from web/, and the ONNX export is mounted at /models/plant-bot/
so Transformers.js finds it under its localModelPath. Cross-origin isolation
headers are set because ONNX Runtime's multithreaded WASM build needs
SharedArrayBuffer.
"""

import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
ONNX = ROOT / "models" / "onnx"
MODEL_URL_PREFIX = "/models/plant-bot/"


class AppHandler(SimpleHTTPRequestHandler):
    """Serve web/ plus the model export, with cross-origin isolation headers."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith(MODEL_URL_PREFIX):
            relative = clean[len(MODEL_URL_PREFIX):]
            target = (ONNX / relative).resolve()
            if not str(target).startswith(str(ONNX.resolve())):
                return str(ONNX)  # refuse traversal outside the export
            return str(target)
        return super().translate_path(path)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not (ONNX / "onnx").is_dir() or not any((ONNX / "onnx").glob("*.onnx")):
        print(f"ERROR: no exported model in {ONNX}", file=sys.stderr)
        print("Run `make export` first.", file=sys.stderr)
        return 1

    handler = partial(AppHandler, directory=str(WEB))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving the chat app on http://{args.host}:{args.port}")
    print(f"Model mounted at {MODEL_URL_PREFIX} from {ONNX}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
