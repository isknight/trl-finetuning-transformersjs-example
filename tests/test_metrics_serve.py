"""Live metrics server: serves the chart, survives an empty or broken runs dir."""

import http.client
import threading
from argparse import Namespace
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from training import metrics_serve


@contextmanager
def serving(refresh=5, smoothing=0.8):
    handler = metrics_serve.build_handler(Namespace(refresh=refresh, smoothing=smoothing))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get(port, path="/"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read().decode()
    finally:
        conn.close()


def test_serves_the_chart(runs_dir):
    runs_dir.add("live-run")
    with serving() as port:
        status, body = get(port)
    assert status == 200
    assert "live-run" in body
    assert "<h2>Loss</h2>" in body


def test_includes_a_refresh_directive(runs_dir):
    runs_dir.add()
    with serving(refresh=3) as port:
        _, body = get(port)
    assert 'http-equiv="refresh"' in body
    assert 'content="3"' in body


def test_refresh_can_be_disabled(runs_dir):
    runs_dir.add()
    with serving(refresh=0) as port:
        _, body = get(port)
    assert "http-equiv" not in body


def test_no_runs_yet_still_serves(runs_dir):
    """Starting the server before the first run must not fail."""
    with serving() as port:
        status, body = get(port)
    assert status == 200
    assert "no runs" in body.lower()


def test_unreadable_run_does_not_kill_the_server(runs_dir):
    (runs_dir.path / "corrupt.json").write_text("{ not json")
    with serving() as port:
        status, body = get(port)
    assert status == 200
    assert "could not render" in body.lower()


def test_unknown_paths_404(runs_dir):
    runs_dir.add()
    with serving() as port:
        assert get(port, "/nope")[0] == 404


def test_picks_up_new_data_between_requests(runs_dir):
    """A run in progress must be visible on refresh, not cached from startup."""
    runs_dir.add("first")
    with serving() as port:
        assert "second" not in get(port)[1]
        runs_dir.add("second")
        assert "second" in get(port)[1]
