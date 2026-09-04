"""Metrics chart: self-contained, in-bounds, and honest about holdout."""

import json
import re

import pytest

from training import plot


def coords(html):
    return [tuple(map(float, pair.split(",")))
            for points in re.findall(r'points="([^"]+)"', html)
            for pair in points.split()]


def test_renders_a_run(runs_dir):
    runs_dir.add("only-run")
    html = plot.render(plot.collect())
    assert "only-run" in html
    assert "<svg" in html


def test_is_self_contained(runs_dir):
    """No network assets: the page must open straight from disk."""
    runs_dir.add()
    html = plot.render(plot.collect())
    assert "http://" not in html.replace("http-equiv", "")
    assert "https://" not in html
    assert "<script" not in html


def test_draws_a_panel_per_metric(runs_dir):
    runs_dir.add()
    html = plot.render(plot.collect())
    for title in ["Loss", "Token accuracy", "Entropy", "Gradient norm"]:
        assert f"<h2>{title}</h2>" in html


def test_holdout_is_drawn_and_best_is_marked(runs_dir):
    runs_dir.add(holdout=True)
    html = plot.render(plot.collect())
    assert 'class="eval"' in html
    assert 'class="best"' in html
    # The fixture's holdout bottoms out at epoch 2.
    assert "@ epoch 2" in html


def test_run_without_holdout_still_renders(runs_dir):
    runs_dir.add(holdout=False)
    html = plot.render(plot.collect())
    assert 'class="eval"' not in html
    assert "<svg" in html


def test_overlays_multiple_runs(runs_dir):
    runs_dir.add("first")
    runs_dir.add("second")
    html = plot.render(plot.collect())
    assert "first" in html and "second" in html
    # Distinct colours, or the overlay is unreadable.
    assert len(set(re.findall(r'stroke="(#[0-9a-f]{6})"', html))) >= 2


def test_geometry_stays_inside_the_viewport(runs_dir):
    runs_dir.add()
    points = coords(plot.render(plot.collect()))
    assert points
    assert all(0 <= x <= plot.WIDTH and 0 <= y <= plot.HEIGHT for x, y in points)
    assert not any(v != v for point in points for v in point)  # no NaN


def test_empty_runs_raise(runs_dir):
    with pytest.raises(ValueError):
        plot.render(plot.collect())


def test_reads_a_saved_training_log(tmp_path):
    """Runs recorded before metrics persistence existed can still be charted."""
    log = tmp_path / "run.log"
    log.write_text(
        "Training on 100 examples\n"
        "{'loss': 2.4, 'grad_norm': 0.5, 'learning_rate': 2e-05, 'epoch': 0.1}\n"
        "{'loss': 2.1, 'grad_norm': 0.4, 'learning_rate': 4e-05, 'epoch': 0.2}\n"
        "{'train_runtime': 90.0, 'epoch': 1.0}\n"
    )
    run = plot.load_run(log)
    assert len(plot.points(run["history"], "loss")) == 2
    assert "<svg" in plot.render([run])


def test_smoothing_zero_is_the_raw_series(runs_dir):
    values = [1.0, 2.0, 3.0]
    assert plot.smooth(values, 0) == values
    assert plot.smooth(values, 0.9) != values


def test_main_writes_a_file(runs_dir, tmp_path, monkeypatch, capsys):
    runs_dir.add()
    out = tmp_path / "metrics.html"
    monkeypatch.setattr("sys.argv", ["plot", "--out", str(out)])
    assert plot.main() == 0
    assert "<svg" in out.read_text()


def test_main_without_runs_exits_non_zero(runs_dir, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["plot"])
    assert plot.main() == 1
    assert "no runs" in capsys.readouterr().err.lower()
