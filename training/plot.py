"""Render training metrics as a standalone HTML chart.

Reads the run histories that train.py writes to models/runs/ and draws one panel
per metric, so several runs can be compared directly. The chart is plain SVG
generated here - no plotting library, no JavaScript, no server.

    python -m training.plot                      # every run in models/runs/
    python -m training.plot models/runs/a.json   # specific runs
    python -m training.plot run.log              # a saved training stdout log
"""

import argparse
import ast
import html
import json
import re
import sys
from pathlib import Path

from training.config import RUNS

# Distinguishable at a glance in both light and dark themes.
PALETTE = ["#2f81f7", "#d29922", "#3fb950", "#f85149", "#a371f7", "#db6d28"]

WIDTH, HEIGHT = 900, 300
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 64, 24, 16, 46

# Each panel: (train key, holdout key or None, title, what a healthy curve looks like)
PANELS = [
    ("loss", "eval_loss", "Loss",
     "Holdout dashed. Where it turns up while training loss keeps falling is your epoch budget."),
    ("mean_token_accuracy", "eval_mean_token_accuracy", "Token accuracy",
     "Fraction of next tokens predicted exactly. Above ~0.9 on training data is memorisation."),
    ("entropy", "eval_entropy", "Entropy",
     "Confidence of the output distribution, in nats. A collapse toward 0 means the model "
     "commits hard to memorised continuations."),
    ("grad_norm", None, "Gradient norm",
     "Size of the update signal before clipping. Sustained values above max_grad_norm (1.0) "
     "mean every step is being clipped."),
]

# transformers prints one dict per logging step, e.g.
# {'loss': 1.8, 'grad_norm': 0.4, 'learning_rate': 0.0002, 'epoch': 0.03}
LOG_LINE = re.compile(r"\{'loss':.*?\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run JSON files or a saved training log")
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument("--smoothing", type=float, default=0.8,
                        help="EMA weight for the solid line, 0 disables (default 0.8)")
    return parser.parse_args()


def load_run(path: Path) -> dict:
    """Load one run, from either a train.py history JSON or a raw stdout log."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return {"name": data.get("run", path.stem), "history": data["log_history"]}

    # Fall back to scraping the dicts transformers prints at each logging step.
    history = [ast.literal_eval(match) for match in LOG_LINE.findall(path.read_text())]
    return {"name": path.stem, "history": history}


def points(history: list[dict], key: str) -> list[tuple[float, float]]:
    """(epoch, value) pairs, skipping entries that do not carry this metric."""
    return [(log["epoch"], log[key]) for log in history if key in log and "epoch" in log]


def smooth(values: list[float], weight: float) -> list[float]:
    """Exponential moving average - step-level metrics are too noisy to read raw."""
    if weight <= 0:
        return values
    out, acc = [], values[0]
    for value in values:
        acc = acc * weight + value * (1 - weight)
        out.append(acc)
    return out


def ticks(low: float, high: float, count: int = 4) -> list[float]:
    step = (high - low) / count
    return [low + step * i for i in range(count + 1)]


def panel(runs: list[dict], train_key: str, eval_key: str | None,
          title: str, note: str, smoothing: float, x_max: float) -> str:
    """One metric across every run, or an empty string if no run has it."""
    series = []
    for index, run in enumerate(runs):
        train = points(run["history"], train_key)
        held = points(run["history"], eval_key) if eval_key else []
        if train or held:
            series.append((run["name"], PALETTE[index % len(PALETTE)], train, held))
    if not series:
        return ""

    ys = [y for _, _, tr, ev in series for _, y in tr + ev]
    y_min, y_max = min(ys), max(ys)
    pad = (y_max - y_min) * 0.08 or 0.1
    y_min, y_max = y_min - pad, y_max + pad

    def sx(x):
        return PAD_LEFT + (x / (x_max or 1)) * (WIDTH - PAD_LEFT - PAD_RIGHT)

    def sy(y):
        span = y_max - y_min or 1
        return PAD_TOP + (1 - (y - y_min) / span) * (HEIGHT - PAD_TOP - PAD_BOTTOM)

    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">']

    for value in ticks(y_min, y_max):
        y = sy(value)
        parts.append(f'<line class="grid" x1="{PAD_LEFT}" y1="{y:.1f}" '
                     f'x2="{WIDTH - PAD_RIGHT}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_LEFT - 10}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{value:.3g}</text>')

    for value in ticks(0.0, x_max):
        parts.append(f'<text class="tick" x="{sx(value):.1f}" '
                     f'y="{HEIGHT - PAD_BOTTOM + 20:.0f}" text-anchor="middle">{value:.1f}</text>')
    parts.append(f'<text class="axis" x="{WIDTH / 2:.0f}" y="{HEIGHT - 8}" '
                 f'text-anchor="middle">epoch</text>')

    for _, colour, train, held in series:
        if train:
            raw = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in train)
            parts.append(f'<polyline class="raw" points="{raw}" stroke="{colour}"/>')
            eased = smooth([y for _, y in train], smoothing)
            line = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for (x, _), y in zip(train, eased))
            parts.append(f'<polyline class="line" points="{line}" stroke="{colour}"/>')
        if held:
            dashed = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in held)
            parts.append(f'<polyline class="eval" points="{dashed}" stroke="{colour}"/>')
            # Ring the best holdout value - that epoch is the budget.
            bx, by = min(held, key=lambda p: p[1]) if "loss" in (eval_key or "") \
                else max(held, key=lambda p: p[1])
            parts.append(f'<circle class="best" cx="{sx(bx):.1f}" cy="{sy(by):.1f}" '
                         f'r="4.5" stroke="{colour}"/>')

    parts.append("</svg>")
    return (f'<section><h2>{html.escape(title)}</h2><p class="note">{html.escape(note)}</p>'
            f'<div class="chart">{"".join(parts)}</div></section>')


def render(runs: list[dict], smoothing: float = 0.8) -> str:
    runs = [r for r in runs if points(r["history"], "loss")]
    if not runs:
        raise ValueError("no loss entries found in those runs")

    x_max = max(x for r in runs for x, _ in points(r["history"], "loss"))

    legend = []
    for index, run in enumerate(runs):
        colour = PALETTE[index % len(PALETTE)]
        loss = points(run["history"], "loss")
        held = points(run["history"], "eval_loss")
        meta = f"{len(loss)} steps &middot; train loss {loss[0][1]:.3f} &rarr; {loss[-1][1]:.3f}"
        if held:
            bx, by = min(held, key=lambda p: p[1])
            meta += f" &middot; best holdout {by:.3f} @ epoch {bx:g}"
        legend.append(f'<li><span class="swatch" style="background:{colour}"></span>'
                      f'<b>{html.escape(run["name"])}</b><span class="meta">{meta}</span></li>')

    panels = [panel(runs, t, e, title, note, smoothing, x_max) for t, e, title, note in PANELS]
    return TEMPLATE.format(panels="\n".join(p for p in panels if p),
                           legend="\n".join(legend))


TEMPLATE = """<title>Training Metrics</title>
<style>
  :root {{ --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --grid:#d8dee4; --card:#f6f8fa; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --grid:#30363d; --card:#161b22;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --grid:#30363d; --card:#161b22;
  }}
  body {{ margin:0; padding:32px; background:var(--bg); color:var(--fg);
         font:15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:0 0 2px; }}
  p.sub {{ margin:0 0 8px; color:var(--muted); font-size:13px; }}
  p.note {{ margin:0 0 8px; color:var(--muted); font-size:12px; }}
  section {{ margin-top:28px; }}
  .chart {{ background:var(--card); border:1px solid var(--grid); border-radius:10px;
            padding:8px; overflow-x:auto; }}
  svg {{ display:block; width:100%; min-width:640px; height:auto; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .tick {{ fill:var(--muted); font-size:11px; }}
  .axis {{ fill:var(--muted); font-size:12px; }}
  .raw {{ fill:none; stroke-width:1; opacity:0.22; }}
  .line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
  .eval {{ fill:none; stroke-width:2; stroke-dasharray:6 4; opacity:0.9; }}
  .best {{ fill:none; stroke-width:2; }}
  ul {{ list-style:none; padding:0; margin:16px 0 0; display:flex; flex-wrap:wrap; gap:8px 24px; }}
  li {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
  .swatch {{ width:12px; height:12px; border-radius:3px; flex:none; }}
  .meta {{ color:var(--muted); }}
</style>
<main>
  <h1>Training metrics</h1>
  <p class="sub">Solid line is smoothed, faint line is raw per-step.</p>
  <ul>
{legend}
  </ul>
{panels}
</main>
"""


def collect(paths: list[Path] | None = None) -> list[dict]:
    """Load the runs to chart, defaulting to everything in models/runs/."""
    if not paths:
        paths = sorted(RUNS.glob("*.json"))
    return [load_run(path) for path in paths]


def main() -> int:
    args = parse_args()

    if args.runs:
        paths = [Path(p) for p in args.runs]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"ERROR: not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
            return 1
    else:
        paths = sorted(RUNS.glob("*.json"))
        if not paths:
            print(f"ERROR: no runs in {RUNS} - run 'make train' first.", file=sys.stderr)
            print("A saved training log also works: python -m training.plot run.log",
                  file=sys.stderr)
            return 1

    try:
        page = render(collect(paths), args.smoothing)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else RUNS / "metrics.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)

    print(f"Wrote {out} ({len(paths)} run{'s' if len(paths) != 1 else ''})")
    print(f"Open it with: open {out}")
    print("Live view while training: make metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
