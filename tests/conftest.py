"""Shared fixtures.

The core suite deliberately depends on nothing heavier than pytest: every module
under test imports only the standard library at module level, and the training
stack (torch, transformers, trl, peft) is imported lazily inside functions. That
keeps `make test` runnable on a fresh clone with no GPU, no model download and no
network access.
"""

import json
from types import SimpleNamespace

import pytest

SYSTEM = "You are a knowledgeable assistant that answers questions about caring for carnivorous plants."


def example(question: str, answer: str = "An answer.", system: str = SYSTEM) -> dict:
    """One well-formed training example."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages += [{"role": "user", "content": question},
                 {"role": "assistant", "content": answer}]
    return {"messages": messages}


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def jsonl(tmp_path):
    """Write rows to a temp .jsonl and return the path."""
    def _write(rows, name="data.jsonl"):
        path = tmp_path / name
        write_jsonl(path, rows)
        return path
    return _write


@pytest.fixture
def run_history():
    """A minimal run history in the shape train.py records."""
    def _build(name="run", epochs=3, holdout=True):
        history = []
        for step in range(1, epochs * 4 + 1):
            history.append({
                "loss": 2.0 - step * 0.05,
                "grad_norm": 0.5 + step * 0.02,
                "mean_token_accuracy": 0.4 + step * 0.01,
                "entropy": 2.0 - step * 0.04,
                "epoch": step / 4,
                "step": step,
            })
        if holdout:
            # U-shaped: best at epoch 2, worse after - the memorisation signature.
            for epoch, value in zip(range(1, epochs + 1), [1.9, 1.7, 1.85]):
                history.append({"eval_loss": value, "eval_mean_token_accuracy": 0.5,
                                "eval_entropy": 1.5, "epoch": epoch})
        return {"run": name, "examples": 100, "holdout": 10,
                "hyperparameters": {"epochs": epochs}, "log_history": history}
    return _build


@pytest.fixture
def runs_dir(tmp_path, monkeypatch, run_history):
    """An isolated models/runs/ directory, with plot.RUNS pointed at it."""
    from training import plot

    directory = tmp_path / "runs"
    directory.mkdir()
    monkeypatch.setattr(plot, "RUNS", directory)

    def _add(name="run", **kwargs):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(run_history(name=name, **kwargs)))
        return path

    return SimpleNamespace(path=directory, add=_add)
