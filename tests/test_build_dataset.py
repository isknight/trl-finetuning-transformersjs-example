"""Corpus assembly: merge sources, force one persona, drop duplicates."""

import json

import pytest

from training import build_dataset
from training.config import SYSTEM_PROMPT
from tests.conftest import example, write_jsonl


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Redirect the builder's fixed paths at a temp tree."""
    generated = tmp_path / "generated"
    generated.mkdir()
    out = tmp_path / "plants.jsonl"
    monkeypatch.setattr(build_dataset, "SEED", tmp_path / "seed.jsonl")
    monkeypatch.setattr(build_dataset, "GENERATED", generated)
    monkeypatch.setattr(build_dataset, "DATASET", out)
    monkeypatch.setattr(build_dataset, "ROOT", tmp_path)
    return type("Corpus", (), {"seed": tmp_path / "seed.jsonl",
                               "generated": generated, "out": out})


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_merges_seed_and_batches(corpus):
    write_jsonl(corpus.seed, [example("From seed")])
    write_jsonl(corpus.generated / "batch_01.jsonl", [example("From batch one")])
    write_jsonl(corpus.generated / "batch_02.jsonl", [example("From batch two")])

    assert build_dataset.main() == 0
    assert len(read(corpus.out)) == 3


def test_forces_the_shared_system_prompt(corpus):
    write_jsonl(corpus.seed, [example("Q", system="Some other persona entirely")])
    build_dataset.main()

    messages = read(corpus.out)[0]["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert sum(1 for m in messages if m["role"] == "system") == 1


def test_adds_a_system_prompt_when_absent(corpus):
    write_jsonl(corpus.seed, [example("Q", system=None)])
    build_dataset.main()

    messages = read(corpus.out)[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT


def test_drops_duplicate_questions_across_files(corpus):
    write_jsonl(corpus.seed, [example("How often should I water?")])
    write_jsonl(corpus.generated / "batch_01.jsonl",
                [example("How often should I water?"), example("Something else")])

    build_dataset.main()
    assert len(read(corpus.out)) == 2


def test_deduplication_ignores_case_and_spacing(corpus):
    write_jsonl(corpus.seed, [example("Tap water OK?")])
    write_jsonl(corpus.generated / "batch_01.jsonl", [example("  tap water ok?  ")])

    build_dataset.main()
    assert len(read(corpus.out)) == 1


def test_no_sources_exits_non_zero(corpus, capsys):
    assert build_dataset.main() == 1
    assert "ERROR" in capsys.readouterr().err


def test_malformed_source_exits_non_zero(corpus, capsys):
    write_jsonl(corpus.seed, [{"messages": []}])
    assert build_dataset.main() == 1
    assert "ERROR" in capsys.readouterr().err


def test_output_is_valid_input(corpus):
    """The builder's output must pass the validator it will later be read by."""
    from training.dataset import read_examples

    write_jsonl(corpus.seed, [example("Q1"), example("Q2")])
    build_dataset.main()
    assert len(read_examples(corpus.out)) == 2
