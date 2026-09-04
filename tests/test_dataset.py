"""Dataset validation: the gate that stops a bad file costing you a training run."""

import pytest

from training.dataset import DatasetError, read_examples
from tests.conftest import example


def test_reads_well_formed_examples(jsonl):
    path = jsonl([example("Q1"), example("Q2")])
    assert len(read_examples(path)) == 2


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(DatasetError) as err:
        read_examples(missing)
    assert "nope.jsonl" in str(err.value)


def test_empty_file_is_rejected(jsonl):
    path = jsonl([])
    with pytest.raises(DatasetError):
        read_examples(path)


@pytest.mark.parametrize("bad, why", [
    ({"not_messages": []}, "no messages key"),
    ({"messages": []}, "empty messages"),
    ({"messages": [{"role": "user", "content": "only a question"}]}, "no assistant turn"),
    ({"messages": [{"role": "wizard", "content": "x"},
                   {"role": "assistant", "content": "y"}]}, "invalid role"),
    ({"messages": [{"role": "user"}, {"role": "assistant", "content": "y"}]}, "missing content"),
    ({"messages": [{"role": "user", "content": ""},
                   {"role": "assistant", "content": "y"}]}, "empty content"),
])
def test_malformed_examples_are_rejected(jsonl, bad, why):
    with pytest.raises(DatasetError):
        read_examples(jsonl([bad]))


def test_error_names_the_offending_line(jsonl):
    path = jsonl([example("fine"), example("also fine"), {"messages": []}])
    with pytest.raises(DatasetError) as err:
        read_examples(path)
    # Line 3 is the bad one; the message has to say so or it is no use.
    assert "3" in str(err.value)


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"messages": [}\n')
    with pytest.raises(DatasetError):
        read_examples(path)
