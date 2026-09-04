"""Load and validate the chat-formatted training dataset.

The dataset is JSONL: one JSON object per line with a "messages" array of
{role, content} objects, the format TRL's SFTTrainer consumes directly.
"""

import json
import sys
from pathlib import Path

from training.config import DATASET

VALID_ROLES = {"system", "user", "assistant"}


class DatasetError(Exception):
    """Raised when the dataset file is missing or malformed."""


def _fail(path: Path, line_no: int | None, problem: str) -> None:
    where = f"{path}:{line_no}" if line_no else str(path)
    raise DatasetError(f"{where}: {problem}")


def _validate_example(path: Path, line_no: int, obj: object) -> dict:
    if not isinstance(obj, dict):
        _fail(path, line_no, "expected a JSON object")
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        _fail(path, line_no, "missing a non-empty 'messages' array")

    roles = []
    for pos, message in enumerate(messages):
        if not isinstance(message, dict):
            _fail(path, line_no, f"message {pos} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            _fail(path, line_no, f"message {pos} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            _fail(path, line_no, f"message {pos} has empty or non-string content")
        roles.append(role)

    if "user" not in roles or "assistant" not in roles:
        _fail(path, line_no, "needs at least one user and one assistant turn")
    return obj


def read_examples(path: Path = DATASET) -> list[dict]:
    """Parse and validate every example, raising DatasetError on any problem."""
    if not path.exists():
        _fail(path, None, "dataset file not found")

    examples = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                _fail(path, line_no, f"invalid JSON ({exc.msg})")
            examples.append(_validate_example(path, line_no, obj))

    if not examples:
        _fail(path, None, "dataset contains no examples")
    return examples


def load_dataset(path: Path = DATASET):
    """Return the validated dataset as a HuggingFace Dataset."""
    from datasets import Dataset

    return Dataset.from_list(read_examples(path))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET
    try:
        examples = read_examples(path)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{path}: {len(examples)} valid examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
