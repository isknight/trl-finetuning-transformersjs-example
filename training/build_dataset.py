"""Merge the seed and generated batches into one training corpus.

Combines training/data/seed.jsonl with every training/data/generated/batch_*.jsonl,
drops duplicate questions, enforces the shared system prompt, and validates the
result. Re-runnable: it always rebuilds the output from its inputs.
"""

import json
import re
import sys
from pathlib import Path

from training.config import DATASET, ROOT, SYSTEM_PROMPT
from training.dataset import DatasetError, read_examples

SEED = ROOT / "training" / "data" / "seed.jsonl"
GENERATED = ROOT / "training" / "data" / "generated"


def question_key(example: dict) -> str:
    """Normalized user question, for duplicate detection."""
    question = next(m["content"] for m in example["messages"] if m["role"] == "user")
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def main() -> int:
    sources = [SEED, *sorted(GENERATED.glob("batch_*.jsonl"))]
    sources = [p for p in sources if p.exists()]
    if not sources:
        print(f"ERROR: no input files found under {GENERATED}", file=sys.stderr)
        return 1

    seen: dict[str, Path] = {}
    merged: list[dict] = []
    duplicates = 0

    for source in sources:
        try:
            examples = read_examples(source)
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        kept = 0
        for example in examples:
            # Force the shared system prompt so every example trains one persona.
            messages = [m for m in example["messages"] if m["role"] != "system"]
            example = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages]}

            key = question_key(example)
            if key in seen:
                duplicates += 1
                continue
            seen[key] = source
            merged.append(example)
            kept += 1

        print(f"  {source.relative_to(ROOT)}: {len(examples)} read, {kept} kept")

    with DATASET.open("w", encoding="utf-8") as handle:
        for example in merged:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(merged)} examples to {DATASET.relative_to(ROOT)}")
    if duplicates:
        print(f"Dropped {duplicates} duplicate questions")

    read_examples(DATASET)
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
