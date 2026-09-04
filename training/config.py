"""Shared paths and model settings for the fine-tuning pipeline.

Each stage reads the previous stage's fixed output path, so the scripts chain
together without any configuration plumbing.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Merged training corpus, built from seed.jsonl plus data/generated/ by build_dataset.py
DATASET = ROOT / "training" / "data" / "plants.jsonl"
CHECKPOINTS = ROOT / "models" / "checkpoints"
MERGED = ROOT / "models" / "merged"
ONNX = ROOT / "models" / "onnx"
# Per-run loss histories. Kept outside CHECKPOINTS, which train.py wipes on every run.
RUNS = ROOT / "models" / "runs"

# SmolLM2's stock template has no {% generation %} markers, so TRL cannot mask the
# prompt. This copy adds them; it is used for training only - merge.py takes the
# tokenizer from BASE_MODEL, so the exported model keeps the stock template.
TRAIN_CHAT_TEMPLATE = ROOT / "training" / "chat_template_assistant_mask.jinja"

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions about caring "
    "for carnivorous plants."
)


def pick_device() -> str:
    """Prefer Apple Silicon GPU, fall back to CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
