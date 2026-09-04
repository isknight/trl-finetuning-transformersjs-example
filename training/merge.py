"""Merge the trained LoRA adapter into the base weights.

Produces a standalone model directory that loads as a plain causal LM, with no
adapter machinery involved downstream. Runs on CPU in fp32: merging is a one-off
weight operation and the ONNX export that follows is CPU-only anyway.
"""

import shutil
import sys

from training.config import BASE_MODEL, CHECKPOINTS, MERGED, SYSTEM_PROMPT

SMOKE_PROMPT = "What kind of water should I use for my Venus flytrap?"


def main() -> int:
    if not (CHECKPOINTS / "adapter_config.json").exists():
        print(f"ERROR: no adapter found in {CHECKPOINTS}", file=sys.stderr)
        print("Run `make train` first.", file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Applying adapter from {CHECKPOINTS}")
    model = PeftModel.from_pretrained(base, str(CHECKPOINTS))
    model = model.merge_and_unload()

    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED))
    tokenizer.save_pretrained(str(MERGED))
    print(f"Merged model saved to {MERGED}")

    print("\nSmoke check - reloading merged model standalone:")
    check_model = AutoModelForCausalLM.from_pretrained(str(MERGED), dtype=torch.float32)
    check_tokenizer = AutoTokenizer.from_pretrained(str(MERGED))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SMOKE_PROMPT},
    ]
    # return_dict gives us the attention mask too; without it generate() warns and
    # falls back to guessing, because pad and eos are the same token here.
    inputs = check_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    outputs = check_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=check_tokenizer.eos_token_id,
    )
    reply = check_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"  Q: {SMOKE_PROMPT}")
    print(f"  A: {reply.strip()}")

    if not reply.strip():
        print("ERROR: merged model produced an empty reply.", file=sys.stderr)
        return 1

    print("\nNext: make export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
