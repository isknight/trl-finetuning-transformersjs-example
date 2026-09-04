"""Smoke-evaluate the merged model on a fixed set of plant care prompts.

Prints each prompt with the model's answer so a human can judge whether
fine-tuning improved domain behavior. Quality judgement stays human; this exits
non-zero only when execution fails. Pass --base to compare against the
un-finetuned base model.
"""

import argparse
import sys

from training.config import BASE_MODEL, MERGED, SYSTEM_PROMPT, pick_device

PROMPTS = [
    "How often should I water my Venus flytrap?",
    "Can I use tap water on my pitcher plant?",
    "What soil mix do carnivorous plants need?",
    "Does my Venus flytrap need a winter dormancy?",
    "Why are my flytrap's traps turning black?",
    "How do I care for a Cape sundew?",
    "Why isn't my Nepenthes making pitchers?",
    "Should I fertilize my carnivorous plants?",
    "What's the best carnivorous plant for a beginner?",
    "Can I feed my flytrap hamburger?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        action="store_true",
        help="evaluate the un-finetuned base model instead (for comparison)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    if args.base:
        model_path = BASE_MODEL
    else:
        if not (MERGED / "config.json").exists():
            print(f"ERROR: no merged model in {MERGED}", file=sys.stderr)
            print("Run `make train` then `make merge` first.", file=sys.stderr)
            return 1
        model_path = str(MERGED)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    print(f"Evaluating {model_path} on {device}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(device)
    model.eval()

    for i, prompt in enumerate(PROMPTS, start=1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"[{i}] Q: {prompt}")
        print(f"    A: {reply.strip()}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
