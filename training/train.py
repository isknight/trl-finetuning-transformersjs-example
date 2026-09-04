"""LoRA fine-tune SmolLM2-360M-Instruct on the carnivorous plant care dataset.

Base weights stay frozen; only the adapter is trained and saved. Runs on Apple
Silicon (MPS) when available, otherwise CPU, in fp32 - bf16 support on MPS is
still uneven and the model is small enough that memory is not the constraint.
"""

import argparse
import json
import shutil
import sys

from training.config import (BASE_MODEL, CHECKPOINTS, DATASET, RUNS,
                            TRAIN_CHAT_TEMPLATE, pick_device)
from training.dataset import DatasetError, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET), help="JSONL training data")
    # 15 is an exploratory default: long enough to overshoot, so the holdout curve
    # shows you where the best epoch actually was. Retrain at that number.
    parser.add_argument("--epochs", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Fraction held out to measure generalisation; 0 disables")
    parser.add_argument("--assistant-loss", action="store_true",
                        help="Train on assistant replies only, masking the prompt out of the loss")
    parser.add_argument("--run-name", default=None,
                        help="Name for the saved loss history; defaults to the hyperparameters")
    parser.add_argument("--track", action="store_true",
                        help="Stream metrics to a local Trackio dashboard as well")
    return parser.parse_args()


def make_history_writer(path, meta: dict):
    """TrainerCallback that flushes run metrics after every log, so `make metrics` is live.

    Must subclass TrainerCallback rather than duck-type it: the callback handler does
    getattr(callback, event) for every event, so a partial class raises on on_init_end.
    Imported lazily here to keep `--help` from pulling in transformers.

    The write is atomic - temp file then rename - because the metrics server may read
    the file mid-run.
    """
    from transformers import TrainerCallback

    class HistoryWriter(TrainerCallback):
        def _flush(self, state):
            payload = dict(meta, log_history=state.log_history)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(path)

        def on_log(self, args, state, control, **kwargs):
            self._flush(state)

        def on_train_end(self, args, state, control, **kwargs):
            self._flush(state)

    return HistoryWriter()


def run_name(args) -> str:
    """Default run name that makes the interesting hyperparameters visible in the legend."""
    if args.run_name:
        return args.run_name
    suffix = "-asstloss" if args.assistant_loss else ""
    return f"e{args.epochs:g}-lr{args.lr:g}-r{args.lora_r}{suffix}"


def main() -> int:
    args = parse_args()

    from pathlib import Path

    try:
        dataset = load_dataset(Path(args.dataset))
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Fix the dataset before training.", file=sys.stderr)
        return 1

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    device = pick_device()

    # Train loss alone cannot distinguish learning from memorising, so hold a slice
    # back and score it each epoch. Where the two curves separate is the epoch budget.
    eval_dataset = None
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        dataset, eval_dataset = split["train"], split["test"]

    held = f" | holdout={len(eval_dataset)}" if eval_dataset else ""
    print(f"Training on {len(dataset)} examples{held} | base={BASE_MODEL} | device={device}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    if CHECKPOINTS.exists():
        shutil.rmtree(CHECKPOINTS)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to=["trackio"] if args.track else [],
        run_name=run_name(args),
        bf16=False,
        fp16=False,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        per_device_eval_batch_size=args.batch_size,
        assistant_only_loss=args.assistant_loss,
        chat_template_path=str(TRAIN_CHAT_TEMPLATE) if args.assistant_loss else None,
        use_cpu=(device == "cpu"),
        seed=42,
    )

    # The trainer wipes CHECKPOINTS on every run and save_strategy="no" writes no
    # trainer_state.json, so persist the run history somewhere it survives.
    RUNS.mkdir(parents=True, exist_ok=True)
    history_path = RUNS / f"{run_name(args)}.json"
    meta = {
        "run": run_name(args),
        "dataset": args.dataset,
        "examples": len(dataset),
        "holdout": len(eval_dataset) if eval_dataset is not None else 0,
        "base_model": BASE_MODEL,
        "device": device,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "assistant_loss": args.assistant_loss,
            "eval_split": args.eval_split,
        },
    }

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.add_callback(make_history_writer(history_path, meta))
    print(f"Metrics streaming to {history_path} - watch live with 'make metrics'")
    trainer.train()
    trainer.save_model(str(CHECKPOINTS))

    print(f"Metrics saved to {history_path} - chart them with 'make plot'")

    evals = [(log["epoch"], log["eval_loss"])
             for log in trainer.state.log_history if "eval_loss" in log]
    if evals:
        best_epoch, best_loss = min(evals, key=lambda x: x[1])
        print(f"Holdout loss: best={best_loss:.4f} at epoch {best_epoch:g} "
              f"| final={evals[-1][1]:.4f}")
        if evals[-1][1] > best_loss * 1.05:
            print(f"WARNING: holdout loss rose after epoch {best_epoch:g} - "
                  f"the model is memorising. Train for ~{best_epoch:g} epochs instead.",
                  file=sys.stderr)

    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    if losses:
        print(f"Loss: first={losses[0]:.4f} last={losses[-1]:.4f} steps={len(losses)}")
        if losses[-1] >= losses[0]:
            print("WARNING: loss did not decrease - check hyperparameters.", file=sys.stderr)

    print(f"Adapter saved to {CHECKPOINTS}")
    print("Next: make merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
