# Fine-tune SmolLM2 on carnivorous plant care Q&A, serve it in the browser.
# All Python runs through the project venv; no shell activation needed.

PY := .venv/bin/python
PIP := .venv/bin/pip

DATASET := training/data/plants.jsonl
CHECKPOINTS := models/checkpoints
MERGED := models/merged
ONNX := models/onnx

# Ops missing from the MPS backend fall back to CPU instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK := 1

# Training knobs. `?=` means the command line wins: make train EPOCHS=10
# 15 epochs is an exploratory default - with the holdout split on, the run tells you
# which epoch was actually best, and you retrain at that number.
EPOCHS ?= 15
# Anything else goes straight through: make train TRAIN_ARGS="--lora-r 32 --assistant-loss"
TRAIN_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup dataset data train merge export eval test plot metrics serve all clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

$(PY):
	@echo "ERROR: no virtualenv at .venv - create one with: python3 -m venv .venv" >&2
	@exit 1

setup: $(PY) ## Install Python dependencies into .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete."

dataset: $(PY) ## Rebuild the training corpus from seed + generated batches
	$(PY) -m training.build_dataset

data: $(PY) ## Validate the training dataset
	$(PY) -m training.dataset $(DATASET)

train: $(PY) ## LoRA fine-tune the base model (EPOCHS=15; TRAIN_ARGS="..." for other flags)
	@test -f $(DATASET) || { echo "ERROR: $(DATASET) not found - add training data first." >&2; exit 1; }
	$(PY) -m training.train --epochs $(EPOCHS) $(TRAIN_ARGS)

merge: $(PY) ## Merge the adapter into the base weights (writes models/merged)
	@test -f $(CHECKPOINTS)/adapter_config.json \
		|| { echo "ERROR: no adapter in $(CHECKPOINTS) - run 'make train' first." >&2; exit 1; }
	$(PY) -m training.merge

export: $(PY) ## Export the merged model to quantized ONNX (writes models/onnx)
	@test -f $(MERGED)/config.json \
		|| { echo "ERROR: no merged model in $(MERGED) - run 'make merge' first." >&2; exit 1; }
	$(PY) -m training.export

test: $(PY) ## Run the test suite (no GPU, no model download, no network)
	$(PY) -m pytest tests/ -q

plot: $(PY) ## Render training metrics to models/runs/metrics.html
	$(PY) -m training.plot

metrics: $(PY) ## Serve live training metrics on http://127.0.0.1:8001
	$(PY) -m training.metrics_serve

eval: $(PY) ## Print answers to a fixed prompt set for human review
	@test -f $(MERGED)/config.json \
		|| { echo "ERROR: no merged model in $(MERGED) - run 'make merge' first." >&2; exit 1; }
	$(PY) -m training.eval

serve: $(PY) ## Serve the browser chat app on http://127.0.0.1:8000
	@test -d $(ONNX)/onnx \
		|| { echo "ERROR: no exported model in $(ONNX) - run 'make export' first." >&2; exit 1; }
	$(PY) web/serve.py

all: train merge export ## Run the full pipeline: train, merge, export

clean: ## Delete all model artifacts in models/
	@test -d models || { echo "Nothing to clean."; exit 0; }
	@printf "Delete everything under models/? [y/N] " && read ans && [ "$$ans" = "y" ] \
		|| { echo "Aborted."; exit 1; }
	rm -rf models/checkpoints models/merged models/onnx models/runs
	@echo "Model artifacts removed."
