# Proposal: add-training-observability-and-fit-control

## Why

The original pipeline could train, merge, export and serve, but not tell you whether a run was any good. It printed only the first and last loss, and `save_strategy="no"` meant no `trainer_state.json`, so a finished run left no record beyond stdout.

That gap caused real failures. A 5-epoch run underfitted so badly the model answered from base-model priors (weights moved 0.75% in norm); a 20-epoch run memorised the corpus (token accuracy 0.947, entropy 2.22 to 0.55) and produced stitched fragments. Training loss fell smoothly in both, because it cannot distinguish learning from memorising.

Separately, the original change deferred real data curation to a follow-up. This is it — and expanding the corpus exposed a data defect no tuning fixes: every "can I use X water?" example named a *bad* water, so the model learned to answer the question shape rather than the question.


## What Changes

- Persist full per-step run metrics to `models/runs/<run>.json`, written after every logging step so a run in progress is observable and a finished run leaves a durable record.
- Add a held-out validation split (default 10%), scored each epoch, with the best epoch reported and a warning when holdout loss rises.
- Add an assistant-only loss option, with a bundled chat template carrying `{% generation %}` markers, since SmolLM2's stock template lacks them and TRL cannot auto-patch it.
- Add a metrics chart (self-contained HTML, plain SVG, no plotting library) and a local server that re-renders it live during training.
- Expose the epoch count and arbitrary extra flags as make variables so the common knobs do not require editing files.
- Build the training corpus reproducibly from a seed file plus topic batches, and grow it to ~900 examples with deliberate attention to question-position coverage.
- Make the export stage exit cleanly despite an ONNX Runtime teardown abort on macOS, and pass an explicit attention mask in the generation smoke checks.

## Capabilities

### New Capabilities

- `training-observability`: Record, chart and live-serve the metrics of a training run, so run quality can be judged and successive runs compared without an external tracking service.

### Modified Capabilities

- `fine-tuning`: Gains generalisation measurement (held-out split), a loss-masking option, make-level hyperparameter overrides, a reproducible dataset build, and a more robust export stage.

## Impact

- New files: `training/plot.py`, `training/metrics_serve.py`, `training/build_dataset.py`, `training/chat_template_assistant_mask.jinja`.
- Modified: `training/train.py`, `training/merge.py`, `training/export.py`, `training/config.py`, `Makefile`, `README.md`.
- New artifact directory `models/runs/` (gitignored), deliberately outside `models/checkpoints/`, which `train.py` wipes at the start of every run.
- New make targets `plot`, `metrics`, `dataset`; new make variables `EPOCHS` (default 15) and `TRAIN_ARGS`.
- No new required dependencies. `trackio` is optional and commented in `requirements.txt`; the chart and server use only the standard library.
- The default epoch count changed from 5 to 15, chosen to overshoot deliberately so the holdout curve reveals the real optimum.
