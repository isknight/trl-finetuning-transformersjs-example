# Design: add-training-observability-and-fit-control

## Context

The pipeline shipped by `add-finetune-and-web-inference` worked mechanically but was unmeasurable. Two runs demonstrated the cost: at 5 epochs the adapter moved the weights by 0.75% in norm and the model answered from base-model priors; at 20 epochs it reached 0.947 token accuracy with entropy collapsed to 0.55 and produced stitched fragments of memorised text. Training loss fell smoothly in both. The information needed to tell those apart was being computed and discarded.

## Decisions

### Metrics live outside the checkpoint directory

`train.py` calls `shutil.rmtree(CHECKPOINTS)` at startup, so anything stored there is destroyed by the next run — which is exactly when you want to compare against the previous one. Run history goes to `models/runs/` instead, one file per run, named from the hyperparameters so successive experiments do not collide.

`save_strategy="no"` means the trainer writes no `trainer_state.json`, so persistence is ours to do. A `TrainerCallback` flushes on every `on_log`. Writes go to a temp file and are renamed, because the metrics server may read the file mid-run and a torn read would break the page.

The callback must subclass `TrainerCallback` rather than duck-type it: the callback handler calls `getattr(callback, event)` for every event in the lifecycle, so an object defining only `on_log` raises `AttributeError` at `on_init_end`. It is constructed inside a factory so `transformers` is imported lazily and `--help` stays fast.

### The chart is generated SVG, not a plotting library

Adding matplotlib for four line charts is a poor trade in a teaching repository — it is a heavy dependency whose output is a bitmap. The chart is emitted as SVG from Python and wrapped in a single HTML file with inline CSS: no dependency, no build step, no network access, theme-aware, and it opens straight from disk. Raw per-step values are drawn faintly with an exponential moving average over them, because at `logging_steps=1` the raw curve is unreadable.

### Held-out split over any heuristic epoch count

There is no epoch number that is right across datasets, and the failure it guards against — memorisation — is invisible in training loss by construction. A 10% split scored each epoch makes the turning point directly observable, and the chart rings it. The default epoch count is deliberately set high enough to overshoot, so the exploratory run reveals the optimum and a second run trains to it.

The corollary is that the same run may be reported twice with different absolute loss values depending on whether assistant-only loss is on, since masking changes what is averaged. Comparisons across that flag should read the shape, not the number.

### Assistant-only loss needs a hand-written template

With a conversational dataset, TRL's `assistant_only_loss` defaults to off and the loss covers the whole sequence — in this corpus, 31% of tokens are system prompt and user question. Enabling the flag requires a chat template with `{% generation %}` markers. SmolLM2 ships none, and TRL's auto-patcher explicitly refuses this template ("patching is not supported for this template"), so the repository provides one.

It renders byte-identically to the stock template, verified across the corpus, so it alters only masking. The end-of-turn token sits inside the generation block; leaving it outside would train a model that never learns to stop. It reaches training only via `chat_template_path` — `merge.py` takes its tokenizer from the base model, so the merged and exported artifacts keep the stock template and the browser is unaffected.

### Make variables rather than a target per configuration

`EPOCHS ?= 15` plus a `TRAIN_ARGS` passthrough covers the common knob and the long tail with two lines. `?=` lets a command-line assignment, an environment variable, or the default all work, and both flow through the aggregate pipeline target. The training script's own default is kept in sync so a direct invocation behaves the same way.

### Exporting past a runtime that aborts at shutdown

ONNX Runtime deadlocks in its own teardown at interpreter exit on macOS, raising `recursive_mutex lock failed` *after* the export has completed and reported success. Because that aborts the process, the make target failed despite a good export. The stage flushes its output and exits without running interpreter teardown. This is narrow: it is applied only at the end of a successful export, so genuine failures still surface through the exit status.

## Risks / Trade-offs

- **Holding data out costs training data.** At ~900 examples, a 10% split removes ~90. That is a real cost, accepted because an unmeasurable run is worth less than a slightly smaller one. The split is configurable for anyone who disagrees.
- **The metrics server renders on every request.** Fine at this scale; a long run with many logged steps would eventually make each render slower. It re-reads from disk rather than caching, which keeps it correct while a run is in progress.
- **Bypassing interpreter teardown on export** skips other atexit handlers. Acceptable because it happens at the very end of a stage whose outputs are already written and verified.
- **A hand-maintained chat template can drift** from the base model's if the base model is ever changed. The mitigation is that it is verified to render identically to the stock template, so drift is detectable.
