# Tasks: add-training-observability-and-fit-control

## 1. Durable run metrics

- [x] 1.1 Add a `RUNS` path in `training/config.py` pointing at `models/runs/`, outside the checkpoint directory that training wipes
- [x] 1.2 Add a `TrainerCallback` in `training/train.py` that writes the run's hyperparameters and full `log_history` on every `on_log` and on `on_train_end`, via temp-file-and-rename so readers never see a partial file
- [x] 1.3 Verify the callback satisfies the callback handler contract (subclasses `TrainerCallback`, so every lifecycle event resolves) and that it writes on a simulated log event

## 2. Held-out validation

- [x] 2.1 Add `--eval-split` (default 0.1) splitting the dataset with a fixed seed, and pass the holdout to the trainer with per-epoch evaluation
- [x] 2.2 Report the best epoch by held-out loss on completion, and warn when the final value ends materially above it
- [x] 2.3 Record `holdout` size and `eval_split` in the run metadata

## 3. Assistant-only loss

- [x] 3.1 Confirm the stock SmolLM2 template lacks `{% generation %}` markers and that TRL's auto-patcher refuses it
- [x] 3.2 Write `training/chat_template_assistant_mask.jinja` adding the markers, with the end-of-turn token inside the generated span
- [x] 3.3 Verify it renders byte-identically to the stock template, produces a valid assistant mask over the whole corpus, and passes TRL's stop-token-trained check
- [x] 3.4 Wire `--assistant-loss` to `assistant_only_loss` plus `chat_template_path`; confirm the merged model still carries the stock template

## 4. Metrics chart

- [x] 4.1 Write `training/plot.py` rendering runs to a self-contained HTML file with generated SVG — no plotting library, no JavaScript, no network assets
- [x] 4.2 Draw one panel per metric family (loss, token accuracy, entropy, gradient norm) on a shared epoch axis, with raw values faint under a smoothed line
- [x] 4.3 Draw held-out series distinctly and ring the best value; summarise each run in the legend
- [x] 4.4 Support charting from a saved training stdout log as well as from run files
- [x] 4.5 Verify rendered geometry stays within the viewport and that a missing-runs invocation exits non-zero with guidance

## 5. Live metrics server

- [x] 5.1 Write `training/metrics_serve.py`: loopback-only HTTP server re-rendering the chart per request with a periodic refresh, on a different port from the chat app
- [x] 5.2 Handle no-runs-yet and unrenderable-run cases by serving an explanatory page rather than failing
- [x] 5.3 Exit non-zero with a clear message when the port is occupied
- [x] 5.4 Verify end to end: panels render, refresh directive present, unknown paths 404

## 6. Entry-point overrides

- [x] 6.1 Add `EPOCHS ?= 15` and `TRAIN_ARGS` to the Makefile and thread them through the train target
- [x] 6.2 Align the training script's own epoch default so direct invocation matches
- [x] 6.3 Verify override from command line, from environment, and through the aggregate pipeline target

## 7. Dataset build and expansion

- [x] 7.1 Add `training/build_dataset.py` merging seed and batch files, forcing the shared system prompt, dropping duplicate questions, and validating the output; expose it as `make dataset`
- [x] 7.2 Grow the corpus to ~900 examples across topic batches covering the genera, care areas, cultivars, propagation, seasonal care and symptom triage
- [x] 7.3 Audit question-position coverage; add corrective examples where correct practice appeared only in answers, including minimal pairs that differ by subject rather than question shape
- [x] 7.4 Audit care-topic coverage; fill growing-season temperature, light intensity, soil ratios and other quantitative questions

## 8. Export robustness

- [x] 8.1 Exit the export stage without interpreter teardown after a successful export, so the runtime's shutdown fault does not fail the target
- [x] 8.2 Pass an explicit attention mask in the merge and export generation smoke checks

## 9. Documentation

- [x] 9.1 Rewrite `README.md` for readers new to fine-tuning: concepts with links, every target and flag, how to read each metric, and the two failure modes with measured numbers
- [x] 9.2 Document the observed failures — underfitting, memorisation, and the question-position data defect — as worked examples
- [x] 9.3 Verify every documented flag, target and internal anchor resolves and every external link responds
