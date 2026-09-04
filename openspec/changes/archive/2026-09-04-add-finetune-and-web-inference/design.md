# Design: add-finetune-and-web-inference

## Context

Greenfield repo containing only a Python 3 virtual environment at `.venv` and IDE config. Development machine is an Apple Silicon Mac (torch MPS backend, no CUDA). The runtime and stack were chosen in prior discussion: SmolLM2-360M-Instruct as the base model, PyTorch/TRL/PEFT for training, Transformers.js (ONNX Runtime Web) for browser inference with WebGPU preferred and WASM fallback. Model artifacts are hundreds of MB and must stay out of git. See `proposal.md` for motivation.

## Goals / Non-Goals

**Goals:**

- A pipeline where each stage is a separate, resumable step (train → merge → export → serve), driven by a Makefile so a single `make <target>` runs any stage.
- The whole loop works today with a small seed dataset; swapping in a better dataset later requires no code changes.
- Everything runs on the local Mac; no cloud dependency.

**Non-Goals:**

- Dataset curation/synthesis at quality (seed data is placeholder; a follow-up change).
- Retrieval-augmented generation in the browser (possible later; the chat app is structured so a retrieval step could be inserted before generation).
- Production hosting/CDN of the web app; local serving only.
- Hyperparameter search or formal eval metrics; a human-judged smoke eval is the quality gate.

## Decisions

**Layout.** `training/` holds Python scripts (`train.py`, `merge.py`, `export.py`, `eval.py`) and `training/data/seed.jsonl`; `web/` holds the static app (`index.html`, `app.js`, `style.css`); `models/` (gitignored) holds `checkpoints/`, `merged/`, and `onnx/` stage outputs. Each script reads its input from the previous stage's fixed output path — no config plumbing.

**Dataset format: JSONL with a `messages` array** per line (OpenAI-style chat messages). It's the native input format for TRL's `SFTTrainer`, which applies the model's chat template automatically. Alternative (prompt/completion columns) rejected: chat format survives a move to multi-turn training data unchanged.

**Training: TRL `SFTTrainer` + PEFT LoRA** (r=16, alpha=32, targeting attention + MLP projections), base weights frozen, fp32 on MPS. Why LoRA over full fine-tune at a size where full FT is affordable: less catastrophic forgetting of general language ability, faster iteration, and merge erases the difference downstream. Why fp32: bf16 on MPS is still patchy; the model is small enough that memory isn't the constraint. Device selection is `mps` if available else `cpu`, per the spec's no-GPU scenario.

**Merge as a separate step** (`merge.py` loads base + adapter, `merge_and_unload()`, saves to `models/merged/` with tokenizer and chat template). Keeping it out of `train.py` lets us re-merge or A/B adapters without retraining.

**Export: Optimum → ONNX, then 4-bit weight quantization** via Optimum's ONNX Runtime quantization tooling, into `models/onnx/` with the Transformers.js layout (`onnx/model_quantized.onnx` beside `config.json`, `tokenizer.json`, etc.). Quantized 360M lands around 250–350 MB, within the spec's 500 MB budget. Export runs on CPU (ONNX export from MPS is not supported); that's fine, export is a one-shot conversion. `export.py` ends with a parity smoke check: run one prompt through the ONNX model with `onnxruntime` and assert non-degenerate output.

**Web app: plain static files, no build step.** `index.html` + `app.js` importing `@huggingface/transformers` as an ES module from a pinned CDN URL. No bundler because the app is one page and a build toolchain would dwarf it. The app feature-detects `navigator.gpu` to pick the WebGPU device, else WASM with a visible slow-mode notice; uses Transformers.js `TextStreamer` for token streaming and its progress callbacks for the download bar; keeps the message array in memory and replays it each turn for multi-turn context.

**Serving: small Python script (`web/serve.py`) over stdlib `http.server`**, not `python -m http.server` directly, because the app needs `Cross-Origin-Opener-Policy`/`Cross-Origin-Embedder-Policy` headers for multithreaded WASM, and the model directory (`models/onnx/`) must be mounted alongside `web/`. The script checks the export exists and exits with guidance if not, per spec.

**Makefile as the only entry point.** Targets: `setup` (pip install into the existing `.venv` — never create a new one), `train`, `merge`, `export`, `eval`, `serve`, `clean` (deletes `models/`, prompting first). Each pipeline target checks its input stage exists and names the missing prerequisite target in its error. Python targets invoke `.venv/bin/python` directly so the Makefile works without shell activation.

## Risks / Trade-offs

- [MPS training instability (op gaps, silent NaNs)] → fp32 training, small model, and the eval step catches garbage output early; CPU fallback is a working escape hatch, and the scripts are hardware-agnostic so a Colab run is a copy-paste away.
- [Seed dataset too small — model may parrot or overfit] → acceptable for pipeline validation; low epochs + LoRA limit damage; quality data is explicitly a follow-up change.
- [Quantization degrades an already-small model] → parity smoke check in export; if q4 output is degenerate, fall back to q8 (larger but still under budget).
- [CDN import of Transformers.js contradicts "fully client-side" purity] → prompt/completions still never leave the browser (spec's actual requirement); pinned version for reproducibility; vendoring the library locally is a trivial later hardening step.
- [Chat context grows past the model's window in long sessions] → truncate oldest turns when near the context limit; Clear button resets.

## Open Questions

- Final LoRA hyperparameters and epoch count — tune once real data lands; defaults are fine for the seed set.
- Whether to add client-side retrieval (embedded care guides) — deferred until the fine-tuned model's factual accuracy is assessed.
