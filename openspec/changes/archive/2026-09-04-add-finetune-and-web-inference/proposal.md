# Proposal: add-finetune-and-web-inference

## Why

We want a chatbot that answers questions about caring for carnivorous plants and runs entirely in the user's browser (no inference server, no per-query cost, works offline once cached). No pipeline exists yet in this project to produce such a model or serve it; this change builds both halves end to end so the full loop — train, export, chat in browser — works before we invest in a high-quality dataset.

## What Changes

- Add a PyTorch-based fine-tuning pipeline (Hugging Face `transformers` + `trl` + `peft`) that LoRA fine-tunes `HuggingFaceTB/SmolLM2-360M-Instruct` on a chat-formatted Q&A dataset about carnivorous plant care, then merges the adapter into a standalone model.
- Add a placeholder/seed dataset in chat format so the pipeline runs end to end today; real data curation is a follow-up change.
- Add an export step (`optimum` + `onnxruntime`) that converts the merged model to quantized ONNX suitable for Transformers.js.
- Add a static web app that loads the exported model with Transformers.js, using the WebGPU backend with WASM fallback, and provides a simple streaming chat UI.
- Add a `Makefile` exposing the whole workflow as simple targets (setup, train, export, serve, clean) using the project's existing `.venv`.

## Capabilities

### New Capabilities

- `fine-tuning`: Produce a fine-tuned, merged SmolLM2-360M model from a chat-formatted Q&A dataset using LoRA, runnable on local hardware (Apple Silicon MPS or CPU), including export to quantized ONNX for browser consumption.
- `web-inference`: Serve the exported model as a static site that runs chat inference fully client-side via Transformers.js (WebGPU preferred, WASM fallback) with streaming responses.

### Modified Capabilities

None — this is a greenfield project with no existing specs.

## Impact

- New Python dependencies installed into the existing `.venv`: `torch`, `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `optimum`, `onnx`, `onnxruntime`.
- New directories: `training/` (scripts + dataset), `web/` (static chat app), `models/` (gitignored artifacts: checkpoints, merged model, ONNX export).
- New `Makefile` at the repo root as the single entry point for all workflows.
- No existing code is affected (repo currently contains only `.venv` and IDE config).
- Model weights and training artifacts are large; they stay out of version control via `.gitignore`.
