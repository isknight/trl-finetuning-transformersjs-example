# Tasks: add-finetune-and-web-inference

## 1. Project scaffolding

- [x] 1.1 Create `training/`, `training/data/`, `web/`, `models/` directories; add `.gitignore` covering `models/`, `.venv/`, `__pycache__/`, `.idea/`
- [x] 1.2 Add `requirements.txt` (torch, transformers, trl, peft, datasets, accelerate, optimum[onnxruntime], onnx, onnxruntime) and a `Makefile` with `setup` target installing into the existing `.venv` via `.venv/bin/pip`
- [x] 1.3 Run `make setup` and verify imports (`torch`, `trl`, `peft`, `optimum`) succeed in `.venv/bin/python`

## 2. Seed dataset

- [x] 2.1 Write `training/data/seed.jsonl`: ~50 carnivorous plant care Q&A examples in chat `messages` format (Venus flytrap, sundews, pitcher plants, butterworts; watering, light, soil, feeding, dormancy, propagation)
- [x] 2.2 Add dataset validation in a shared helper (`training/dataset.py`): load JSONL, verify each line has a `messages` array with valid roles/content, exit non-zero naming file and line on failure

## 3. Fine-tuning pipeline

- [x] 3.1 Write `training/train.py`: load SmolLM2-360M-Instruct, device pick (mps→cpu), LoRA config per design, TRL `SFTTrainer` on the validated dataset, save adapter to `models/checkpoints/`
- [x] 3.2 Run `make train` end to end on the seed dataset; confirm loss decreases and adapter files land in `models/checkpoints/`
- [x] 3.3 Write `training/merge.py`: load base + adapter, `merge_and_unload()`, save merged model + tokenizer to `models/merged/`; verify merged dir loads standalone and generates a coherent chat reply
- [x] 3.4 Write `training/eval.py`: run a fixed list of ~10 plant-care prompts through `models/merged/` and print prompt/answer pairs; run `make eval` and human-check output

## 4. ONNX export

- [x] 4.1 Write `training/export.py`: Optimum ONNX export of `models/merged/` on CPU, 4-bit weight quantization, output in Transformers.js layout under `models/onnx/`, then parity smoke check via onnxruntime asserting non-degenerate generation
- [x] 4.2 Run `make export`; verify layout (`onnx/model_quantized.onnx` + tokenizer/config files) and total quantized size under 500 MB

## 5. Web chat app

- [x] 5.1 Write `web/index.html` + `web/style.css`: chat transcript, input box (disabled until model ready), progress bar, backend indicator, Clear button
- [x] 5.2 Write `web/app.js`: import pinned `@huggingface/transformers`, feature-detect WebGPU vs WASM (with slow-mode notice), load model from served `models/onnx/` path with progress callbacks, streaming generation via `TextStreamer`, in-memory multi-turn message history with truncation near context limit, Clear resets
- [x] 5.3 Write `web/serve.py`: stdlib HTTP server with COOP/COEP headers, serving `web/` and mounting `models/onnx/`; exit with "run `make export` first" if the export is missing

## 6. Makefile integration and verification

- [x] 6.1 Finish Makefile targets `train`, `merge`, `export`, `eval`, `serve`, `clean` (prompting before delete); each target checks its prerequisite stage's output exists and names the missing target in its error
- [x] 6.2 Full-loop verification: `make clean && make train merge export serve`, then in a WebGPU browser confirm progress bar, GPU indicator, streamed answer to a flytrap question, working follow-up turn, and no inference network requests after load; also confirm `make serve` without export and `make train` without dataset fail with the specified messages
- [x] 6.3 Write `README.md`: prerequisites, make targets, pipeline overview, how to swap in a real dataset
