# Carnivorous Plant Care Bot

A complete, small fine-tuning project you can run end to end on a laptop: take an open base
model, teach it a narrow domain, and serve it **in the browser** with no inference server and
no per-query cost.

It is built as a worked example. Every stage is a short, readable Python file, the whole
pipeline runs from `make`, and this README explains not just the commands but what each step
is actually doing and how to tell whether it worked.

**What you end up with:** a chat page where you ask about carnivorous plants and a 360M
parameter model answers, running entirely on your own GPU via WebGPU. Prompts never leave the
browser.

---

## Table of contents

- [How it works](#how-it-works) - the pipeline and the concepts behind it
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Every make target](#every-make-target)
- [The dataset](#the-dataset)
- [Reading the training metrics](#reading-the-training-metrics) - the part most tutorials skip
- [Choosing hyperparameters](#choosing-hyperparameters)
- [Lessons this repo learned the hard way](#lessons-this-repo-learned-the-hard-way)
- [Troubleshooting](#troubleshooting)
- [Layout](#layout)
- [Further reading](#further-reading)

---

## How it works

```
plants.jsonl  ->  make train  ->  make merge  ->  make export  ->  make serve
  chat Q&A        LoRA adapter    standalone      quantized       browser chat
                  (35 MB)         model           ONNX            on WebGPU
```

Each stage writes to a fixed path under `models/` and the next stage reads it, so you can
re-run any stage on its own.

| Stage | Output | What happens |
| --- | --- | --- |
| `train` | `models/checkpoints/` | LoRA adapter only; base weights stay frozen |
| `merge` | `models/merged/` | Adapter folded into the weights; loads as a plain causal LM |
| `export` | `models/onnx/` | Quantized ONNX in the layout Transformers.js expects |
| `serve` | - | Static app on `http://127.0.0.1:8000` |

### The base model

[SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) - 360
million parameters, already instruction-tuned so it can hold a conversation. It is small
enough to fine-tune on a laptop in minutes and small enough to ship to a browser. That size is
also its limit; see [expectations](#a-realistic-expectation).

### LoRA: why we don't train all 362M parameters

Ordinary fine-tuning updates every weight, which means storing gradients and optimizer state
for all of them - roughly 4x the model in memory - and producing a full 1.4 GB copy for every
experiment.

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) freezes the base model and
learns two small matrices alongside each targeted layer. Instead of learning a full update
`ΔW`, it learns `A` and `B` whose product approximates it, so the layer computes
`W·x + B·A·x`. Only `A` and `B` receive gradients.

At this project's settings (`r=16`, all seven projections, 32 layers) that is **8.68M
trainable parameters - 2.4% of the model**, and an adapter file of 35 MB instead of 1.4 GB.
The trade-off is capacity: a rank-16 adapter can only express so much change. It is
implemented by [PEFT](https://huggingface.co/docs/peft).

Because the adapter is a *diff*, not a model, `make merge` computes `W + (alpha/r)·B·A` and
writes the result back into the real weights. That is mathematically exact, costs nothing at
inference, and matters here because ONNX has no concept of an adapter.

### SFTTrainer: the training loop

[TRL's](https://huggingface.co/docs/trl/sft_trainer) `SFTTrainer` wraps the standard
`transformers` Trainer and handles the fiddly parts of instruction tuning: applying the
[chat template](https://huggingface.co/docs/transformers/chat_templating) to your `messages`
arrays, tokenizing, batching, and optionally masking the loss so only the assistant's replies
count. You give it a dataset and a `peft_config`; it applies the adapter for you.

### Getting it into a browser

Browsers cannot run PyTorch. The model is exported to **ONNX**, a portable graph format, via
[Optimum](https://huggingface.co/docs/optimum/exporters/onnx/overview), then **quantized** -
weights stored at 4 or 8 bits instead of 32 - which is what makes the download feasible:

| Graph | Size | Download (with tokenizer etc.) |
| --- | --- | --- |
| `model_q4.onnx` | 369 MB | 374 MB |
| `model_quantized.onnx` (8-bit) | 482 MB | 487 MB |

[Transformers.js](https://huggingface.co/docs/transformers.js) then runs that graph through
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), which uses
[WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) when available and falls
back to WASM on CPU. This is *not* TensorFlow.js, and WebGL is not involved - WebGPU is the
only viable browser backend for LLMs today.

---

## Prerequisites

- **Python 3.11** in a virtualenv at `.venv` (already present here; otherwise
  `python3 -m venv .venv`)
- **~5 GB free disk** for the base model, checkpoints and ONNX exports
- **A WebGPU-capable browser** for fast inference - recent Chrome, Edge, Firefox or Safari 26+.
  Without WebGPU the app still works on CPU via WASM, just slower.

Training uses the Apple Silicon GPU (MPS) when available and falls back to CPU. It runs in
fp32; bf16 support on MPS is still uneven and this model is small enough that memory is not the
constraint. The ONNX export always runs on CPU.

Rough timing on an M-series MacBook: **about 8 minutes per 5 epochs** on the 901-example
corpus.

---

## Quick start

```bash
make setup     # install dependencies into .venv
make all       # train, merge and export in one go
make serve     # then open http://127.0.0.1:8000
```

To watch training as it happens, open a second terminal:

```bash
make metrics   # live dashboard on http://127.0.0.1:8001
```

---

## Every make target

| Target | What it does |
| --- | --- |
| `make setup` | Install Python dependencies into `.venv` |
| `make dataset` | Rebuild `plants.jsonl` from the seed and generated batches |
| `make data` | Validate the dataset without training |
| `make train` | LoRA fine-tune, writing the adapter and run metrics (see overrides below) |
| `make merge` | Merge the adapter into the base weights, with a generation smoke check |
| `make eval` | Print answers to 10 fixed prompts for human review |
| `make test` | Run the test suite (seconds; no GPU, model download or network) |
| `make plot` | Render metrics to `models/runs/metrics.html` |
| `make metrics` | Serve those metrics live on `:8001`, refreshing every 5s |
| `make export` | Quantized ONNX export, with a parity smoke check |
| `make serve` | Serve the browser chat app on `:8000` |
| `make all` | `train` + `merge` + `export` |
| `make clean` | Delete everything under `models/` (prompts first) |

Each stage checks that the previous one ran and gives you a clear error if not.

### Training options

Override the epoch count straight from make:

```bash
make train EPOCHS=10
```

Anything else goes through `TRAIN_ARGS`:

```bash
make train EPOCHS=12 TRAIN_ARGS="--assistant-loss --lora-r 32"
```

Both work as environment variables too (`EPOCHS=10 make train`) and both flow through
`make all`. Or call the module directly if you prefer:

```bash
.venv/bin/python -m training.train --epochs 10 --assistant-loss
```

| Flag | Default | What it is for |
| --- | --- | --- |
| `--epochs` | 15 | How many passes over the data. **The most important knob** - see below |
| `--lr` | 2e-4 | Learning rate, on a cosine schedule with 10% warmup |
| `--lora-r` | 16 | Adapter rank. Raise it if the model plateaus while still underfitting |
| `--lora-alpha` | 32 | Adapter scaling; effective strength is `alpha/r` |
| `--eval-split` | 0.1 | Fraction held out to measure generalisation. **Leave this on** |
| `--assistant-loss` | off | Train only on assistant replies, masking the prompt out of the loss |
| `--batch-size` | 4 | Per-device batch; effective batch is this x `--grad-accum` |
| `--run-name` | auto | Names the metrics file; defaults to the hyperparameters |
| `--track` | off | Also stream to a local Trackio dashboard (needs `pip install trackio`) |

**`--assistant-loss` is worth understanding.** By default the loss covers the *entire*
sequence, so the model is also trained to reproduce your system prompt and invent user
questions. In this corpus that is **31% of the training signal spent on tokens the model never
needs to generate**. The flag masks them out. It needs a chat template with `{% generation %}`
markers, which SmolLM2 does not ship and TRL cannot auto-patch, so this repo includes one at
`training/chat_template_assistant_mask.jinja`. It is used for training only - `merge.py` takes
its tokenizer from the base model, so the exported model keeps the stock template.

---

## The dataset

`training/data/plants.jsonl` is the training corpus - **901 examples**, built by `make dataset`
from two sources:

- `training/data/seed.jsonl` - 55 hand-written examples
- `training/data/generated/batch_*.jsonl` - 17 topic batches covering water and soil, light and
  dormancy, temperature, feeding, the individual genera, cultivars, pests and disease,
  propagation, flowering and seed, seasonal care, symptom triage, and equipment

`make dataset` merges them, forces the shared system prompt onto every example, drops duplicate
questions and validates the result. Edit the sources and re-run it rather than editing
`plants.jsonl`, which is regenerated from scratch each time.

### Format

One JSON object per line, with a `messages` array - the same conversational format the
[chat template](https://huggingface.co/docs/transformers/chat_templating) expects:

```json
{"messages": [
  {"role": "system", "content": "You are a knowledgeable assistant..."},
  {"role": "user", "content": "How often should I water my Venus flytrap?"},
  {"role": "assistant", "content": "Keep the soil damp at all times..."}
]}
```

### Using your own data

```bash
.venv/bin/python -m training.dataset path/to/yours.jsonl   # validate first
.venv/bin/python -m training.train --dataset path/to/yours.jsonl
```

The validator names the offending file and line number, which is worth running before you
spend compute on a bad file.

---

## Reading the training metrics

This is the part most tutorials skip, and it is where you will actually spend your time.

`make metrics` serves four panels sharing an epoch axis. Training writes its history after
every logging step, so the page fills in live.

### Loss

How wrong the model's next-token predictions are. Falling is good, but **training loss alone
cannot tell you whether the model is learning or memorising** - it falls in both cases. That is
why the holdout exists.

### Holdout loss (the dashed line)

`--eval-split 0.1` keeps 10% of examples out of training and scores them each epoch. This is
the number that matters. While it falls, the model is genuinely generalising. **When it turns
upward while training loss keeps falling, you have started memorising** - that crossover is
your epoch budget, and the chart rings the minimum for you. `train.py` also prints the best
epoch and warns if you overshot.

### Token accuracy

The fraction of next tokens predicted exactly. This is the most legible overfitting detector
you have, because it needs no calibration: **above roughly 0.9 on training data means the model
has memorised the corpus.** Loss of 0.278 says the same thing, but you have to know what 0.278
means.

### Entropy

The confidence of the output distribution, in nats. A healthy run declines gently. A
**collapse toward zero means the model commits hard to one continuation with no hedging** -
which is what produces confident nonsense, as opposed to the vague waffle you get from an
undertrained model.

### Gradient norm

The size of the update signal, reported *before* clipping. `max_grad_norm` defaults to `1.0`,
so **sustained values above 1.0 mean nearly every step is being clipped** and your effective
learning rate is lower than you set. Occasional spikes are normal; a permanently pinned value
is not.

Note that with a cosine schedule the learning rate decays to zero, so learning can stop dead
while gradient norm still looks healthy. If a run ends with a strong gradient norm and a loss
curve that never flattened, it ran out of schedule rather than converging.

---

## Choosing hyperparameters

The honest workflow is: **one exploratory run with the holdout on, then one real run.**

```bash
make train TRAIN_ARGS="--assistant-loss"          # explore at the default 15 epochs
make plot                                         # read the ring on the holdout curve
make train EPOCHS=<best> TRAIN_ARGS="--assistant-loss"
make merge && make export
```

Runs are named from their hyperparameters, so `make plot` overlays them all on one chart and
you can compare directly.

### The two failure modes, with real numbers from this repo

**Underfitting** - 5 epochs on 552 examples:

```
final loss 1.69, still falling 0.065/epoch, never flattened
weights moved 0.75% (‖ΔW‖/‖W‖)
```

The model produced fluent, confident, *wrong* answers drawn from the base model's general
houseplant knowledge - recommending fertiliser and softened water, denying dormancy. A model
perturbed by three-quarters of one percent is still essentially the base model.

**Overfitting** - 20 epochs on 852 examples:

```
final loss 0.278, token accuracy 0.947, entropy collapsed 2.22 -> 0.55
```

The model produced incoherent stitched fragments: *"Anything other than distilled, tap, or
filtered water will give the soil a gritty, mineral-free surface. The label will listPP, RO
water..."* Those are all memorised corpus phrases, reassembled without the logic that connected
them.

The useful epoch count is somewhere between, and the holdout curve is how you find it rather
than guess.

---

## Lessons this repo learned the hard way

### More data does not automatically mean a better model

Going from 55 to 552 examples made the model measurably worse, because the extra data was not
matched by extra training. Diversity raises the amount the model must absorb; if the epoch
budget stays the same, you get a worse fit, not a better one.

### The *shape* of your questions matters as much as your facts

The corpus was factually clean and still taught the model something false. Every
"can I use X water?" question in it named a *bad* water - tap, spring, filtered - so the model
learned the shortcut `"can I use <water>?" -> "No, that is not safe"` and applied it to
distilled water, which the corpus recommends 115 times in *answers* but never as the subject of
a question.

The general trap: **if your corpus only ever puts problems in question position, the model has
no template for a question about a good thing.** Check this in your own data:

```bash
.venv/bin/python - training/data/plants.jsonl distilled rainwater "full sun" <<'PY'
import json, sys
rows = [json.loads(l)["messages"] for l in open(sys.argv[1])]
qa = [(m[1]["content"].lower(), m[2]["content"].lower()) for m in rows]
for term in sys.argv[2:]:
    in_q = sum(1 for question, _ in qa if term in question)
    in_a = sum(1 for _, answer in qa if term in answer)
    print(f"{term:<12} in questions {in_q:>3} | in answers {in_a:>3}")
PY
```

A term that appears constantly in answers and almost never in questions is a blind spot.

The fix is not more data, it is *minimal pairs* - the same question shape with the opposite
answer, differing only in the noun:

```
[NO ] Can I use tap water?
[YES] Can I use distilled water?
```

`batch_16.jsonl` is the corrective batch built for exactly this, and it brought the corpus from
44 NO / 27 YES to 70 NO / 77 YES across yes/no questions.

### A realistic expectation

Fine-tuning a 360M model teaches it **style and format far more reliably than it teaches it
facts**. Expect it to sound like a plant care guide while still inventing details about species
it half-knows, even with a few hundred examples and a well-chosen epoch count. If factual
accuracy is what you need, the next step is retrieval over real care guides, not training
harder. That limit is a property of the model size, not of your dataset.

---

## Troubleshooting

**The browser still serves the old model after re-exporting.** Transformers.js caches model
files in the browser's Cache API, keyed by URL, and a re-export writes to the same paths. Clear
it from the page's devtools console:

```js
await caches.delete('transformers-cache'); location.reload();
```

If the tokenizer config changed, the symptom is errors that make no sense against the new files.

**`make export` ends with `Abort trap: 6`.** If you see `Export complete` and `Next: make serve`
first, the export succeeded - that abort is ONNX Runtime's teardown deadlocking at interpreter
shutdown on macOS. `export.py` now flushes and exits before that runs, so it should no longer
fail the target.

**Export prints `values not close enough, max diff: 0.0009`.** Ordinary fp32 tracing divergence
against a strict `1e-5` tolerance. Not a problem, and unrelated to answer quality.

**`make merge` fails with "no adapter".** `train.py` wipes `models/checkpoints/` when it starts,
so an interrupted run leaves it empty. Re-run `make train`.

**Answers are fluent but wrong** - underfitting. Train longer; check the loss curve actually
flattened.

**Answers are garbled or self-contradictory** - overfitting. Check token accuracy and the
holdout curve; train for fewer epochs.

---

## Layout

```
training/
  config.py          shared paths and the system prompt
  build_dataset.py   merges seed + batches into plants.jsonl
  dataset.py         format validation, usable standalone
  train.py           LoRA fine-tune; writes adapter + run metrics
  merge.py           folds the adapter into the base weights
  eval.py            10 fixed prompts for human review
  export.py          ONNX export + 4-bit and 8-bit quantization
  plot.py            renders metrics to a self-contained HTML chart
  metrics_serve.py   serves that chart live during training
  data/              seed.jsonl, generated/, plants.jsonl
web/
  index.html app.js style.css    the chat UI
  serve.py                       static server with the COOP/COEP headers
                                 multithreaded WASM requires
models/                          gitignored stage outputs
openspec/                        change specs and design docs
```

The chart is plain SVG generated in Python - no plotting library, no JavaScript, no tracking
service.

---

## Further reading

- [PEFT documentation](https://huggingface.co/docs/peft) - LoRA and the other adapter methods
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - the
  original paper, very readable
- [TRL `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) - the training loop and its
  many options
- [Chat templating](https://huggingface.co/docs/transformers/chat_templating) - how `messages`
  becomes tokens, and why `{% generation %}` matters
- [Transformers.js](https://huggingface.co/docs/transformers.js) - running models in the browser
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) - the engine underneath
- [Optimum ONNX export](https://huggingface.co/docs/optimum/exporters/onnx/overview) - how the
  graph is produced
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) - the base
  model
- [WebGPU browser support](https://caniuse.com/webgpu) - check whether your target audience
  has it
