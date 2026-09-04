# fine-tuning Specification

## Purpose
Turn a chat-formatted Q&A dataset about carnivorous plant care into a fine-tuned small language model, exported in a form the browser inference capability can serve.
## Requirements
### Requirement: Dataset in chat format

The system SHALL read training data from a project-local dataset file of Q&A examples in chat-message format (each example holding user and assistant turns), and SHALL fail with a clear error message if the file is missing or malformed. The corpus SHALL be produced reproducibly by a build step that merges a hand-written seed file with topic batch files, applies the shared system prompt uniformly to every example, removes duplicate questions, and validates the result — so the corpus is regenerated from its sources rather than edited in place.

#### Scenario: Valid dataset is loaded

- **WHEN** training starts and the dataset file exists with well-formed examples
- **THEN** every example is used for training after being rendered through the model's chat template

#### Scenario: Missing or malformed dataset

- **WHEN** training starts and the dataset file is absent or an example lacks required fields
- **THEN** the run exits non-zero with a message naming the file and the problem, before any training compute is spent

#### Scenario: Corpus is rebuilt from sources

- **WHEN** the dataset build step runs
- **THEN** it regenerates the corpus from the seed and batch files, reports how many examples were kept and how many duplicates were dropped, and validates the output

### Requirement: LoRA fine-tune of the base model

The system SHALL fine-tune the SmolLM2-360M-Instruct base model on the dataset using LoRA adapters (base weights frozen) and SHALL write the resulting adapter and training checkpoints under a gitignored artifacts directory. Training SHALL run on locally available hardware, selecting Apple Silicon GPU (MPS) when available and falling back to CPU otherwise.

#### Scenario: Training completes

- **WHEN** the train command is run with a valid dataset
- **THEN** it completes without error and writes an adapter that changes the model's answers on training-set questions relative to the base model

#### Scenario: No GPU available

- **WHEN** the train command is run on a machine without a supported GPU
- **THEN** training still completes on CPU, without code changes

### Requirement: Merged standalone model

The system SHALL merge the trained LoRA adapter into the base weights, producing a standalone model directory (weights, tokenizer, chat template, config) that loads without any adapter machinery.

#### Scenario: Merge produces a loadable model

- **WHEN** the merge step runs after training
- **THEN** the merged directory loads as a plain causal LM and generates coherent chat responses

### Requirement: Quantized ONNX export for the browser

The system SHALL export the merged model to ONNX with weight quantization, laid out in the directory structure Transformers.js expects (ONNX files under an `onnx/` subdirectory alongside tokenizer and config files), and the quantized decoder SHALL be small enough for web delivery (under ~500 MB). A successful export SHALL report success through its exit status, notwithstanding faults raised by the runtime during interpreter shutdown after the export itself has completed. Generation smoke checks SHALL pass an explicit attention mask, so their output is well defined for a tokenizer whose padding and end-of-sequence tokens coincide.

#### Scenario: Export succeeds

- **WHEN** the export step runs on a merged model
- **THEN** it produces an ONNX model directory that the web-inference capability loads successfully, with a parity smoke check confirming the ONNX model generates sane output for a sample prompt

#### Scenario: Runtime aborts during shutdown

- **WHEN** the export completes but the ONNX runtime raises a fault while tearing down at interpreter exit
- **THEN** the export stage still reports success and the pipeline continues

### Requirement: Smoke evaluation of the fine-tuned model

The system SHALL provide an evaluation command that runs a fixed set of carnivorous-plant-care prompts through the merged model and prints the answers, so a human can judge whether fine-tuning improved domain behavior.

#### Scenario: Eval prints comparable answers

- **WHEN** the eval command is run after a merge
- **THEN** each fixed prompt is printed with the merged model's answer, exiting non-zero only on execution failure (judging quality stays human)

### Requirement: Held-out validation split

The system SHALL by default hold a fraction of the dataset out of training, score it at the end of every epoch, and record those held-out metrics alongside the training metrics. On completion it SHALL report the epoch at which held-out loss was lowest, and SHALL warn when held-out loss ended materially above that minimum, since that indicates the run continued past the point of useful learning. The fraction SHALL be configurable, including disabling the split entirely.

#### Scenario: Overfitting is detected

- **WHEN** a run continues past the point where held-out loss stops improving
- **THEN** the run reports the best epoch and warns that later epochs memorised rather than learned

#### Scenario: Held-out metrics are recorded

- **WHEN** a run completes with the split enabled
- **THEN** the run's metric history contains a held-out score for each epoch, distinguishable from the training scores

#### Scenario: Split disabled

- **WHEN** the split fraction is set to zero
- **THEN** training runs on the whole dataset and no held-out metrics are produced

### Requirement: Assistant-only loss

The system SHALL offer a mode that computes the training loss only over assistant replies, masking the system prompt and user turns out of it, so the training signal is not spent on tokens the model never has to generate. Because the base model's chat template does not mark assistant spans, the repository SHALL provide a training template that does, and it SHALL render identically to the stock template so it changes only what is masked. The training template SHALL include the end-of-turn token in the trained span so the model still learns to stop, and SHALL NOT propagate to the merged or exported model.

#### Scenario: Prompt tokens are excluded from the loss

- **WHEN** training runs with assistant-only loss enabled
- **THEN** only the tokens of assistant replies contribute to the loss

#### Scenario: The exported model keeps the stock template

- **WHEN** a model trained with assistant-only loss is merged and exported
- **THEN** the artifact carries the base model's original chat template, not the training template

### Requirement: Hyperparameter overrides from the entry point

The system SHALL allow the epoch count, and arbitrary additional training flags, to be set when invoking the workflow entry point, without editing files. Overrides SHALL work from the command line and from the environment, and SHALL apply when the training stage is invoked as part of the full pipeline.

#### Scenario: Overriding the epoch count

- **WHEN** the training target is invoked with an epoch override
- **THEN** training runs for that many epochs instead of the default

#### Scenario: Passing other flags

- **WHEN** the training target is invoked with additional flags
- **THEN** those flags reach the training script unchanged

