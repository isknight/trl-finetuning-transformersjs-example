# Spec Delta: fine-tuning

## Purpose

Turn a chat-formatted Q&A dataset about carnivorous plant care into a fine-tuned small language model, exported in a form the browser inference capability can serve.

## ADDED Requirements

### Requirement: Dataset in chat format

The system SHALL read training data from a project-local dataset file of Q&A examples in chat-message format (each example holding user and assistant turns), and SHALL fail with a clear error message if the file is missing or malformed. The repository SHALL include a small seed dataset of carnivorous plant care Q&A so the pipeline runs end to end without external data.

#### Scenario: Valid dataset is loaded

- **WHEN** training starts and the dataset file exists with well-formed examples
- **THEN** every example is used for training after being rendered through the model's chat template

#### Scenario: Missing or malformed dataset

- **WHEN** training starts and the dataset file is absent or an example lacks required fields
- **THEN** the run exits non-zero with a message naming the file and the problem, before any training compute is spent

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

The system SHALL export the merged model to ONNX with weight quantization, laid out in the directory structure Transformers.js expects (ONNX files under an `onnx/` subdirectory alongside tokenizer and config files), and the quantized decoder SHALL be small enough for web delivery (under ~500 MB).

#### Scenario: Export succeeds

- **WHEN** the export step runs on a merged model
- **THEN** it produces an ONNX model directory that the web-inference capability loads successfully, with a parity smoke check confirming the ONNX model generates sane output for a sample prompt

### Requirement: Smoke evaluation of the fine-tuned model

The system SHALL provide an evaluation command that runs a fixed set of carnivorous-plant-care prompts through the merged model and prints the answers, so a human can judge whether fine-tuning improved domain behavior.

#### Scenario: Eval prints comparable answers

- **WHEN** the eval command is run after a merge
- **THEN** each fixed prompt is printed with the merged model's answer, exiting non-zero only on execution failure (judging quality stays human)
