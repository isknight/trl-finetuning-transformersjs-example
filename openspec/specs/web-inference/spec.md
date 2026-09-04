# web-inference Specification

## Purpose
Serve the fine-tuned model as a static web page that answers carnivorous plant care questions with all inference running client-side in the visitor's browser.
## Requirements
### Requirement: Fully client-side chat inference

The system SHALL provide a static web app that loads the exported ONNX model in the browser and answers chat messages locally: after the page and model assets are fetched, generating a response SHALL require no network requests to any inference backend.

#### Scenario: Question answered locally

- **WHEN** the page has finished loading the model and the user submits a question
- **THEN** a generated answer appears, and no network request carrying the prompt or completion leaves the browser

### Requirement: WebGPU with WASM fallback

The app SHALL run inference on WebGPU when the browser supports it and SHALL fall back to WASM (CPU) execution otherwise, telling the user which backend is active and warning that the fallback is slower.

#### Scenario: WebGPU available

- **WHEN** the page loads in a browser with WebGPU enabled
- **THEN** inference runs on the WebGPU backend and the UI indicates GPU acceleration is active

#### Scenario: WebGPU unavailable

- **WHEN** the page loads in a browser without WebGPU
- **THEN** the app still works via WASM and the UI notes reduced speed

### Requirement: Load progress and streaming responses

The app SHALL show download/initialization progress while model assets load (with input disabled until ready) and SHALL stream generated tokens into the chat as they are produced rather than waiting for the full answer.

#### Scenario: First visit load

- **WHEN** a user first opens the page
- **THEN** a progress indicator reflects model download state, and the input box becomes active only once the model is ready

#### Scenario: Streaming answer

- **WHEN** the model is generating a response
- **THEN** tokens appear incrementally in the chat transcript

### Requirement: Multi-turn chat session

The app SHALL maintain the conversation within a page session, sending prior turns as context so follow-up questions work, and SHALL provide a control to clear the conversation. Conversation state MAY be lost on page reload.

#### Scenario: Follow-up question uses context

- **WHEN** the user asks "how often should I water it?" after a prior exchange about a Venus flytrap
- **THEN** the model receives the earlier turns as context for its answer

### Requirement: Local development server

The system SHALL provide a one-command way to serve the web app and exported model locally over HTTP with the response headers required for browser inference (cross-origin isolation for multithreaded WASM), and SHALL fail with a clear message if the model export is missing.

#### Scenario: Serve after export

- **WHEN** the serve command runs after a successful export
- **THEN** the chat app is reachable on a local port and successfully loads the exported model

#### Scenario: Serve without export

- **WHEN** the serve command runs and no exported model exists
- **THEN** it exits with a message telling the user to run the export step first

