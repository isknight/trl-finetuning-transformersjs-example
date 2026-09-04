import {
  pipeline,
  TextStreamer,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

// Load the fine-tuned model from this server only; nothing is fetched from the Hub.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';

const MODEL_ID = 'plant-bot';
const SYSTEM_PROMPT =
  'You are a knowledgeable assistant that answers questions about caring for carnivorous plants.';
const MAX_NEW_TOKENS = 256;
// SmolLM2 handles 8k tokens; stay well short of it and estimate ~4 chars per token.
const CONTEXT_CHAR_BUDGET = 6000;

const els = {
  backend: document.getElementById('backend-badge'),
  loading: document.getElementById('loading'),
  loadingLabel: document.getElementById('loading-label'),
  loadingDetail: document.getElementById('loading-detail'),
  progressBar: document.getElementById('progress-bar'),
  transcript: document.getElementById('transcript'),
  form: document.getElementById('composer'),
  input: document.getElementById('input'),
  send: document.getElementById('send'),
  clear: document.getElementById('clear'),
};

let generator = null;
let busy = false;
let history = [{ role: 'system', content: SYSTEM_PROMPT }];

function showEmptyState() {
  els.transcript.replaceChildren();
  const hint = document.createElement('p');
  hint.className = 'empty-state';
  hint.textContent =
    'Ask about watering, soil, light, feeding, dormancy, or propagation.';
  els.transcript.append(hint);
}

function addMessage(role, text) {
  const existing = els.transcript.querySelector('.empty-state');
  if (existing) existing.remove();
  const node = document.createElement('div');
  node.className = `message ${role}`;
  node.textContent = text;
  els.transcript.append(node);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return node;
}

/** Keep the system prompt plus the most recent turns that fit the budget. */
function trimHistory() {
  const system = history[0];
  const turns = history.slice(1);
  let chars = system.content.length;
  const kept = [];
  for (let i = turns.length - 1; i >= 0; i--) {
    chars += turns[i].content.length;
    if (chars > CONTEXT_CHAR_BUDGET && kept.length > 0) break;
    kept.unshift(turns[i]);
  }
  history = [system, ...kept];
}

async function pickDevice() {
  if (!navigator.gpu) return 'wasm';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? 'webgpu' : 'wasm';
  } catch {
    return 'wasm';
  }
}

function setBackendBadge(device) {
  if (device === 'webgpu') {
    els.backend.textContent = 'WebGPU accelerated';
    els.backend.className = 'badge';
  } else {
    els.backend.textContent = 'CPU (WASM) - slower';
    els.backend.className = 'badge badge-slow';
  }
}

function reportProgress(event) {
  if (event.status === 'progress' && event.total) {
    const pct = Math.min(100, (event.loaded / event.total) * 100);
    els.progressBar.style.width = `${pct}%`;
    els.loadingDetail.textContent = `${event.file} - ${pct.toFixed(0)}%`;
  } else if (event.status === 'initiate') {
    els.loadingDetail.textContent = `Fetching ${event.file}`;
  } else if (event.status === 'ready') {
    els.progressBar.style.width = '100%';
  }
}

function setReady(ready) {
  els.input.disabled = !ready;
  els.send.disabled = !ready;
  els.clear.disabled = !ready;
  els.input.placeholder = ready
    ? 'Ask about caring for carnivorous plants...'
    : 'Loading model...';
}

async function loadModel() {
  const device = await pickDevice();
  setBackendBadge(device);
  els.loadingLabel.textContent =
    device === 'webgpu'
      ? 'Downloading model (cached after the first visit)...'
      : 'Downloading model - WebGPU unavailable, falling back to CPU...';

  generator = await pipeline('text-generation', MODEL_ID, {
    device,
    dtype: device === 'webgpu' ? 'q4' : 'q8',
    progress_callback: reportProgress,
  });

  els.loading.hidden = true;
  setReady(true);
  els.input.focus();
}

async function respondTo(question) {
  history.push({ role: 'user', content: question });
  trimHistory();

  const bubble = addMessage('assistant', '');
  let answer = '';
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      answer += text;
      bubble.textContent = answer;
      els.transcript.scrollTop = els.transcript.scrollHeight;
    },
  });

  const output = await generator(history, {
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: false,
    streamer,
  });

  const final = output[0].generated_text.at(-1).content.trim();
  bubble.textContent = final || answer;
  history.push({ role: 'assistant', content: final || answer });
}

els.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = els.input.value.trim();
  if (!question || busy || !generator) return;

  busy = true;
  setReady(false);
  els.input.value = '';
  addMessage('user', question);

  try {
    await respondTo(question);
  } catch (error) {
    console.error(error);
    addMessage('error', `Generation failed: ${error.message}`);
  } finally {
    busy = false;
    setReady(true);
    els.input.focus();
  }
});

els.clear.addEventListener('click', () => {
  if (busy) return;
  history = [{ role: 'system', content: SYSTEM_PROMPT }];
  showEmptyState();
  els.input.focus();
});

showEmptyState();
loadModel().catch((error) => {
  console.error(error);
  els.loadingLabel.textContent = 'Could not load the model.';
  els.loadingDetail.textContent = `${error.message} - run \`make export\` and reload.`;
  els.backend.textContent = 'load failed';
  els.backend.className = 'badge badge-error';
});
