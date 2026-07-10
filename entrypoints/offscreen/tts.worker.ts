// Dedicated worker for pocket-tts (kyutai-labs) neural text-to-speech, run on the CPU via
// ONNX Runtime Web. Adapted from KevinAHM/pocket-tts-web's inference-worker.js: the
// text_conditioner → flow_lm_main / flow_lm_flow (flow-matching) → mimi_decoder generation
// loop and the KV-state threading are ported faithfully; the changes are (1) ORT is imported
// statically from the bundled CPU build instead of a CSP-blocked CDN import, (2) weights are
// fetched from our own self-hosted HF repo and cached in Cache Storage (with progress), and (3)
// voice-cloning / language-switching / DOM code was removed (English, single preset voice).
//
// The model is Kyutai pocket-tts (CC-BY-4.0); the tokenizer is reimplemented in pure JS
// (lib/spTokenizer.ts) because the upstream sentencepiece.js needs 'unsafe-eval' (forbidden in
// MV3). Attribution: see THIRD_PARTY_NOTICES.md.
//
// OFFSCREEN-ONLY: lives in the offscreen document's worker so the ~150 MB of sessions and the
// single-threaded wasm inference never touch the side panel's thread. Runs on the CPU (wasm),
// deliberately NOT WebGPU, so it never contends with the WebLLM engine for the GPU.
import * as ort from 'onnxruntime-web/wasm';
import { SpUnigramTokenizer } from '@/lib/spTokenizer';

// ---- Message protocol (offscreen main.ts <-> this worker) ----
interface LoadMsg { type: 'ttsLoad'; id: number }
interface SpeakMsg { type: 'ttsSpeak'; id: number; text: string }
interface SpeakStreamMsg { type: 'ttsSpeakStream'; id: number }
interface AppendMsg { type: 'ttsAppend'; id: number; seq: number; text: string }
interface FinishMsg { type: 'ttsFinish'; id: number }
interface StopMsg { type: 'ttsStop' }
type TtsWorkerRequest = LoadMsg | SpeakMsg | SpeakStreamMsg | AppendMsg | FinishMsg | StopMsg;

// ---- Assets (English, single CC-BY-4.0 "alba" voice) fetched once, then served from Cache
//      Storage. Self-hosted in our OWN public HF repo (not a third-party Space) so availability
//      and licensing are under our control — the weights are Kyutai pocket-tts under CC-BY-4.0,
//      redistributed with attribution (see THIRD_PARTY_NOTICES.md). Regenerate/upload the repo
//      contents with `node scripts/prepare-tts-assets.mjs`. ----
// NOTE: set this to YOUR public Hugging Face repo (created from tts-assets/). Update the owner
// if your HF username differs from the GitHub org.
const TTS_ASSET_BASE = 'https://huggingface.co/team-edt/enclave-tts/resolve/main/';
const CACHE_NAME = 'enclave-tts-v1';
// mimi_encoder is intentionally absent — it's only needed for cloning a voice from audio.
// Bytes are approximate (for the download progress bar); real Content-Length is used when present.
const ASSETS: { file: string; bytes: number }[] = [
  { file: 'bundle.json', bytes: 40_000 },
  { file: 'text_conditioner_int8.onnx', bytes: 16_400_000 },
  { file: 'flow_lm_main_int8.onnx', bytes: 76_300_000 },
  { file: 'flow_lm_flow_int8.onnx', bytes: 10_000_000 },
  { file: 'mimi_decoder_int8.onnx', bytes: 22_700_000 },
  { file: 'voices.bin', bytes: 6_200_000 }, // alba only (CC-BY-4.0); was 52 MB with all voices
  { file: 'tokenizer.model', bytes: 100_000 },
  { file: 'bos_before_voice.npy', bytes: 5_000 },
];
const TOTAL_BYTES = ASSETS.reduce((a, b) => a + b.bytes, 0);

// ---- Loop constants (verbatim from the reference worker) ----
const MAX_FRAMES = 500;
const LSD_STEPS = 1;
const CHUNK_GAP_SEC = 0.25;
const RESET_FLOW_STATE_EACH_CHUNK = true;
const RESET_MIMI_STATE_EACH_CHUNK = true;

interface ManifestEntry {
  dtype: 'float32' | 'int64' | 'bool';
  fill: string;
  input_name: string;
  output_name: string;
  key: string;
  module: string;
  shape: number[];
}
interface BundleMeta {
  sample_rate: number;
  samples_per_frame: number;
  latent_dim: number;
  conditioning_dim: number;
  max_token_per_chunk?: number;
  tokenizer_file: string;
  bos_before_voice_file?: string;
  insert_bos_before_voice?: boolean;
  remove_semicolons?: boolean;
  pad_with_spaces_for_short_inputs?: boolean;
  model_recommended_frames_after_eos?: number;
  predefined_voices?: string[];
  flow_lm_state_manifest: ManifestEntry[];
  mimi_state_manifest: ManifestEntry[];
}
type Raw = { data: ArrayBufferLike | Float32Array | BigInt64Array | Uint8Array; shape: number[]; dtype?: string };
type State = Record<string, ort.Tensor>;

let bundleMeta: BundleMeta | null = null;
let tokenizer: SpUnigramTokenizer | null = null;
let bosBeforeVoice: Raw | null = null;
let textConditionerSession: ort.InferenceSession | null = null;
let flowLmMainSession: ort.InferenceSession | null = null;
let flowLmFlowSession: ort.InferenceSession | null = null;
let mimiDecoderSession: ort.InferenceSession | null = null;

let sampleRate = 24000;
let samplesPerFrame = 1920;
let latentDim = 32;
let conditioningDim = 1024;
let maxTokenPerChunk = 50;

let voiceState: State | null = null; // conditioned flow-LM state for the default voice
let stTensors: { s: ort.Tensor; t: ort.Tensor }[] = [];
let isReady = false;

// ---- Streaming session state ----
// A session is opened by ttsSpeakStream, fed sentences via ttsAppend as the LLM produces them,
// and closed by ttsFinish. speakLoop drains the queue in order; playback streams out as more
// sentences arrive. Cancellation keys off sessionId (not a bare boolean) so a new session that
// preempts an old one can never be revived by the old loop.
let sessionId: number | null = null;
let queue: { seq: number; text: string }[] = [];
let finished = false;
let waiter: (() => void) | null = null;

function wakeWaiter() {
  const w = waiter;
  waiter = null;
  w?.();
}

// ---- ORT setup: the 'onnxruntime-web/wasm' bundle build inlines its JS glue and Vite emits
//      the .wasm into assets/, which ORT self-locates — so no wasmPaths is needed. Pin to a
//      single thread (extension pages aren't cross-origin isolated, so wasm threads can't
//      engage anyway) and run on the CPU (wasm), never the GPU. ----
function configureOrt() {
  ort.env.wasm.numThreads = 1;
  (ort.env.wasm as { simd?: boolean }).simd = true;
  // We already run inside a worker; don't let ORT spawn its own proxy worker (a blob/script
  // worker would trip the extension CSP). Single-threaded wasm needs no proxy.
  (ort.env.wasm as { proxy?: boolean }).proxy = false;
}

// ---- Fetch with Cache Storage + streamed progress ----
async function fetchCached(file: string, onBytes: (delta: number) => void): Promise<ArrayBuffer> {
  const url = TTS_ASSET_BASE + file;
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    onBytes(ASSETS.find((a) => a.file === file)?.bytes ?? buf.byteLength);
    return buf;
  }
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Failed to fetch ${file} (${res.status})`);
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) { bytes.set(p, offset); offset += p.length; }
  // Store under the request URL so a later match() hits regardless of response headers.
  await cache.put(url, new Response(bytes.slice().buffer, { headers: { 'content-type': 'application/octet-stream' } }));
  return bytes.buffer;
}

// ---- Tensor helpers (ported) ----
function makeFilledArray(shape: number[], dtype: string, fill: string) {
  const size = shape.reduce((a, b) => a * b, 1);
  if (dtype === 'int64') return new BigInt64Array(size);
  if (dtype === 'bool') return new Uint8Array(size);
  const data = new Float32Array(size);
  if (fill === 'nan') data.fill(NaN);
  else if (fill === 'ones') data.fill(1);
  return data;
}
function tensor(dtype: string, data: ort.Tensor['data'], dims: number[]): ort.Tensor {
  return new ort.Tensor(dtype as 'float32', data as Float32Array, dims);
}
function initStateFromManifest(manifest: ManifestEntry[]): State {
  const state: State = {};
  for (const e of manifest) {
    state[e.input_name] = tensor(e.dtype, makeFilledArray(e.shape, e.dtype, e.fill) as ort.Tensor['data'], e.shape);
  }
  return state;
}
function cloneState(state: State): State {
  return { ...state };
}
function updateStateFromOutputs(state: State, result: ort.InferenceSession.OnnxValueMapType, manifest: ManifestEntry[]) {
  for (const e of manifest) state[e.input_name] = result[e.output_name] as ort.Tensor;
}

// ---- voices.bin parser (ported) ----
function groupVoiceRecordByModule(record: Record<string, Raw>) {
  const grouped: Record<string, Record<string, Raw>> = {};
  for (const [key, value] of Object.entries(record)) {
    const slash = key.indexOf('/');
    if (slash === -1) continue;
    const moduleName = key.slice(0, slash);
    const tensorKey = key.slice(slash + 1);
    (grouped[moduleName] ||= {})[tensorKey] = value;
  }
  return grouped;
}
function adaptTypedArray(source: Raw, entry: ManifestEntry) {
  const targetShape = entry.shape;
  const targetSize = targetShape.reduce((a, b) => a * b, 1);
  const target = makeFilledArray(targetShape, entry.dtype, entry.fill);
  const srcData = source.data as ArrayLike<number> & Iterable<number>;
  const exact =
    source.shape.length === targetShape.length && source.shape.every((d, i) => d === targetShape[i]);
  if (exact || (srcData as ArrayLike<number>).length === targetSize) {
    if (entry.dtype === 'int64') return new BigInt64Array(srcData as Iterable<bigint>);
    if (entry.dtype === 'bool') return new Uint8Array(srcData as ArrayLike<number>);
    return new Float32Array(srcData as ArrayLike<number>);
  }
  if (source.shape.length !== targetShape.length) return target;
  const strides: number[] = [];
  let stride = 1;
  for (let i = source.shape.length - 1; i >= 0; i--) { strides[i] = stride; stride *= source.shape[i]; }
  const indices = new Array(source.shape.length).fill(0);
  const maxIndices = source.shape.map((d, i) => Math.min(d, targetShape[i]));
  const targetIndex = (coords: number[]) => {
    let idx = 0, tStride = 1;
    for (let i = targetShape.length - 1; i >= 0; i--) { idx += coords[i] * tStride; tStride *= targetShape[i]; }
    return idx;
  };
  const arr = srcData as ArrayLike<number>;
  let done = false;
  while (!done) {
    let sourceIdx = 0;
    for (let i = 0; i < indices.length; i++) sourceIdx += indices[i] * strides[i];
    (target as Float32Array)[targetIndex(indices)] = arr[sourceIdx];
    for (let dim = indices.length - 1; dim >= 0; dim--) {
      indices[dim] += 1;
      if (indices[dim] < maxIndices[dim]) break;
      indices[dim] = 0;
      if (dim === 0) done = true;
    }
  }
  return target;
}
function deriveStep(moduleState: Record<string, Raw>): Raw {
  if (moduleState.step) return { data: BigInt64Array.from([BigInt((moduleState.step.data as BigInt64Array)[0])]), shape: [1], dtype: 'int64' };
  if (moduleState.offset && !moduleState.end_offset) return { data: BigInt64Array.from([BigInt((moduleState.offset.data as BigInt64Array)[0])]), shape: [1], dtype: 'int64' };
  if (moduleState.current_end) return { data: BigInt64Array.from([BigInt(moduleState.current_end.shape[0])]), shape: [1], dtype: 'int64' };
  return { data: BigInt64Array.from([0n]), shape: [1], dtype: 'int64' };
}
function stateFromVoiceRecord(record: Record<string, Raw>): State {
  const grouped = groupVoiceRecordByModule(record);
  const state = initStateFromManifest(bundleMeta!.flow_lm_state_manifest);
  for (const entry of bundleMeta!.flow_lm_state_manifest) {
    const moduleState = grouped[entry.module] || {};
    let source = moduleState[entry.key];
    if (!source && entry.key === 'step') source = deriveStep(moduleState);
    if (!source) continue;
    const data = adaptTypedArray(source, entry);
    state[entry.input_name] = tensor(entry.dtype, data as ort.Tensor['data'], entry.shape);
  }
  return state;
}
function parseNpyFloat32(buffer: ArrayBuffer): Raw {
  const view = new DataView(buffer);
  const magic = new Uint8Array(buffer, 0, 6);
  const expected = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < expected.length; i++) if (magic[i] !== expected[i]) throw new Error('Invalid NPY file');
  const major = view.getUint8(6);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, headerOffset, headerLen));
  const shapeMatch = headerText.match(/\(\s*([0-9,\s]+)\)/);
  if (!shapeMatch) throw new Error('Could not parse NPY shape');
  const shape = shapeMatch[1].split(',').map((p) => p.trim()).filter(Boolean).map((p) => Number.parseInt(p, 10));
  const data = new Float32Array(buffer, headerOffset + headerLen);
  return { data: new Float32Array(data), shape };
}
function parseVoiceStatesBin(buffer: ArrayBuffer): Record<string, Record<string, Raw>> {
  const view = new DataView(buffer);
  let offset = 0;
  const magic = new TextDecoder().decode(new Uint8Array(buffer, offset, 5));
  offset += 5;
  if (magic !== 'PTVB1') throw new Error('Invalid voices.bin header');
  const voices: Record<string, Record<string, Raw>> = {};
  const voiceCount = view.getUint32(offset, true); offset += 4;
  for (let vi = 0; vi < voiceCount; vi++) {
    const nameLen = view.getUint16(offset, true); offset += 2;
    const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen)); offset += nameLen;
    const tensorCount = view.getUint16(offset, true); offset += 2;
    const tensors: Record<string, Raw> = {};
    for (let ti = 0; ti < tensorCount; ti++) {
      const keyLen = view.getUint16(offset, true); offset += 2;
      const key = new TextDecoder().decode(new Uint8Array(buffer, offset, keyLen)); offset += keyLen;
      const dtypeCode = view.getUint8(offset); offset += 1;
      const rank = view.getUint8(offset); offset += 1;
      const shape: number[] = [];
      for (let di = 0; di < rank; di++) { shape.push(view.getUint32(offset, true)); offset += 4; }
      const byteLength = view.getUint32(offset, true); offset += 4;
      let data: Float32Array | BigInt64Array | Uint8Array;
      if (dtypeCode === 0) data = new Float32Array(buffer.slice(offset, offset + byteLength));
      else if (dtypeCode === 1) data = new BigInt64Array(buffer.slice(offset, offset + byteLength));
      else if (dtypeCode === 2) data = new Uint8Array(buffer.slice(offset, offset + byteLength));
      else throw new Error(`Unsupported voices.bin dtype code: ${dtypeCode}`);
      offset += byteLength;
      tensors[key] = { data, shape, dtype: dtypeCode === 0 ? 'float32' : dtypeCode === 1 ? 'int64' : 'bool' };
    }
    voices[name] = tensors;
  }
  return voices;
}

// ---- Text preprocessing / chunking (ported) ----
function prepareTextPrompt(text: string) {
  let prompt = text.trim();
  if (!prompt) return { text: '', framesAfterEos: 1 };
  prompt = prompt.replace(/\r/g, ' ').replace(/\n/g, ' ').replace(/\s+/g, ' ');
  if (bundleMeta!.remove_semicolons) prompt = prompt.replace(/;/g, ',');
  const wordCount = prompt.split(/\s+/).filter(Boolean).length;
  let framesAfterEos = wordCount <= 4 ? 3 : 1;
  if (bundleMeta!.model_recommended_frames_after_eos != null) framesAfterEos = Number(bundleMeta!.model_recommended_frames_after_eos);
  if (prompt && !/[A-ZÀ-Þ]/.test(prompt[0])) prompt = prompt[0].toUpperCase() + prompt.slice(1);
  if (prompt && /[0-9A-Za-zÀ-ÿ]/.test(prompt[prompt.length - 1])) prompt += '.';
  if (bundleMeta!.pad_with_spaces_for_short_inputs && wordCount < 5) prompt = '        ' + prompt;
  return { text: prompt, framesAfterEos };
}
const SENTENCE_SPLIT_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;
function splitTextIntoSentences(text: string) {
  const matches = text.match(SENTENCE_SPLIT_RE);
  return matches ? matches.map((s) => s.trim()).filter(Boolean) : [];
}
function splitTokenIdsIntoChunks(tokenIds: number[], maxTokens: number) {
  const chunks: string[] = [];
  for (let i = 0; i < tokenIds.length; i += maxTokens) {
    const t = tokenizer!.decodeIds(tokenIds.slice(i, i + maxTokens)).trim();
    if (t) chunks.push(t);
  }
  return chunks;
}
function splitIntoBestSentences(text: string) {
  const prepared = prepareTextPrompt(text);
  if (!prepared.text) return { chunks: [] as string[], framesAfterEos: prepared.framesAfterEos };
  const sentences = splitTextIntoSentences(prepared.text);
  if (!sentences.length) return { chunks: [prepared.text], framesAfterEos: prepared.framesAfterEos };
  const chunks: string[] = [];
  let currentChunk = '';
  for (const sentenceText of sentences) {
    const sentenceTokenIds = tokenizer!.encodeIds(sentenceText);
    if (sentenceTokenIds.length > maxTokenPerChunk) {
      if (currentChunk) { chunks.push(currentChunk.trim()); currentChunk = ''; }
      for (const s of splitTokenIdsIntoChunks(sentenceTokenIds, maxTokenPerChunk)) if (s) chunks.push(s.trim());
      continue;
    }
    if (!currentChunk) { currentChunk = sentenceText; continue; }
    const combined = `${currentChunk} ${sentenceText}`;
    if (tokenizer!.encodeIds(combined).length > maxTokenPerChunk) { chunks.push(currentChunk.trim()); currentChunk = sentenceText; }
    else currentChunk = combined;
  }
  if (currentChunk) chunks.push(currentChunk.trim());
  return { chunks, framesAfterEos: prepared.framesAfterEos };
}

function precomputeFlowBuffers() {
  stTensors = [];
  const dt = 1.0 / LSD_STEPS;
  for (let step = 0; step < LSD_STEPS; step++) {
    const s = step / LSD_STEPS;
    stTensors.push({
      s: tensor('float32', new Float32Array([s]), [1, 1]),
      t: tensor('float32', new Float32Array([s + dt]), [1, 1]),
    });
  }
}

// ---- Load everything (fetch+cache weights, build sessions, tokenizer, default voice) ----
async function load(id: number) {
  configureOrt();
  precomputeFlowBuffers();

  let loaded = 0;
  const bump = (delta: number) => {
    loaded += delta;
    post({ type: 'ttsProgress', id, progress: Math.min(1, loaded / TOTAL_BYTES) });
  };
  const buffers: Record<string, ArrayBuffer> = {};
  // Sequential fetch keeps peak memory and the progress bar sane.
  for (const a of ASSETS) buffers[a.file] = await fetchCached(a.file, bump);

  bundleMeta = JSON.parse(new TextDecoder().decode(buffers['bundle.json'])) as BundleMeta;
  sampleRate = Number(bundleMeta.sample_rate);
  samplesPerFrame = Number(bundleMeta.samples_per_frame);
  latentDim = Number(bundleMeta.latent_dim);
  conditioningDim = Number(bundleMeta.conditioning_dim);
  maxTokenPerChunk = Number(bundleMeta.max_token_per_chunk || 50);

  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  [textConditionerSession, flowLmMainSession, flowLmFlowSession, mimiDecoderSession] = await Promise.all([
    ort.InferenceSession.create(new Uint8Array(buffers['text_conditioner_int8.onnx']), opts),
    ort.InferenceSession.create(new Uint8Array(buffers['flow_lm_main_int8.onnx']), opts),
    ort.InferenceSession.create(new Uint8Array(buffers['flow_lm_flow_int8.onnx']), opts),
    ort.InferenceSession.create(new Uint8Array(buffers['mimi_decoder_int8.onnx']), opts),
  ]);

  // Tokenizer: pure-JS SentencePiece Unigram (no eval — the emscripten build's embind needs
  // 'unsafe-eval', which MV3 forbids). Verified id-for-id against Python sentencepiece.
  tokenizer = new SpUnigramTokenizer();
  tokenizer.load(buffers['tokenizer.model']);

  bosBeforeVoice = null;
  if (bundleMeta.bos_before_voice_file && buffers['bos_before_voice.npy']) {
    bosBeforeVoice = parseNpyFloat32(buffers['bos_before_voice.npy']);
  }

  const voiceRecords = parseVoiceStatesBin(buffers['voices.bin']);
  let defaultVoice = bundleMeta.predefined_voices?.includes('alba') ? 'alba' : null;
  if (!defaultVoice) defaultVoice = Object.keys(voiceRecords)[0] || null;
  if (!defaultVoice || !voiceRecords[defaultVoice]) throw new Error('No preset voice available in voices.bin');
  voiceState = stateFromVoiceRecord(voiceRecords[defaultVoice]);

  // Sanity-touch bos so the linter doesn't flag it as unused (voice cloning would consume it).
  void bosBeforeVoice;

  isReady = true;
  post({ type: 'ttsReady', id });
}

// ---- Streaming session control ----
/** Open a session and start draining it. Preempts any session already running. */
function startSession(id: number) {
  sessionId = id; // an old speakLoop keyed to a different id will see the mismatch and exit
  queue = [];
  finished = false;
  wakeWaiter();
  void speakLoop(id);
}

function appendToSession(id: number, seq: number, text: string) {
  if (id !== sessionId) return; // a stale append for a preempted session
  queue.push({ seq, text });
  wakeWaiter();
}

function finishSession(id: number) {
  if (id !== sessionId) return;
  finished = true;
  wakeWaiter();
}

/** Stop the current session immediately (main.ts's ttsStop handler emits ttsEnded itself, so we
 *  deliberately do NOT post ttsSpeakDone here). */
function stopSession() {
  sessionId = null;
  queue = [];
  finished = true;
  wakeWaiter();
}

/** Drain the session's sentence queue in order, generating and streaming audio as sentences
 *  arrive. Posts ttsSpeakDone exactly once, only after the queue is drained AND finished — so
 *  the panel's ttsEnded fires at the true end of the whole reply. */
async function speakLoop(id: number) {
  if (!isReady || !flowLmMainSession || !flowLmFlowSession || !mimiDecoderSession || !textConditionerSession || !voiceState || !bundleMeta) {
    post({ type: 'ttsError', id, message: 'Voice model not loaded.' });
    if (sessionId === id) sessionId = null;
    return;
  }
  let firstAudio = true;
  try {
    while (sessionId === id) {
      if (!queue.length) {
        if (finished) break;
        await new Promise<void>((r) => { waiter = r; });
        continue;
      }
      const item = queue.shift()!;
      if (sessionId !== id) break;
      firstAudio = await generateItem(id, item.seq, item.text, firstAudio);
    }
    if (sessionId === id) post({ type: 'ttsSpeakDone', id });
  } catch (err) {
    post({ type: 'ttsError', id, message: err instanceof Error ? err.message : String(err) });
  } finally {
    if (sessionId === id) sessionId = null;
  }
}

// ---- Generation loop for one sentence (ported faithfully; tags audio with the append `seq`
//      so the panel can reveal the shown text in step with the spoken audio). Returns the
//      updated firstAudio flag so the very first audio of the whole session uses the small
//      first-chunk framing for a fast start. ----
async function generateItem(id: number, seq: number, text: string, firstAudio: boolean): Promise<boolean> {
  const { chunks, framesAfterEos } = splitIntoBestSentences(text);
  if (!chunks.length) return firstAudio;

  let mimiState = initStateFromManifest(bundleMeta!.mimi_state_manifest);
  const emptySeq = tensor('float32', new Float32Array(0), [1, 0, latentDim]);
  const emptyTextEmb = tensor('float32', new Float32Array(0), [1, 0, conditioningDim]);
  const baseFlowState = voiceState!;
  let flowLmState = cloneState(baseFlowState);

  const firstChunkFrames = 3;
  const normalChunkFrames = 12;
  let isFirstAudioChunk = firstAudio;

  for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
    if (sessionId !== id) break;
    if (RESET_FLOW_STATE_EACH_CHUNK && chunkIdx > 0) flowLmState = cloneState(baseFlowState);
    if (RESET_MIMI_STATE_EACH_CHUNK && chunkIdx > 0) mimiState = initStateFromManifest(bundleMeta!.mimi_state_manifest);

    const chunkText = chunks[chunkIdx];
    const tokenIds = tokenizer!.encodeIds(chunkText);
    const textInput = tensor('int64', BigInt64Array.from(tokenIds.map((t) => BigInt(t))), [1, tokenIds.length]);

    let textEmb = (await textConditionerSession!.run({ token_ids: textInput }))[textConditionerSession!.outputNames[0]] as ort.Tensor;
    if (textEmb.dims.length === 2) textEmb = tensor('float32', new Float32Array(textEmb.data as Float32Array), [1, textEmb.dims[0], textEmb.dims[1]]);

    const condResult = await flowLmMainSession!.run({ sequence: emptySeq, text_embeddings: textEmb, ...flowLmState });
    updateStateFromOutputs(flowLmState, condResult, bundleMeta!.flow_lm_state_manifest);

    const chunkLatents: Float32Array[] = [];
    let chunkDecodedFrames = 0;
    let currentLatent = tensor('float32', new Float32Array(latentDim).fill(NaN), [1, 1, latentDim]);
    let eosStep: number | null = null;
    let chunkEnded = false;

    for (let step = 0; step < MAX_FRAMES; step++) {
      if (sessionId !== id) break;
      // Yield periodically so ttsStop / ttsAppend / ttsFinish messages are processed mid-generation.
      if (step > 0 && step % 4 === 0) await new Promise((r) => setTimeout(r, 0));

      const arResult = await flowLmMainSession!.run({ sequence: currentLatent, text_embeddings: emptyTextEmb, ...flowLmState });
      const conditioning = arResult.conditioning as ort.Tensor;
      const eosLogit = (arResult.eos_logit as ort.Tensor).data as Float32Array;
      const isEos = eosLogit[0] > -4.0;
      if (isEos && eosStep == null) eosStep = step;
      const shouldStop = eosStep != null && step >= eosStep + framesAfterEos;

      const temperature = 0.7;
      const std = Math.sqrt(temperature);
      const latentData = new Float32Array(latentDim);
      for (let i = 0; i < latentDim; i++) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        latentData[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
      }
      const dt = 1.0 / LSD_STEPS;
      for (let lsdIndex = 0; lsdIndex < LSD_STEPS; lsdIndex++) {
        const flowResult = await flowLmFlowSession!.run({
          c: conditioning,
          s: stTensors[lsdIndex].s,
          t: stTensors[lsdIndex].t,
          x: tensor('float32', latentData, [1, latentDim]),
        });
        const flowDir = (flowResult.flow_dir as ort.Tensor).data as Float32Array;
        for (let i = 0; i < latentDim; i++) latentData[i] += flowDir[i] * dt;
      }

      chunkLatents.push(new Float32Array(latentData));
      currentLatent = tensor('float32', latentData, [1, 1, latentDim]);
      updateStateFromOutputs(flowLmState, arResult, bundleMeta!.flow_lm_state_manifest);

      const pending = chunkLatents.length - chunkDecodedFrames;
      let decodeSize = 0;
      if (shouldStop) decodeSize = pending;
      else if (isFirstAudioChunk && pending >= firstChunkFrames) decodeSize = firstChunkFrames;
      else if (pending >= normalChunkFrames) decodeSize = normalChunkFrames;

      if (decodeSize > 0) {
        const decodeLatents = new Float32Array(decodeSize * latentDim);
        for (let f = 0; f < decodeSize; f++) decodeLatents.set(chunkLatents[chunkDecodedFrames + f], f * latentDim);
        const decodeResult = await mimiDecoderSession!.run({ latent: tensor('float32', decodeLatents, [1, decodeSize, latentDim]), ...mimiState });
        for (const e of bundleMeta!.mimi_state_manifest) mimiState[e.input_name] = decodeResult[e.output_name] as ort.Tensor;
        chunkDecodedFrames += decodeSize;
        const audio = new Float32Array((decodeResult[mimiDecoderSession!.outputNames[0]] as ort.Tensor).data as Float32Array);
        post({ type: 'audioChunk', data: audio, sampleRate, seq, isFirst: isFirstAudioChunk, isLast: false }, [audio.buffer]);
        isFirstAudioChunk = false;
      }
      if (shouldStop) { chunkEnded = true; break; }
    }

    // Brief silence between sentence chunks.
    if (chunkEnded && sessionId === id && chunkIdx < chunks.length - 1) {
      const gap = new Float32Array(Math.max(1, Math.floor(CHUNK_GAP_SEC * sampleRate)));
      post({ type: 'audioChunk', data: gap, sampleRate, seq, isFirst: false, isLast: false }, [gap.buffer]);
    }
  }

  void samplesPerFrame; // (kept from bundle for parity; not needed for playback)
  return isFirstAudioChunk;
}

function post(msg: Record<string, unknown>, transfer?: Transferable[]) {
  if (transfer) (self as unknown as Worker).postMessage(msg, transfer);
  else (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (e: MessageEvent<TtsWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'ttsStop') { stopSession(); return; }
    if (msg.type === 'ttsLoad') {
      if (isReady) { post({ type: 'ttsReady', id: msg.id }); return; }
      await load(msg.id);
      return;
    }
    if (msg.type === 'ttsSpeakStream') { startSession(msg.id); return; }
    if (msg.type === 'ttsAppend') { appendToSession(msg.id, msg.seq, msg.text); return; }
    if (msg.type === 'ttsFinish') { finishSession(msg.id); return; }
    if (msg.type === 'ttsSpeak') {
      // One-shot speak is a stream of exactly one sentence-blob: open, append, finish.
      startSession(msg.id);
      appendToSession(msg.id, 0, msg.text);
      finishSession(msg.id);
      return;
    }
  } catch (err) {
    post({ type: 'ttsError', id: (msg as { id?: number }).id ?? -1, message: err instanceof Error ? err.message : String(err) });
  }
};
