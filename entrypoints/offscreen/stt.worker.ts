// Dedicated worker for local speech-to-text: Silero VAD v5 for endpointing + Moonshine (base)
// for transcription. Runs on the CPU (wasm) in the offscreen document's worker so the ~60 MB
// Moonshine model and the per-frame inference never touch the side panel's thread, and never
// contend with the WebLLM engine for the GPU (which owns it).
//
// Mic PCM arrives here as transferred 512-sample Float32 frames (16 kHz) from the offscreen main
// thread (lib/micCapture.ts); only small JSON control/status messages leave over the panel Port.
//
// TWO ORT INSTANCES COEXIST in this worker, by design: Silero runs on the 'onnxruntime-web/wasm'
// bundle build (self-locating .wasm, like tts.worker.ts), while Moonshine runs through
// @huggingface/transformers, which uses its own onnxruntime-web loaded from the bundled /ort/
// copy (lib/ortEnv.ts). They are separate module instances (~15-20 MB extra wasm memory —
// acceptable) whose env configs don't collide.
//
// Attribution (see THIRD_PARTY_NOTICES.md): Moonshine is MIT (Useful Sensors); Silero VAD is
// MIT (Silero Team). The VAD endpointing logic is ported from @ricky0123/vad-web (MIT).
import * as ort from 'onnxruntime-web/wasm';
import { pipeline, type AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers';
import { configureOrtRuntime } from '@/lib/ortEnv';
import { MOONSHINE_MODEL } from '@/lib/voiceCache';
import { VadFrameProcessor, VAD_FRAME_SAMPLES, DEFAULT_VAD_OPTIONS, pauseMsToRedemptionFrames } from '@/lib/vadFrameProcessor';

// ---- Message protocol (offscreen main.ts <-> this worker) ----
interface LoadMsg { type: 'sttLoad'; id: number }
interface StartMsg { type: 'sttStart'; id: number; mode: 'ptt' | 'auto'; pauseMs?: number }
interface FrameMsg { type: 'frame'; data: Float32Array }
interface MuteMsg { type: 'sttMute'; muted: boolean }
interface DuckingMsg { type: 'sttDucking'; active: boolean }
interface StopMsg { type: 'sttStop'; flush?: boolean }
interface ReleaseMsg { type: 'sttRelease' }
type SttWorkerRequest = LoadMsg | StartMsg | FrameMsg | MuteMsg | DuckingMsg | StopMsg | ReleaseMsg;

// While TTS is audibly playing (barge-in mode), the mic stays live but the VAD is made stricter
// so the assistant's own voice / a transient can't trigger a false interruption (~160 ms of
// clearly-voiced speech is needed to confirm an onset).
const DUCK_PROFILE = { positiveSpeechThreshold: 0.7, negativeSpeechThreshold: 0.5, minSpeechFrames: 8, startConfirmFrames: 5 };
// The base onset profile the ducking overrides restore (redemptionFrames is per-session and left
// untouched).
const BASE_ONSET = {
  positiveSpeechThreshold: DEFAULT_VAD_OPTIONS.positiveSpeechThreshold,
  negativeSpeechThreshold: DEFAULT_VAD_OPTIONS.negativeSpeechThreshold,
  minSpeechFrames: DEFAULT_VAD_OPTIONS.minSpeechFrames,
  startConfirmFrames: DEFAULT_VAD_OPTIONS.startConfirmFrames,
};

// ---- Models ----
// Silero VAD is vendored in the extension (public/vad/); Moonshine's weights are fetched once
// from the Hugging Face CDN and auto-cached by transformers.js (like the embedding model).
const SILERO_URL = 'vad/silero_vad_v5.onnx';
// Per-file dtypes, both verified: q8/int8 DECODER files fail session creation under the bundled
// onnxruntime-web ("TransposeDQWeightsForMatMulNBits Missing required scale ..." — a QDQ
// transform bug this runtime hits on those exports; onnxruntime-node loads them fine). The q4
// decoder loads AND transcribes identically to fp32 (JFK sample, word-for-word). Keep
// STT_DOWNLOAD_MB (lib/sttClient.ts) in sync when changing this: encoder q8 ~21 MB + decoder
// q4 ~73 MB ≈ 93 MB.
const MOONSHINE_DTYPE = { encoder_model: 'q8', decoder_model_merged: 'q4' } as const;
const SAMPLE_RATE = 16000;

let sileroSession: ort.InferenceSession | null = null;
let sileroState: ort.Tensor | null = null;
let sileroSr: ort.Tensor | null = null;
let asr: AutomaticSpeechRecognitionPipeline | null = null;
let isReady = false;
// A second sttLoad while one is in flight must join it, not start a duplicate download.
let loadPromise: Promise<void> | null = null;

// ---- Session state (one active listening session at a time) ----
let activeId: number | null = null;
let mode: 'ptt' | 'auto' = 'auto';
let vad: VadFrameProcessor | null = null;
let pttBuffer: Float32Array[] = []; // ptt mode records everything, then transcribes on stop
let muted = false;
let transcribing = false;
let ducked = false; // VAD currently in the stricter barge-in profile (TTS playing)
// Frames are processed one at a time: Silero threads a recurrent `state` across calls, so
// concurrent runs would corrupt it. Incoming frames queue here and a single loop drains them.
let frameQueue: Float32Array[] = [];
let draining = false;

// ---- ORT setup for Silero (the 'onnxruntime-web/wasm' bundle build; identical to tts.worker) ----
function configureSileroOrt() {
  ort.env.wasm.numThreads = 1;
  (ort.env.wasm as { simd?: boolean }).simd = true;
  (ort.env.wasm as { proxy?: boolean }).proxy = false;
}

/** Extension-origin URL for a bundled asset. chrome.runtime is unavailable in dedicated
 *  workers, so fall back to resolving against the worker script's own URL. */
function extUrl(path: string): string {
  const runtime = (globalThis as { chrome?: { runtime?: { getURL?: (p: string) => string } } }).chrome?.runtime;
  if (runtime?.getURL) return runtime.getURL(path.replace(/^\//, ''));
  return new URL('/' + path.replace(/^\//, ''), location.href).href;
}

function resetSileroState() {
  sileroState = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]);
}

// ---- Load Silero + Moonshine ----
async function load(id: number) {
  configureSileroOrt();
  // Silero is tiny and local — fetch the bundled model and build its session.
  const bytes = new Uint8Array(await (await fetch(extUrl(SILERO_URL))).arrayBuffer());
  sileroSession = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  resetSileroState();
  sileroSr = new ort.Tensor('int64', BigInt64Array.from([BigInt(SAMPLE_RATE)]));

  // Moonshine via transformers.js — uses the bundled /ort/ wasm paths (CSP-safe).
  configureOrtRuntime();
  asr = (await pipeline('automatic-speech-recognition', MOONSHINE_MODEL, {
    device: 'wasm',
    dtype: MOONSHINE_DTYPE,
    progress_callback: (e: unknown) => {
      const info = e as { status?: string; progress?: number };
      if (typeof info.progress === 'number') post({ type: 'sttProgress', id, progress: info.progress / 100 });
    },
  })) as AutomaticSpeechRecognitionPipeline;

  isReady = true;
  post({ type: 'sttReady', id });
}

/** One Silero VAD step over a 512-sample frame → speech probability in [0,1]. */
async function sileroProb(frame: Float32Array): Promise<number> {
  if (!sileroSession || !sileroState || !sileroSr) return 0;
  const input = new ort.Tensor('float32', frame, [1, frame.length]);
  const out = await sileroSession.run({ input, state: sileroState, sr: sileroSr });
  sileroState = out.stateN as ort.Tensor;
  return (out.output.data as Float32Array)[0];
}

/** Transcribe a raw 16 kHz Float32 utterance with Moonshine. Returns trimmed text. */
async function transcribe(audio: Float32Array): Promise<string> {
  if (!asr) return '';
  const result = (await asr(audio)) as { text?: string } | { text?: string }[];
  const text = Array.isArray(result) ? result[0]?.text : result.text;
  return (text ?? '').trim();
}

/** Auto mode discards very short transcripts — Moonshine hallucinates a word or two on pure
 *  noise/breath, and we don't want those auto-submitted. */
function isNoiseTranscript(text: string): boolean {
  const words = text.split(/\s+/).filter(Boolean);
  return words.length < 2;
}

function startSession(id: number, m: 'ptt' | 'auto', pauseMs?: number) {
  activeId = id;
  mode = m;
  muted = false;
  transcribing = false;
  ducked = false;
  pttBuffer = [];
  frameQueue = [];
  // The "pause before responding" setting maps straight onto the VAD redemption window.
  vad = new VadFrameProcessor(
    pauseMs != null ? { redemptionFrames: pauseMsToRedemptionFrames(pauseMs) } : {},
  );
  resetSileroState();
  post({ type: 'sttState', id, state: 'listening' });
}

/** Queue a mic frame and ensure the single drain loop is running. */
function enqueueFrame(frame: Float32Array) {
  // Cap the backlog so a stall (e.g. a slow transcription) can't grow it without bound; drop the
  // oldest, which is the least relevant for live VAD.
  if (frameQueue.length > 200) frameQueue.shift();
  frameQueue.push(frame);
  void drainFrames();
}

async function drainFrames() {
  if (draining) return;
  draining = true;
  try {
    while (frameQueue.length) {
      await onFrame(frameQueue.shift()!);
    }
  } finally {
    draining = false;
  }
}

async function onFrame(frame: Float32Array) {
  if (activeId == null || !isReady) return;
  if (muted || transcribing) return; // half-duplex: ignore mic while TTS plays / mid-transcribe
  const id = activeId;

  if (mode === 'ptt') {
    // Push-to-talk records everything; VAD isn't used to gate (the user controls start/stop).
    pttBuffer.push(frame);
    return;
  }

  // auto mode: endpoint with Silero + the VAD state machine.
  let prob = 0;
  try {
    prob = await sileroProb(frame);
  } catch {
    return; // a transient ORT hiccup on one frame shouldn't kill the session
  }
  // A mute/stop may have landed while awaiting the Silero step.
  if (activeId !== id || muted || transcribing || !vad) return;

  const ev = vad.process(frame, prob);
  if (ev.event === 'speechStart') {
    // A confirmed onset during playback IS the barge-in — drop the ducking so the rest of the
    // utterance (the user's words) is captured at normal sensitivity instead of clipped.
    if (ducked) { vad.updateOptions(BASE_ONSET); ducked = false; }
    post({ type: 'sttState', id, state: 'speech' });
  } else if (ev.event === 'speechEnd') {
    if (!ev.audio) {
      // Misfire (too short) — silently return to listening.
      post({ type: 'sttState', id, state: 'listening' });
      return;
    }
    await runTranscription(id, ev.audio, false);
  }
}

/** Transcribe an utterance and emit the result. In auto mode we keep listening afterward; in
 *  ptt mode we signal the session is done. */
async function runTranscription(id: number, audio: Float32Array, isPtt: boolean) {
  transcribing = true;
  post({ type: 'sttState', id, state: 'transcribing' });
  try {
    const text = await transcribe(audio);
    if (activeId !== id) return; // session was stopped mid-transcription
    if (text && (isPtt || !isNoiseTranscript(text))) {
      post({ type: 'sttTranscript', id, text });
    }
  } catch (err) {
    post({ type: 'sttError', id, message: err instanceof Error ? err.message : String(err) });
  } finally {
    transcribing = false;
    // Frames captured while we were transcribing are stale (the user's utterance is over) —
    // discard them so they can't spuriously trigger the VAD when we resume.
    frameQueue = [];
    if (isPtt) {
      post({ type: 'sttStopped', id });
    } else if (activeId === id) {
      resetSileroState();
      vad?.reset();
      post({ type: 'sttState', id, state: 'listening' });
    }
  }
}

/** Stop the session. In ptt mode (or on an explicit flush) transcribe whatever is buffered
 *  before signaling stopped; otherwise just tear down. */
async function stop(flush: boolean | undefined) {
  const id = activeId;
  if (id == null) return;
  if (mode === 'ptt') {
    const audio = pttBuffer.length ? concat(pttBuffer) : null;
    pttBuffer = [];
    if (audio) {
      await runTranscription(id, audio, true);
    } else {
      post({ type: 'sttStopped', id });
    }
  } else {
    // auto mode: an explicit flush transcribes the in-progress utterance (rare); normally we
    // just stop listening.
    if (flush && vad) {
      const ev = vad.flush();
      if (ev.event === 'speechEnd' && ev.audio) {
        await runTranscription(id, ev.audio, false);
      }
    }
    post({ type: 'sttStopped', id });
  }
  activeId = null;
  vad = null;
  frameQueue = [];
}

function concat(frames: Float32Array[]): Float32Array {
  let total = 0;
  for (const f of frames) total += f.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const f of frames) { out.set(f, offset); offset += f.length; }
  return out;
}

function post(msg: Record<string, unknown>) {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (e: MessageEvent<SttWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'frame') { enqueueFrame(msg.data); return; }
    if (msg.type === 'sttLoad') {
      if (isReady) { post({ type: 'sttReady', id: msg.id }); return; }
      if (loadPromise) {
        // Join the in-flight load (progress streams under its id); reply ready under this id.
        await loadPromise;
        post({ type: 'sttReady', id: msg.id });
        return;
      }
      loadPromise = load(msg.id);
      try {
        await loadPromise;
      } finally {
        if (!isReady) loadPromise = null; // failed: allow a retry
      }
      return;
    }
    if (msg.type === 'sttStart') { startSession(msg.id, msg.mode, msg.pauseMs); return; }
    if (msg.type === 'sttMute') {
      muted = msg.muted;
      if (muted) { vad?.reset(); resetSileroState(); frameQueue = []; }
      else if (activeId != null && !transcribing) post({ type: 'sttState', id: activeId, state: 'listening' });
      return;
    }
    if (msg.type === 'sttDucking') {
      // Barge-in: stiffen the VAD while TTS plays, restore the base profile when it stops. The
      // session's pause (redemptionFrames) is untouched — only the onset thresholds change.
      ducked = msg.active;
      if (msg.active) {
        vad?.updateOptions(DUCK_PROFILE);
      } else {
        vad?.updateOptions(BASE_ONSET);
        // Playback ended. If we aren't mid a real (user) utterance, discard whatever the VAD
        // buffered during playback — mostly the assistant's own voice bleeding into the mic — so
        // the next turn starts from a clean slate and isn't transcribed as garbled echo.
        if (vad && !vad.inSpeech) {
          vad.reset();
          resetSileroState();
          frameQueue = [];
        }
      }
      return;
    }
    if (msg.type === 'sttStop') { await stop(msg.flush); return; }
    if (msg.type === 'sttRelease') {
      activeId = null;
      vad = null;
      ducked = false;
      pttBuffer = [];
      frameQueue = [];
      sileroSession = null;
      sileroState = null;
      asr = null;
      isReady = false;
      loadPromise = null; // a later sttLoad must really reload, not join this stale load
      return;
    }
  } catch (err) {
    post({ type: 'sttError', id: (msg as { id?: number }).id ?? activeId ?? -1, message: err instanceof Error ? err.message : String(err) });
  }
};

void VAD_FRAME_SAMPLES; // documents the frame size the worklet must emit
