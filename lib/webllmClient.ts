// Panel-side WebLLM client. Talks to the offscreen document that hosts the engine.
// Deliberately imports NO @mlc-ai/web-llm code, so the side-panel bundle stays small.
import { browser } from 'wxt/browser';
import type { ChatMessage, Chunk, RetrievedChunk } from './types';

export interface WebllmModelOption {
  id: string;
  label: string;
  /** Approximate memory footprint at base context, in GB (for the RAM-tiered menu). */
  approxGb: number;
  /** Max context (tokens) the build supports, capped for memory sanity. */
  maxCtx: number;
  /**
   * Hard cap on the tokens of any SINGLE prompt we submit to this model (page body +
   * scaffold). Prefill is one near-uninterruptible GPU workload (the compiled
   * prefill_chunk_size is 2048–8192, so a whole prompt can be a single submission);
   * on an integrated GPU a large prefill on a large model can starve the OS compositor
   * past its watchdog (macOS WindowServer kills the session after 40s). Sized so the
   * worst-case prefill stays a few seconds even on a low-power integrated GPU: the
   * bigger the model, the smaller the prompt it is allowed to receive.
   */
  safePromptTokens: number;
  /** Short tier note shown in the picker. */
  note: string;
}

/** RAM-tiered in-browser models — ids, memory, and per-model context caps verified
 *  against WebLLM's prebuilt catalog and each model's mlc-chat-config.
 *  maxCtx bounds the KV-cache the engine preallocates (WebLLM's vram_required_MB is
 *  quoted at 4096 ctx); the 8B models are held to 8192 so weights+KV stay well inside
 *  a 16 GB unified-memory machine. */
export const WEBLLM_MODELS: WebllmModelOption[] = [
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 1B', approxGb: 1.0, maxCtx: 16384, safePromptTokens: 4096, note: 'Lightest & fastest' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC', label: 'Gemma 2 2B', approxGb: 1.9, maxCtx: 4096, safePromptTokens: 2048, note: 'Light' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 3B', approxGb: 2.3, maxCtx: 16384, safePromptTokens: 3072, note: 'Balanced' },
  // Prefill saturates the GPU and starves the OS compositor (frozen animations/scroll), so
  // caps are sized for tolerable freeze windows, not just crash safety. 4B at ~1536 tokens
  // keeps the single prefill slab (compiled chunk 2048) short even in Low Power Mode.
  { id: 'Qwen3-4B-q4f16_1-MLC', label: 'Qwen3 4B', approxGb: 3.4, maxCtx: 16384, safePromptTokens: 1536, note: 'Recommended' },
  { id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC', label: 'Llama 3.1 8B', approxGb: 5.0, maxCtx: 8192, safePromptTokens: 1536, note: 'High quality' },
  { id: 'Qwen3-8B-q4f16_1-MLC', label: 'Qwen3 8B', approxGb: 5.7, maxCtx: 8192, safePromptTokens: 1536, note: 'Best, heaviest' },
];

const DEFAULT_MODEL_ID = 'Qwen3-4B-q4f16_1-MLC';
/** Lighter default suggested when the device reports limited memory. */
const LOW_MEMORY_MODEL_ID = 'Llama-3.2-3B-Instruct-q4f16_1-MLC';

/** Look up a model option by id (falls back to the default model). */
export function webllmModel(id: string): WebllmModelOption {
  return WEBLLM_MODELS.find((m) => m.id === id) ?? WEBLLM_MODELS.find((m) => m.id === DEFAULT_MODEL_ID)!;
}

/** Suggested model for this machine. navigator.deviceMemory is coarse (Chromium caps it at
 *  8 and rounds down), so this only separates "8 GB+" machines from smaller ones — enough to
 *  choose between a 2.3 GB and a 3.4 GB download as the starting point. Only consulted on a
 *  fresh install; a stored model choice always wins. */
export function defaultModelForDevice(): string {
  const mem = (globalThis.navigator as (Navigator & { deviceMemory?: number }) | undefined)?.deviceMemory;
  return mem != null && mem < 8 ? LOW_MEMORY_MODEL_ID : DEFAULT_MODEL_ID;
}

export const PORT_NAME = 'webllm';

/** True when this browser exposes WebGPU (required for in-browser inference). */
export function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export interface LoadProgress {
  text: string;
  progress: number;
}

/** Per-request generation knobs carried over the port. */
export interface GenerateOptions {
  temperature: number;
  maxTokens?: number;
  /** XGrammar-constrained JSON; schema is a stringified JSON Schema. */
  responseFormat?: { type: 'json_object'; schema?: string };
  /** Qwen3 only: disable the <think> phase (used for JSON extraction). */
  enableThinking?: boolean;
  /** Stream deltas (default) vs return one result. JSON extraction uses false. */
  stream?: boolean;
}

// ---- Port protocol between the side panel and the offscreen document ----
export type PanelToOffscreen =
  | { type: 'init'; id: number; model: string; contextWindowSize: number }
  | { type: 'prewarmModel'; id: number; model: string; contextWindowSize: number; supersede?: boolean }
  | { type: 'cancelLoad'; model: string; contextWindowSize: number }
  | { type: 'generate'; id: number; messages: ChatMessage[]; options: GenerateOptions }
  | { type: 'index'; id: number; url: string; contentHash: string; chunks: Chunk[] }
  | { type: 'retrieve'; id: number; url: string; contentHash: string; query: string; topK: number }
  | { type: 'compress'; id: number; texts: string[]; rate: number }
  | { type: 'interrupt'; id?: number }
  // Text-to-speech (pocket-tts). PCM never crosses this port — it plays in the offscreen
  // document; only these small command/status messages travel over the Port.
  | { type: 'ttsLoad'; id: number }
  | { type: 'ttsSpeak'; id: number; text: string }
  | { type: 'ttsStop' }
  | { type: 'ttsRelease' }
  // Speech-to-text (Moonshine + Silero VAD). Mic PCM never crosses this Port either — it is
  // captured and transcribed in the offscreen doc; only these control/status messages travel.
  | { type: 'sttLoad'; id: number }
  | { type: 'sttStart'; id: number; mode: 'ptt' | 'auto' }
  | { type: 'sttMute'; muted: boolean }
  | { type: 'sttStop'; flush?: boolean }
  | { type: 'sttRelease' };

export type OffscreenToPanel =
  | { type: 'progress'; id: number; report: LoadProgress }
  | { type: 'ready'; id: number }
  // Sent once a prewarm decides to stage the model, before the (possibly long) load: it says
  // whether that load is a fast GPU load from cache or attaches to a download already running.
  | { type: 'prewarmStarted'; id: number; downloading: boolean }
  | { type: 'prewarmed'; id: number; loaded: boolean }
  | { type: 'chunk'; id: number; delta: string }
  | { type: 'result'; id: number; content: string }
  | { type: 'done'; id: number }
  | { type: 'error'; id: number; message: string }
  | { type: 'embedProgress'; id: number; report: LoadProgress }
  | { type: 'indexed'; id: number; chunkCount: number; fromCache: boolean }
  | { type: 'retrieved'; id: number; results: RetrievedChunk[] }
  | { type: 'compressed'; id: number; texts: string[] }
  // Text-to-speech status (playback happens in the offscreen doc; no audio data here).
  | { type: 'ttsProgress'; id: number; progress: number }
  | { type: 'ttsReady'; id: number }
  | { type: 'ttsEnded'; id: number }
  | { type: 'ttsError'; id: number; message: string }
  // Speech-to-text status. `id` ties messages to the sttLoad/sttStart request that owns them.
  | { type: 'sttProgress'; id: number; progress: number }
  | { type: 'sttReady'; id: number }
  | { type: 'sttState'; id: number; state: 'listening' | 'speech' | 'transcribing' | 'muted' }
  | { type: 'sttTranscript'; id: number; text: string }
  | { type: 'sttStopped'; id: number }
  | { type: 'sttError'; id: number; message: string };

export type WebllmPort = ReturnType<typeof browser.runtime.connect>;

let nextInitId = 0;

/** Ask the background worker to create the offscreen document if needed. */
export async function ensureOffscreen(): Promise<void> {
  await browser.runtime.sendMessage({ type: 'ENSURE_OFFSCREEN' });
}

/** Ask the background worker to close the offscreen document, freeing the model. */
export async function releaseOffscreen(): Promise<void> {
  await browser.runtime.sendMessage({ type: 'RELEASE_OFFSCREEN' });
}

/** Load (or confirm already-loaded) the model in the offscreen doc; resolves on ready.
 *  Aborting the signal cancels the load itself — including an in-flight weight download,
 *  which resumes later from its cached shards — and rejects with an AbortError. */
export function initModel(
  port: WebllmPort,
  model: string,
  contextWindowSize: number,
  onProgress: (p: LoadProgress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const id = --nextInitId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (msg.id !== id) return;
      if (msg.type === 'progress') onProgress(msg.report);
      else if (msg.type === 'ready') {
        cleanup();
        resolve();
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    const onAbort = () => {
      try {
        port.postMessage({ type: 'cancelLoad', model, contextWindowSize } satisfies PanelToOffscreen);
      } catch {
        /* port already gone */
      }
      cleanup();
      reject(new DOMException('Aborted', 'AbortError'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
      signal?.removeEventListener('abort', onAbort);
    }
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    signal?.addEventListener('abort', onAbort, { once: true });
    port.postMessage({ type: 'init', id, model, contextWindowSize } satisfies PanelToOffscreen);
  });
}

/** Warm the model ahead of the first question: loads it into the GPU only when its weights
 *  are already downloaded (never triggers a multi-GB download). Resolves with whether the
 *  model ended up loaded. Best-effort — callers should swallow rejections.
 *  `supersede` marks an explicit model/context switch: it may cancel a different model's
 *  in-flight load (newest choice wins); a background warm-up must leave that flag off.
 *  `onStart` fires once staging begins, reporting whether it attaches to a running download
 *  (so the caller can show a download vs a quick cache-load state before progress arrives). */
export function prewarmModel(
  port: WebllmPort,
  model: string,
  contextWindowSize: number,
  onProgress: (p: LoadProgress) => void,
  opts?: { supersede?: boolean; onStart?: (downloading: boolean) => void },
): Promise<boolean> {
  const id = --nextInitId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (msg.id !== id) return;
      if (msg.type === 'progress') onProgress(msg.report);
      else if (msg.type === 'prewarmStarted') opts?.onStart?.(msg.downloading);
      else if (msg.type === 'prewarmed') {
        cleanup();
        resolve(msg.loaded);
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({
      type: 'prewarmModel',
      id,
      model,
      contextWindowSize,
      supersede: opts?.supersede,
    } satisfies PanelToOffscreen);
  });
}

/** Stream a generation from the offscreen engine, yielding content deltas. */
export function streamGenerate(
  port: WebllmPort,
  id: number,
  messages: ChatMessage[],
  options: GenerateOptions,
  signal: AbortSignal,
): AsyncGenerator<string, void, unknown> {
  const queue: string[] = [];
  let resolveNext: (() => void) | null = null;
  let finished = false;
  let err: Error | null = null;
  const wake = () => {
    const r = resolveNext;
    resolveNext = null;
    r?.();
  };
  const onMsg = (raw: unknown) => {
    const msg = raw as OffscreenToPanel;
    if ('id' in msg && msg.id !== id) return;
    if (msg.type === 'chunk') {
      queue.push(msg.delta);
      wake();
    } else if (msg.type === 'done') {
      finished = true;
      wake();
    } else if (msg.type === 'error') {
      err = new Error(msg.message);
      finished = true;
      wake();
    }
  };
  const onDisc = () => {
    err = new Error('In-browser model host disconnected.');
    finished = true;
    wake();
  };
  const onAbort = () => {
    try {
      port.postMessage({ type: 'interrupt', id } satisfies PanelToOffscreen);
    } catch {
      /* port already gone */
    }
    finished = true;
    wake();
  };
  port.onMessage.addListener(onMsg);
  port.onDisconnect.addListener(onDisc);
  signal.addEventListener('abort', onAbort, { once: true });
  port.postMessage({ type: 'generate', id, messages, options: { ...options, stream: true } } satisfies PanelToOffscreen);

  return (async function* () {
    try {
      while (true) {
        if (queue.length) {
          yield queue.shift()!;
          continue;
        }
        if (err) throw err;
        if (finished) return;
        await new Promise<void>((r) => {
          resolveNext = r;
        });
      }
    } finally {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
      signal.removeEventListener('abort', onAbort);
    }
  })();
}

/** Ensure the page is chunked + embedded + indexed in the offscreen doc. Idempotent: a fresh
 *  index for {url, contentHash} returns immediately. Uses the negative id space. */
export function indexPage(
  port: WebllmPort,
  url: string,
  contentHash: string,
  chunks: Chunk[],
  onProgress: (p: LoadProgress) => void,
): Promise<{ chunkCount: number; fromCache: boolean }> {
  const id = --nextInitId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (msg.id !== id) return;
      if (msg.type === 'embedProgress') onProgress(msg.report);
      else if (msg.type === 'indexed') {
        cleanup();
        resolve({ chunkCount: msg.chunkCount, fromCache: msg.fromCache });
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'index', id, url, contentHash, chunks } satisfies PanelToOffscreen);
  });
}

/** Retrieve the top-k most relevant chunks for a query from a previously-indexed page. */
export function retrieveChunks(
  port: WebllmPort,
  url: string,
  contentHash: string,
  query: string,
  topK: number,
): Promise<RetrievedChunk[]> {
  const id = --nextInitId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (msg.id !== id) return;
      if (msg.type === 'retrieved') {
        cleanup();
        resolve(msg.results);
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'retrieve', id, url, contentHash, query, topK } satisfies PanelToOffscreen);
  });
}

/** Compress texts with LLMLingua-2 in the offscreen doc (keeps ~rate of the tokens). */
export function compressTexts(port: WebllmPort, texts: string[], rate: number): Promise<string[]> {
  const id = --nextInitId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (msg.id !== id) return;
      if (msg.type === 'compressed') {
        cleanup();
        resolve(msg.texts);
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'compress', id, texts, rate } satisfies PanelToOffscreen);
  });
}

/** Run a single non-streaming generation (JSON extraction, map-reduce calls). Resolves with
 *  the full content; rejects on error/abort. */
export function generateOnce(
  port: WebllmPort,
  id: number,
  messages: ChatMessage[],
  options: GenerateOptions,
  signal: AbortSignal,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (!('id' in msg) || msg.id !== id) return;
      if (msg.type === 'result') {
        cleanup();
        resolve(msg.content);
      } else if (msg.type === 'error') {
        cleanup();
        reject(new Error(msg.message));
      }
    };
    const onDisc = () => {
      cleanup();
      reject(new Error('In-browser model host disconnected.'));
    };
    const onAbort = () => {
      try {
        port.postMessage({ type: 'interrupt', id } satisfies PanelToOffscreen);
      } catch {
        /* port already gone */
      }
      cleanup();
      reject(new DOMException('Aborted', 'AbortError'));
    };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
      signal.removeEventListener('abort', onAbort);
    }
    if (signal.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    signal.addEventListener('abort', onAbort, { once: true });
    port.postMessage({ type: 'generate', id, messages, options: { ...options, stream: false } } satisfies PanelToOffscreen);
  });
}

/** Fast, stable non-crypto content hash (cyrb53) for cache-keying a page's text. */
export function hashText(text: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}
