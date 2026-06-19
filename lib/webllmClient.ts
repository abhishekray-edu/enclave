// Panel-side WebLLM client. Talks to the offscreen document that hosts the engine.
// Deliberately imports NO @mlc-ai/web-llm code, so the side-panel bundle stays small.
import { browser } from 'wxt/browser';
import type { ChatMessage } from './types';

export interface WebllmModelOption {
  id: string;
  label: string;
  /** Approximate memory footprint at base context, in GB (for the RAM-tiered menu). */
  approxGb: number;
  /** Max context (tokens) the build supports, capped for memory sanity. */
  maxCtx: number;
  /** Short tier note shown in the picker. */
  note: string;
}

/** RAM-tiered in-browser models — ids, memory, and per-model context caps verified
 *  against WebLLM's prebuilt catalog and each model's mlc-chat-config. */
export const WEBLLM_MODELS: WebllmModelOption[] = [
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 1B', approxGb: 1.0, maxCtx: 40960, note: 'Lightest & fastest' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC', label: 'Gemma 2 2B', approxGb: 1.9, maxCtx: 4096, note: 'Light' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 3B', approxGb: 2.3, maxCtx: 40960, note: 'Balanced' },
  { id: 'Qwen3-4B-q4f16_1-MLC', label: 'Qwen3 4B', approxGb: 3.4, maxCtx: 40960, note: 'Recommended' },
  { id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC', label: 'Llama 3.1 8B', approxGb: 5.0, maxCtx: 40960, note: 'High quality' },
  { id: 'Qwen3-8B-q4f16_1-MLC', label: 'Qwen3 8B', approxGb: 5.7, maxCtx: 40960, note: 'Best, heaviest' },
];

/** Look up a model option by id (falls back to the default model). */
export function webllmModel(id: string): WebllmModelOption {
  return WEBLLM_MODELS.find((m) => m.id === id) ?? WEBLLM_MODELS[3];
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

// ---- Port protocol between the side panel and the offscreen document ----
export type PanelToOffscreen =
  | { type: 'init'; id: number; model: string; contextWindowSize: number }
  | { type: 'generate'; id: number; messages: ChatMessage[]; temperature: number }
  | { type: 'interrupt' };

export type OffscreenToPanel =
  | { type: 'progress'; id: number; report: LoadProgress }
  | { type: 'ready'; id: number }
  | { type: 'chunk'; id: number; delta: string }
  | { type: 'done'; id: number }
  | { type: 'error'; id: number; message: string };

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

/** Load (or confirm already-loaded) the model in the offscreen doc; resolves on ready. */
export function initModel(
  port: WebllmPort,
  model: string,
  contextWindowSize: number,
  onProgress: (p: LoadProgress) => void,
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
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'init', id, model, contextWindowSize } satisfies PanelToOffscreen);
  });
}

/** Stream a generation from the offscreen engine, yielding content deltas. */
export function streamGenerate(
  port: WebllmPort,
  id: number,
  messages: ChatMessage[],
  temperature: number,
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
      port.postMessage({ type: 'interrupt' } satisfies PanelToOffscreen);
    } catch {
      /* port already gone */
    }
    finished = true;
    wake();
  };
  port.onMessage.addListener(onMsg);
  port.onDisconnect.addListener(onDisc);
  signal.addEventListener('abort', onAbort, { once: true });
  port.postMessage({ type: 'generate', id, messages, temperature } satisfies PanelToOffscreen);

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
