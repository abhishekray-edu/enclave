// Panel-side WebLLM client. Talks to the offscreen document that hosts the engine.
// Deliberately imports NO @mlc-ai/web-llm code, so the side-panel bundle stays small.
import { browser } from 'wxt/browser';
import type { ChatMessage } from './types';

/** Curated in-browser models (ids verified against WebLLM's prebuilt catalog). */
export const WEBLLM_MODELS: Array<{ id: string; label: string }> = [
  { id: 'Qwen3-4B-q4f16_1-MLC', label: 'Qwen3 4B — recommended' },
  { id: 'gemma-2-2b-it-q4f16_1-MLC', label: 'Gemma 2 2B — lightest' },
  { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', label: 'Qwen2.5 3B' },
  { id: 'Llama-3.1-8B-Instruct-q4f16_1-MLC', label: 'Llama 3.1 8B — best, heavier' },
  { id: 'Phi-3.5-mini-instruct-q4f16_1-MLC', label: 'Phi-3.5 mini' },
];

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
  | { type: 'init'; model: string; contextWindowSize: number }
  | { type: 'generate'; id: number; messages: ChatMessage[]; temperature: number }
  | { type: 'interrupt' };

export type OffscreenToPanel =
  | { type: 'progress'; report: LoadProgress }
  | { type: 'ready' }
  | { type: 'chunk'; id: number; delta: string }
  | { type: 'done'; id: number }
  | { type: 'error'; id?: number; message: string };

export type WebllmPort = ReturnType<typeof browser.runtime.connect>;

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
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
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
    port.postMessage({ type: 'init', model, contextWindowSize } satisfies PanelToOffscreen);
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
