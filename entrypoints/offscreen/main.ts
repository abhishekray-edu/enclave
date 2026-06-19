// Offscreen document: hosts the WebLLM engine so the model stays resident in
// memory across side-panel open/close. The panel talks to it over a runtime Port.
import { browser } from 'wxt/browser';
import type { MLCEngine } from '@mlc-ai/web-llm';
import { createWebllmEngine, chatStreamWebllm } from '@/lib/webllm';
import { PORT_NAME, type OffscreenToPanel, type PanelToOffscreen, type WebllmPort } from '@/lib/webllmClient';

let engine: MLCEngine | null = null;
let loadedModel: string | null = null;
let loadedCtx: number | null = null;
let loadQueue: Promise<void> = Promise.resolve();
let abort: AbortController | null = null;

function errStr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function send(port: WebllmPort, msg: OffscreenToPanel) {
  try {
    port.postMessage(msg);
  } catch {
    /* port closed (panel went away) — keep the engine loaded regardless */
  }
}

/** Ensure the requested model + context is loaded, reusing it if already resident. */
async function ensureModel(
  model: string,
  ctx: number,
  onProgress: (text: string, progress: number) => void,
) {
  const load = async () => {
    if (engine && loadedModel === model && loadedCtx === ctx) return;
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU is not available, so the in-browser engine cannot run.');
    }
    if (engine) {
      try {
        await engine.unload();
      } catch {
        /* ignore */
      }
      engine = null;
      loadedModel = null;
      loadedCtx = null;
    }

    const nextEngine = await createWebllmEngine(model, ctx, (r) => onProgress(r.text, r.progress));
    engine = nextEngine;
    loadedModel = model;
    loadedCtx = ctx;
  };

  const queuedLoad = loadQueue.then(load, load);
  loadQueue = queuedLoad.catch(() => {});
  await queuedLoad;
}

browser.runtime.onConnect.addListener((port) => {
  if (port.name !== PORT_NAME) return;

  port.onMessage.addListener(async (raw) => {
    const msg = raw as PanelToOffscreen;

    if (msg.type === 'init') {
      try {
        await ensureModel(msg.model, msg.contextWindowSize, (text, progress) =>
          send(port, { type: 'progress', id: msg.id, report: { text, progress } }),
        );
        send(port, { type: 'ready', id: msg.id });
      } catch (e) {
        send(port, { type: 'error', id: msg.id, message: errStr(e) });
      }
      return;
    }

    if (msg.type === 'generate') {
      if (!engine) {
        send(port, { type: 'error', id: msg.id, message: 'Model not loaded.' });
        return;
      }
      abort = new AbortController();
      try {
        for await (const delta of chatStreamWebllm({
          engine,
          messages: msg.messages,
          temperature: msg.temperature,
          signal: abort.signal,
        })) {
          send(port, { type: 'chunk', id: msg.id, delta });
        }
        send(port, { type: 'done', id: msg.id });
      } catch (e) {
        send(port, { type: 'error', id: msg.id, message: errStr(e) });
      } finally {
        abort = null;
      }
      return;
    }

    if (msg.type === 'interrupt') {
      abort?.abort();
      try {
        engine?.interruptGenerate();
      } catch {
        /* ignore */
      }
    }
  });
});
