// Offscreen document: hosts the WebLLM engine so the model stays resident in
// memory across side-panel open/close. The panel talks to it over a runtime Port.
import { browser } from 'wxt/browser';
import type { WebWorkerMLCEngine } from '@mlc-ai/web-llm';
import { createWebllmEngine, chatStreamWebllm, chatCompleteWebllm, isModelCached } from '@/lib/webllm';
import { createModelLoader } from '@/lib/modelLoader';
import { PORT_NAME, type OffscreenToPanel, type PanelToOffscreen, type WebllmPort } from '@/lib/webllmClient';
// NOTE: lib/retrieval (Transformers.js + onnxruntime) and lib/compress (LLMLingua-2) run in
// a dedicated worker (ml.worker.ts, created lazily) so a plain Q&A never loads that heavy
// machinery AND embedding never blocks this document's main thread — the side panel usually
// shares it, so a busy main thread here freezes the panel UI.

/** Owns the single resident engine. Loads never queue: an identical request joins the one
 *  in flight, a different one cancels it (newest wins — a canceled download resumes later
 *  from its cached shards, so nothing is lost). */
const loader = createModelLoader<WebWorkerMLCEngine, Worker>({
  // The engine lives in a dedicated worker: weight deserialization and generation would
  // otherwise run on this document's main thread, which the side panel usually shares —
  // freezing its UI for the duration.
  spawnWorker: () => new Worker(new URL('./webllm.worker.ts', import.meta.url), { type: 'module' }),
  terminateWorker: (worker) => worker.terminate(),
  createEngine: (worker, model, ctx, onProgress) =>
    createWebllmEngine(worker, model, ctx, (r) => onProgress(r.text, r.progress)),
  unloadEngine: (engine) => engine.unload(),
});

let abort: AbortController | null = null;
let activeId: number | null = null;

function errStr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// ---- CPU-side ML worker (embedding / retrieval / compression) ----
let mlWorker: Worker | null = null;

function ensureMlWorker(): Worker {
  if (!mlWorker) {
    mlWorker = new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' });
  }
  return mlWorker;
}

/** Send one request to the ML worker and relay every response for that id to the port
 *  until a terminal message (result or error) arrives. */
function relayToMlWorker(
  port: WebllmPort,
  msg: PanelToOffscreen & { id: number },
  terminal: OffscreenToPanel['type'][],
): void {
  const worker = ensureMlWorker();
  const onMsg = (e: MessageEvent<OffscreenToPanel>) => {
    const out = e.data;
    if (!('id' in out) || out.id !== msg.id) return;
    send(port, out);
    if (terminal.includes(out.type) || out.type === 'error') {
      worker.removeEventListener('message', onMsg);
    }
  };
  worker.addEventListener('message', onMsg);
  worker.postMessage(msg);
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
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available, so the in-browser engine cannot run.');
  }
  await loader.ensure(model, ctx, onProgress);
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

    if (msg.type === 'prewarmModel') {
      // Warm start: load only when the weights are already on disk — a prewarm must never
      // kick off a multi-GB download the user didn't ask for. And only an explicit model
      // switch (supersede) may displace a different model that is still mid-load; a
      // background warm-up never cancels a download the user started.
      try {
        const cached = await isModelCached(msg.model);
        const busyWithOther =
          loader.loading !== null &&
          !(loader.loading.model === msg.model && loader.loading.ctx === msg.contextWindowSize);
        const loadable = cached && (!busyWithOther || msg.supersede === true);
        if (loadable) {
          await ensureModel(msg.model, msg.contextWindowSize, (text, progress) =>
            send(port, { type: 'progress', id: msg.id, report: { text, progress } }),
          );
        }
        send(port, { type: 'prewarmed', id: msg.id, loaded: loadable });
      } catch (e) {
        send(port, { type: 'error', id: msg.id, message: errStr(e) });
      }
      return;
    }

    if (msg.type === 'cancelLoad') {
      // Stop an in-flight load/download (the panel's Stop, or a retarget). Already-fetched
      // weight shards stay in the browser cache, so a canceled download resumes later.
      loader.cancel(msg.model, msg.contextWindowSize);
      return;
    }

    if (msg.type === 'generate') {
      const engine = loader.engine;
      if (!engine) {
        send(port, { type: 'error', id: msg.id, message: 'Model not loaded.' });
        return;
      }
      abort = new AbortController();
      activeId = msg.id;
      try {
        if (msg.options.stream === false) {
          const content = await chatCompleteWebllm({
            engine,
            messages: msg.messages,
            options: msg.options,
            signal: abort.signal,
          });
          send(port, { type: 'result', id: msg.id, content });
          send(port, { type: 'done', id: msg.id });
        } else {
          for await (const delta of chatStreamWebllm({
            engine,
            messages: msg.messages,
            options: msg.options,
            signal: abort.signal,
          })) {
            send(port, { type: 'chunk', id: msg.id, delta });
          }
          send(port, { type: 'done', id: msg.id });
        }
      } catch (e) {
        send(port, { type: 'error', id: msg.id, message: errStr(e) });
      } finally {
        abort = null;
        activeId = null;
      }
      return;
    }

    if (msg.type === 'index') {
      relayToMlWorker(port, msg, ['indexed']);
      return;
    }

    if (msg.type === 'retrieve') {
      relayToMlWorker(port, msg, ['retrieved']);
      return;
    }

    if (msg.type === 'compress') {
      relayToMlWorker(port, msg, ['compressed']);
      return;
    }

    if (msg.type === 'interrupt') {
      // Only interrupt the request the panel meant to stop (or any, when id is omitted).
      if (msg.id === undefined || msg.id === activeId) {
        abort?.abort();
        try {
          loader.engine?.interruptGenerate();
        } catch {
          /* ignore */
        }
      }
    }
  });
});
