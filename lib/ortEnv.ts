// Point onnxruntime-web (the Transformers.js backend) at the runtime files bundled with
// the extension. By default it dynamic-imports its ES-module loader from cdn.jsdelivr.net,
// which the extension CSP (script-src 'self') blocks — leaving RAG/compression dead with
// "no available backend found". The files are copied into <output>/ort/ by a build hook
// in wxt.config.ts. OFFSCREEN-ONLY (imports @huggingface/transformers).
import { env } from '@huggingface/transformers';

let configured = false;

/** Extension-origin URL of the bundled ORT runtime dir. chrome.runtime is unavailable in
 *  dedicated workers, so fall back to resolving against the worker script's own URL. */
function ortBaseUrl(): string | null {
  const runtime = (globalThis as { chrome?: { runtime?: { getURL?: (p: string) => string } } }).chrome?.runtime;
  if (runtime?.getURL) return runtime.getURL('ort/');
  if (typeof location !== 'undefined' && location.protocol === 'chrome-extension:') {
    return new URL('/ort/', location.href).href;
  }
  return null;
}

export function configureOrtRuntime(): void {
  if (configured) return;
  const base = ortBaseUrl();
  const wasm = env.backends?.onnx?.wasm as { wasmPaths?: string; numThreads?: number } | undefined;
  if (!base || !wasm) return;
  wasm.wasmPaths = base;
  // Extension pages are not crossOriginIsolated, so wasm threads can't engage anyway;
  // pinning to 1 avoids the threaded worker probe.
  wasm.numThreads = 1;
  configured = true;
}
