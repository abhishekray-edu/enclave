// Panel-side client for pocket-tts text-to-speech. Talks to the offscreen document over the
// same runtime Port as the LLM engine. Imports NO ONNX/audio code — all of that is
// offscreen-only, so the side-panel bundle stays light (bundle-split invariant).
import type { OffscreenToPanel, PanelToOffscreen, WebllmPort } from './webllmClient';

/** One-time download size shown in the consent affordance (English, single alba voice). */
export const TTS_DOWNLOAD_MB = 132;

// TTS requests use their own id space (kept clear of webllmClient's init/generate ids).
let nextTtsId = 1_000_000;

/** Ensure the voice model is downloaded (once) and loaded in the offscreen worker.
 *  Reports 0..1 download progress; resolves when the model is ready. */
export function ttsLoad(port: WebllmPort, onProgress: (p: number) => void): Promise<void> {
  const id = ++nextTtsId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (!('id' in msg) || msg.id !== id) return;
      if (msg.type === 'ttsProgress') onProgress(msg.progress);
      else if (msg.type === 'ttsReady') { cleanup(); resolve(); }
      else if (msg.type === 'ttsError') { cleanup(); reject(new Error(msg.message)); }
    };
    const onDisc = () => { cleanup(); reject(new Error('In-browser model host disconnected.')); };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'ttsLoad', id } satisfies PanelToOffscreen);
  });
}

/** Speak `text`. Resolves when playback finishes (or is stopped); rejects on error.
 *  Call ttsStop(port) to interrupt. */
export function ttsSpeak(port: WebllmPort, text: string): Promise<void> {
  const id = ++nextTtsId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (!('id' in msg) || msg.id !== id) return;
      if (msg.type === 'ttsEnded') { cleanup(); resolve(); }
      else if (msg.type === 'ttsError') { cleanup(); reject(new Error(msg.message)); }
    };
    const onDisc = () => { cleanup(); resolve(); /* panel gone: treat as stopped */ };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'ttsSpeak', id, text } satisfies PanelToOffscreen);
  });
}

/** Stop any in-flight speech (halts generation + playback in the offscreen doc). */
export function ttsStop(port: WebllmPort): void {
  try { port.postMessage({ type: 'ttsStop' } satisfies PanelToOffscreen); } catch { /* port gone */ }
}

/** Free the voice model's ONNX sessions from memory. */
export function ttsRelease(port: WebllmPort): void {
  try { port.postMessage({ type: 'ttsRelease' } satisfies PanelToOffscreen); } catch { /* port gone */ }
}
