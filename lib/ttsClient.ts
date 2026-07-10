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

/** A streamed speech session: append sentences as the LLM produces them, then finish. Audio
 *  plays as sentences arrive, so the reply is heard while it is still being written. */
export interface TtsStreamSession {
  /** Queue a sentence to speak; returns its sequence number (for chunk-started sync). */
  append(text: string): number;
  /** No more sentences will be appended. `ended` resolves once the last one finishes playing. */
  finish(): void;
  /** Stop immediately (barge-in / new question). Resolves `ended`. */
  cancel(): void;
  /** Resolves when playback of the whole reply finishes, or on error/disconnect. */
  ended: Promise<void>;
  /** Fires when a sentence's audio actually begins playing (carries its append seq). */
  onChunkStarted(cb: (seq: number) => void): void;
  /** Fires if synthesis errors mid-session. */
  onError(cb: (err: Error) => void): void;
}

/** Open a streamed speech session over the port. Sentences appended via the returned handle are
 *  spoken in order as they arrive. */
export function ttsSpeakStream(port: WebllmPort): TtsStreamSession {
  const id = ++nextTtsId;
  let seqCounter = 0;
  let chunkCb: ((seq: number) => void) | null = null;
  let errorCb: ((err: Error) => void) | null = null;
  let resolveEnded: () => void = () => {};
  const ended = new Promise<void>((res) => {
    resolveEnded = res;
  });
  const onMsg = (raw: unknown) => {
    const msg = raw as OffscreenToPanel;
    if (!('id' in msg) || msg.id !== id) return;
    if (msg.type === 'ttsChunkStarted') chunkCb?.(msg.seq);
    else if (msg.type === 'ttsEnded') { cleanup(); resolveEnded(); }
    else if (msg.type === 'ttsError') { cleanup(); errorCb?.(new Error(msg.message)); resolveEnded(); }
  };
  const onDisc = () => { cleanup(); resolveEnded(); /* panel gone: treat as ended */ };
  function cleanup() {
    port.onMessage.removeListener(onMsg);
    port.onDisconnect.removeListener(onDisc);
  }
  port.onMessage.addListener(onMsg);
  port.onDisconnect.addListener(onDisc);
  port.postMessage({ type: 'ttsSpeakStream', id } satisfies PanelToOffscreen);

  return {
    append(text: string): number {
      const seq = seqCounter++;
      try { port.postMessage({ type: 'ttsAppend', id, seq, text } satisfies PanelToOffscreen); } catch { /* port gone */ }
      return seq;
    },
    finish() {
      try { port.postMessage({ type: 'ttsFinish', id } satisfies PanelToOffscreen); } catch { /* port gone */ }
    },
    cancel() {
      try { port.postMessage({ type: 'ttsStop' } satisfies PanelToOffscreen); } catch { /* port gone */ }
      cleanup();
      resolveEnded();
    },
    ended,
    onChunkStarted(cb) { chunkCb = cb; },
    onError(cb) { errorCb = cb; },
  };
}

/** Stop any in-flight speech (halts generation + playback in the offscreen doc). */
export function ttsStop(port: WebllmPort): void {
  try { port.postMessage({ type: 'ttsStop' } satisfies PanelToOffscreen); } catch { /* port gone */ }
}

/** Free the voice model's ONNX sessions from memory. */
export function ttsRelease(port: WebllmPort): void {
  try { port.postMessage({ type: 'ttsRelease' } satisfies PanelToOffscreen); } catch { /* port gone */ }
}
