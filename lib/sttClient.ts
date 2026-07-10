// Panel-side client for local speech-to-text (Moonshine + Silero VAD). Talks to the offscreen
// document over the same runtime Port as the LLM engine. Imports NO ONNX/audio code — capture
// and inference are offscreen-only, so the side-panel bundle stays light (bundle-split invariant),
// exactly like lib/ttsClient.ts.
import { browser } from 'wxt/browser';
import type { OffscreenToPanel, PanelToOffscreen, WebllmPort } from './webllmClient';

/** One-time download size shown in the consent affordance (Moonshine base, English:
 *  q8 encoder ~21 MB + q4 decoder ~73 MB — see MOONSHINE_DTYPE in stt.worker.ts). */
export const STT_DOWNLOAD_MB = 93;

// STT requests use their own id space, kept clear of TTS (>= 1,000,000) and the engine's ids.
let nextSttId = 2_000_000;

export type SttState = 'listening' | 'speech' | 'transcribing' | 'muted';

/** Ensure the speech model is downloaded (once) and loaded in the offscreen worker.
 *  Reports 0..1 download progress; resolves when ready. Mirror of ttsLoad. */
export function sttLoad(port: WebllmPort, onProgress: (p: number) => void): Promise<void> {
  const id = ++nextSttId;
  return new Promise((resolve, reject) => {
    const onMsg = (raw: unknown) => {
      const msg = raw as OffscreenToPanel;
      if (!('id' in msg) || msg.id !== id) return;
      if (msg.type === 'sttProgress') onProgress(msg.progress);
      else if (msg.type === 'sttReady') { cleanup(); resolve(); }
      else if (msg.type === 'sttError') { cleanup(); reject(new Error(msg.message)); }
    };
    const onDisc = () => { cleanup(); reject(new Error('In-browser model host disconnected.')); };
    function cleanup() {
      port.onMessage.removeListener(onMsg);
      port.onDisconnect.removeListener(onDisc);
    }
    port.onMessage.addListener(onMsg);
    port.onDisconnect.addListener(onDisc);
    port.postMessage({ type: 'sttLoad', id } satisfies PanelToOffscreen);
  });
}

export interface SttSession {
  /** Id tying this session's status messages together (also used for muting). */
  id: number;
  /** Stop listening. `flush` transcribes the in-progress utterance first (push-to-talk). */
  stop(flush?: boolean): void;
}

export interface SttListenOptions {
  mode: 'ptt' | 'auto';
  onState?: (state: SttState) => void;
  onTranscript: (text: string) => void;
  onError?: (err: Error) => void;
  onStopped?: () => void;
}

/** Begin a listening session in the offscreen doc. Returns a handle to stop it. The offscreen
 *  document captures the mic and streams transcripts back; call `stop()` to end. */
export function sttStartListening(port: WebllmPort, opts: SttListenOptions): SttSession {
  const id = ++nextSttId;
  const onMsg = (raw: unknown) => {
    const msg = raw as OffscreenToPanel;
    if (!('id' in msg) || msg.id !== id) return;
    if (msg.type === 'sttState') opts.onState?.(msg.state);
    else if (msg.type === 'sttTranscript') opts.onTranscript(msg.text);
    else if (msg.type === 'sttStopped') { cleanup(); opts.onStopped?.(); }
    else if (msg.type === 'sttError') opts.onError?.(new Error(msg.message)); // session continues
  };
  const onDisc = () => { cleanup(); opts.onStopped?.(); };
  function cleanup() {
    port.onMessage.removeListener(onMsg);
    port.onDisconnect.removeListener(onDisc);
  }
  port.onMessage.addListener(onMsg);
  port.onDisconnect.addListener(onDisc);
  port.postMessage({ type: 'sttStart', id, mode: opts.mode } satisfies PanelToOffscreen);
  return {
    id,
    stop(flush?: boolean) {
      try { port.postMessage({ type: 'sttStop', flush } satisfies PanelToOffscreen); } catch { /* port gone */ }
    },
  };
}

/** Mute/unmute the mic without ending the session (half-duplex echo guard during TTS). */
export function sttMute(port: WebllmPort, muted: boolean): void {
  try { port.postMessage({ type: 'sttMute', muted } satisfies PanelToOffscreen); } catch { /* port gone */ }
}

/** Free the speech model's sessions from memory. */
export function sttRelease(port: WebllmPort): void {
  try { port.postMessage({ type: 'sttRelease' } satisfies PanelToOffscreen); } catch { /* port gone */ }
}

/** Current microphone permission for the extension origin. 'unavailable' when the Permissions
 *  API can't answer (older browsers) — callers should then just try to open the permission page. */
export async function micPermissionState(): Promise<PermissionState | 'unavailable'> {
  try {
    const nav = globalThis.navigator as Navigator & {
      permissions?: { query(d: { name: PermissionName }): Promise<PermissionStatus> };
    };
    if (!nav.permissions?.query) return 'unavailable';
    const status = await nav.permissions.query({ name: 'microphone' as PermissionName });
    return status.state;
  } catch {
    return 'unavailable';
  }
}

/** Open the one-time mic-permission page in a normal tab (getUserMedia can't prompt from the
 *  side panel or offscreen document). The page posts a MIC_PERMISSION_RESULT runtime message. */
export async function openMicPermissionPage(): Promise<void> {
  const url = browser.runtime.getURL('/mic-permission.html');
  await browser.tabs.create({ url });
}
