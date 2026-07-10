// Microphone capture for the offscreen document (mirror of lib/ttsPlayer.ts, but the input
// side). Owns the AudioContext + the mic-capture AudioWorklet (public/mic-capture-worklet.js)
// and emits fixed 512-sample Float32 frames at 16 kHz to a callback, which stt.worker.ts runs
// through Silero VAD + Moonshine.
//
// OFFSCREEN-ONLY: getUserMedia can't prompt from a side panel or offscreen document, so the mic
// permission is granted once via the mic-permission page (a normal tab). After that the
// offscreen document — created with the 'USER_MEDIA' reason (see entrypoints/background.ts) —
// may capture without a foreground gesture. PCM only ever reaches here; it never crosses the
// panel⇄offscreen runtime Port (which is JSON-only).

/** Extension-origin URL for a public/ asset (offscreen doc runs at the chrome-extension: origin). */
function extUrl(path: string): string {
  const runtime = (globalThis as { chrome?: { runtime?: { getURL?: (p: string) => string } } }).chrome?.runtime;
  if (runtime?.getURL) return runtime.getURL(path.replace(/^\//, ''));
  return new URL(path, location.href).href;
}

export class MicCapture {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  /** When muted, frames are dropped here (the track stays live, so no Chrome mic churn). */
  private muted = false;

  get isRunning(): boolean {
    return this.ctx != null;
  }

  /** Begin capturing. `onFrame` receives 512-sample Float32 frames at 16 kHz. Idempotent-ish:
   *  a second call while already running is a no-op (the existing capture keeps feeding). */
  async start(onFrame: (frame: Float32Array) => void): Promise<void> {
    if (this.ctx) return;
    // 16 kHz context → Chrome resamples the mic for us, so frames are already at Silero/Moonshine
    // rate. Mono + the browser's own echo/noise/gain cleanup (a first line of defense against the
    // speaker feeding back into the mic; the authoritative echo guard is muting during TTS).
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule(extUrl('/mic-capture-worklet.js'));
    if (ctx.state === 'suspended') {
      try { await ctx.resume(); } catch { /* USER_MEDIA offscreen docs may auto-run */ }
    }
    const source = ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(ctx, 'mic-capture-processor');
    node.port.onmessage = (e: MessageEvent) => {
      const data = e.data as { type: string; data?: Float32Array };
      if (data.type === 'frame' && data.data && !this.muted) onFrame(data.data);
    };
    source.connect(node);
    // The worklet has no audio output; connecting to the destination just keeps it pulling.
    node.connect(ctx.destination);
    this.ctx = ctx;
    this.node = node;
    this.source = source;
  }

  /** Gate frames on/off without touching the track — so the Chrome mic indicator doesn't flicker
   *  and getUserMedia isn't re-prompted every time TTS plays. */
  setMuted(muted: boolean): void {
    this.muted = muted;
  }

  /** Stop capturing and release the mic. Stopping the tracks is what makes the Chrome mic
   *  indicator disappear, so this must run when voice mode ends or the panel closes. */
  async stop(): Promise<void> {
    this.muted = false;
    try { this.node?.disconnect(); } catch { /* ignore */ }
    try { this.source?.disconnect(); } catch { /* ignore */ }
    this.stream?.getTracks().forEach((t) => t.stop());
    try { await this.ctx?.close(); } catch { /* ignore */ }
    this.node = null;
    this.source = null;
    this.stream = null;
    this.ctx = null;
  }
}
