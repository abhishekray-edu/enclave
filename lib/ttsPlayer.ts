// Streamed PCM player for the offscreen document. Owns the AudioContext + the pcm-processor
// AudioWorklet (public/pcm-player-worklet.js) and feeds it Float32 chunks with capacity-based
// backpressure so playback is gapless and the worklet's ring buffer never overflows.
//
// OFFSCREEN-ONLY: the offscreen document is created with the 'AUDIO_PLAYBACK' reason (see
// entrypoints/background.ts), which lets it play audio without a foreground user gesture.
// PCM only ever reaches here via structured-clone postMessage from tts.worker.ts — it never
// crosses the panel⇄offscreen runtime Port (which is JSON-only).

/** Extension-origin URL for a public/ asset (offscreen doc runs at chrome-extension: origin). */
function extUrl(path: string): string {
  const runtime = (globalThis as { chrome?: { runtime?: { getURL?: (p: string) => string } } }).chrome?.runtime;
  if (runtime?.getURL) return runtime.getURL(path.replace(/^\//, ''));
  return new URL(path, location.href).href;
}

export class TtsPlayer {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private initPromise: Promise<void> | null = null;
  private pending: Float32Array[] = [];
  private availableCapacity = 0;
  private gotInitialCapacity = false;
  private streamEndPending = false;
  private endedCb: (() => void) | null = null;
  private sampleRate = 24000;

  onEnded(cb: () => void) {
    this.endedCb = cb;
  }

  private async ensure(sampleRate: number): Promise<void> {
    if (this.ctx && this.sampleRate === sampleRate) {
      await this.resume();
      return this.initPromise ?? Promise.resolve();
    }
    // A different sample rate means a fresh context.
    if (this.ctx) await this.dispose();
    this.sampleRate = sampleRate;
    this.initPromise = (async () => {
      const ctx = new AudioContext({ sampleRate });
      await ctx.audioWorklet.addModule(extUrl('/pcm-player-worklet.js'));
      const node = new AudioWorkletNode(ctx, 'pcm-processor');
      node.connect(ctx.destination);
      node.port.onmessage = (e: MessageEvent) => this.onWorkletMessage(e.data);
      this.ctx = ctx;
      this.node = node;
      await this.resume();
    })();
    return this.initPromise;
  }

  private async resume(): Promise<void> {
    if (this.ctx && this.ctx.state === 'suspended') {
      try { await this.ctx.resume(); } catch { /* AUDIO_PLAYBACK offscreen docs may auto-run */ }
    }
  }

  private onWorkletMessage(data: { type: string; capacity?: number; buffered?: number; requestSamples?: number }) {
    if (data.type === 'capacity') {
      this.availableCapacity = data.capacity ?? 0;
      if (!this.gotInitialCapacity) {
        this.gotInitialCapacity = true;
        this.flush();
      } else if ((data.requestSamples ?? 0) > 0 && this.pending.length) {
        this.flush();
      }
    } else if (data.type === 'underrun') {
      this.flush();
    } else if (data.type === 'playback-complete') {
      this.endedCb?.();
    }
  }

  /** Send one queued chunk if it fits; capacity zeroes until the next worklet update. */
  private flush() {
    if (!this.node || !this.pending.length || this.availableCapacity <= 0) return;
    const chunk = this.pending[0];
    if (chunk.length <= this.availableCapacity) {
      this.pending.shift();
      this.node.port.postMessage({ type: 'audio', data: chunk }, [chunk.buffer]);
      this.availableCapacity = 0;
    } else if (this.availableCapacity > 4096) {
      const partial = chunk.slice(0, this.availableCapacity);
      this.pending[0] = chunk.slice(this.availableCapacity);
      this.node.port.postMessage({ type: 'audio', data: partial }, [partial.buffer]);
      this.availableCapacity = 0;
    }
    if (!this.pending.length && this.streamEndPending) {
      this.node.port.postMessage({ type: 'stream-ended' });
      this.streamEndPending = false;
    }
  }

  /** Queue a chunk for playback (creating the context on first use). */
  async play(chunk: Float32Array, sampleRate: number): Promise<void> {
    await this.ensure(sampleRate);
    this.pending.push(chunk);
    if (this.gotInitialCapacity && this.availableCapacity > 0) this.flush();
  }

  /** No more chunks are coming; the worklet fires playback-complete once the buffer drains. */
  end() {
    if (!this.node) { this.endedCb?.(); return; }
    if (this.pending.length) this.streamEndPending = true;
    else this.node.port.postMessage({ type: 'stream-ended' });
  }

  /** Stop immediately and clear everything (Stop button / panel close / new utterance). */
  stop() {
    this.pending = [];
    this.streamEndPending = false;
    this.availableCapacity = 0;
    this.node?.port.postMessage({ type: 'reset' });
  }

  /** Tear down the AudioContext + worklet entirely (used by the release-voice-model path). */
  async dispose() {
    this.stop();
    this.gotInitialCapacity = false;
    try { this.node?.disconnect(); } catch { /* ignore */ }
    try { await this.ctx?.close(); } catch { /* ignore */ }
    this.node = null;
    this.ctx = null;
    this.initPromise = null;
  }
}
