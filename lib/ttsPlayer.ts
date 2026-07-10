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
  private pending: { data: Float32Array; seq?: number }[] = [];
  private availableCapacity = 0;
  private gotInitialCapacity = false;
  private streamEndPending = false;
  private endedCb: (() => void) | null = null;
  private chunkStartedCb: ((seq: number) => void) | null = null;
  private sampleRate = 24000;
  // Per-sentence sync: as samples are written to the worklet we record the sample index at which
  // each `seq` first appears; when the worklet reports the playhead crossing that index, the
  // sentence's audio has started and we fire onChunkStarted(seq).
  private totalWrittenSamples = 0;
  private lastWrittenSeq: number | undefined = undefined;
  private watermarks: { seq: number; startSample: number }[] = [];

  onEnded(cb: () => void) {
    this.endedCb = cb;
  }

  onChunkStarted(cb: (seq: number) => void) {
    this.chunkStartedCb = cb;
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

  private onWorkletMessage(data: {
    type: string;
    capacity?: number;
    buffered?: number;
    requestSamples?: number;
    totalReadSamples?: number;
  }) {
    if (data.type === 'capacity') {
      this.availableCapacity = data.capacity ?? 0;
      this.fireWatermarks(data.totalReadSamples ?? 0);
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

  /** Fire onChunkStarted for every sentence whose first sample the playhead has now reached. */
  private fireWatermarks(totalReadSamples: number) {
    while (this.watermarks.length && totalReadSamples >= Math.max(1, this.watermarks[0].startSample)) {
      const wm = this.watermarks.shift()!;
      this.chunkStartedCb?.(wm.seq);
    }
  }

  /** Send one buffer to the worklet, recording a watermark when its `seq` first appears. */
  private writeToNode(data: Float32Array, seq: number | undefined) {
    if (seq !== undefined && seq !== this.lastWrittenSeq) {
      this.watermarks.push({ seq, startSample: this.totalWrittenSamples });
      this.lastWrittenSeq = seq;
    }
    this.totalWrittenSamples += data.length;
    this.node!.port.postMessage({ type: 'audio', data }, [data.buffer]);
  }

  /** Send one queued chunk if it fits; capacity zeroes until the next worklet update. */
  private flush() {
    if (!this.node || !this.pending.length || this.availableCapacity <= 0) return;
    const chunk = this.pending[0];
    if (chunk.data.length <= this.availableCapacity) {
      this.pending.shift();
      this.writeToNode(chunk.data, chunk.seq);
      this.availableCapacity = 0;
    } else if (this.availableCapacity > 4096) {
      const partial = chunk.data.slice(0, this.availableCapacity);
      this.pending[0] = { data: chunk.data.slice(this.availableCapacity), seq: chunk.seq };
      this.writeToNode(partial, chunk.seq);
      this.availableCapacity = 0;
    }
    if (!this.pending.length && this.streamEndPending) {
      this.node.port.postMessage({ type: 'stream-ended' });
      this.streamEndPending = false;
    }
  }

  /** Queue a chunk for playback (creating the context on first use). `seq` (when given) ties the
   *  chunk to a streamed sentence for text/audio sync. */
  async play(chunk: Float32Array, sampleRate: number, seq?: number): Promise<void> {
    await this.ensure(sampleRate);
    this.pending.push({ data: chunk, seq });
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
    this.totalWrittenSamples = 0;
    this.lastWrittenSeq = undefined;
    this.watermarks = [];
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
