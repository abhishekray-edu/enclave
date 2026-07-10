// Pure, ONNX-free voice-activity state machine. It turns a stream of (frame, speechProbability)
// pairs into utterance boundaries: speechStart when the probability crosses the positive
// threshold, speechEnd after enough consecutive sub-threshold ("redemption") frames. It buffers
// the utterance's audio — plus a short pre-speech pad so word onsets aren't clipped — and hands
// the concatenated Float32 back on speechEnd.
//
// The semantics are ported from @ricky0123/vad-web's frame processor (MIT); the Silero
// probability is supplied by the caller (stt.worker.ts runs the ONNX model), which keeps this
// module dependency-free and unit-testable (see lib/__tests__/vadFrameProcessor.test.ts).

/** One Silero step per 512 samples at 16 kHz ≈ 31.25 frames/sec. */
export const VAD_FRAME_SAMPLES = 512;
const FRAMES_PER_SEC = 16000 / VAD_FRAME_SAMPLES;

export interface VadOptions {
  /** Probability at/above which a frame is speech (starts an utterance). */
  positiveSpeechThreshold: number;
  /** Probability below which a frame counts toward ending the utterance. */
  negativeSpeechThreshold: number;
  /** Consecutive sub-threshold frames tolerated before the utterance ends. The default (~1.2 s)
   *  is overridden per-session from the user's "pause before responding" setting. */
  redemptionFrames: number;
  /** Frames of retained audio before the first speech frame (~onset padding). */
  preSpeechPadFrames: number;
  /** Utterances with fewer than this many positive frames are discarded as noise. */
  minSpeechFrames: number;
  /** Hard cap on utterance length; forces an end so a stuck VAD can't buffer forever (~30 s). */
  maxSpeechFrames: number;
  /** Consecutive positive frames required before an utterance is confirmed to have started.
   *  1 = start on the first positive frame (default). Raised during TTS playback (barge-in) so a
   *  cough or the tail of the assistant's own voice can't trigger a false interruption. */
  startConfirmFrames: number;
}

export const DEFAULT_VAD_OPTIONS: VadOptions = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionFrames: 38, // ~1.2 s at 31.25 frames/sec
  preSpeechPadFrames: 8,
  minSpeechFrames: 4,
  maxSpeechFrames: Math.round(30 * FRAMES_PER_SEC),
  startConfirmFrames: 1,
};

/** Convert a "pause before responding" duration (ms) to VAD redemption frames. */
export function pauseMsToRedemptionFrames(pauseMs: number): number {
  return Math.max(1, Math.round((pauseMs / 1000) * FRAMES_PER_SEC));
}

export type VadEvent =
  | { event: 'none' }
  | { event: 'speechStart' }
  /** audio present → a real utterance to transcribe; audio undefined → a discarded misfire. */
  | { event: 'speechEnd'; audio?: Float32Array };

function concatFrames(frames: Float32Array[]): Float32Array {
  let total = 0;
  for (const f of frames) total += f.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const f of frames) {
    out.set(f, offset);
    offset += f.length;
  }
  return out;
}

export class VadFrameProcessor {
  private opts: VadOptions;
  private speaking = false;
  private redemption = 0;
  private positiveFrames = 0;
  private speechFrames: Float32Array[] = [];
  /** Rolling window of the most recent frames while idle (the pre-speech pad). */
  private preBuffer: Float32Array[] = [];
  /** Positive frames seen since a potential onset, awaiting confirmation (startConfirmFrames). */
  private pendingStart: Float32Array[] = [];
  private pendingPositive = 0;

  constructor(opts: Partial<VadOptions> = {}) {
    this.opts = { ...DEFAULT_VAD_OPTIONS, ...opts };
  }

  /** Adjust options mid-session (e.g. raise thresholds during TTS playback for barge-in).
   *  Applies from the next frame; does not disturb an utterance already in progress. */
  updateOptions(patch: Partial<VadOptions>): void {
    this.opts = { ...this.opts, ...patch };
  }

  /** True while an utterance is being buffered (speechStart fired, speechEnd hasn't). */
  get inSpeech(): boolean {
    return this.speaking;
  }

  /** Feed one frame and its Silero speech probability. */
  process(frame: Float32Array, prob: number): VadEvent {
    const o = this.opts;
    if (!this.speaking) {
      if (prob >= o.positiveSpeechThreshold) {
        this.pendingStart.push(frame);
        this.pendingPositive++;
        if (this.pendingPositive >= o.startConfirmFrames) {
          // Confirmed: utterance = retained pre-speech pad + the confirming frames.
          this.speaking = true;
          this.redemption = 0;
          this.positiveFrames = this.pendingPositive;
          this.speechFrames = [...this.preBuffer, ...this.pendingStart];
          this.preBuffer = [];
          this.pendingStart = [];
          this.pendingPositive = 0;
          return { event: 'speechStart' };
        }
        return { event: 'none' }; // still confirming the onset
      }
      // Non-positive frame while a burst was being confirmed.
      if (this.pendingStart.length) {
        if (prob < o.negativeSpeechThreshold) {
          // The burst wasn't sustained speech (cough / TTS transient): fold it back into the pad.
          for (const f of this.pendingStart) {
            this.preBuffer.push(f);
            if (this.preBuffer.length > o.preSpeechPadFrames) this.preBuffer.shift();
          }
          this.pendingStart = [];
          this.pendingPositive = 0;
        } else {
          // Between thresholds: keep it as part of the potential onset, don't count or collapse.
          this.pendingStart.push(frame);
          return { event: 'none' };
        }
      }
      // Idle frame: keep it in the rolling pre-speech pad so a following onset isn't clipped.
      this.preBuffer.push(frame);
      if (this.preBuffer.length > o.preSpeechPadFrames) this.preBuffer.shift();
      return { event: 'none' };
    }

    // In speech: keep buffering.
    this.speechFrames.push(frame);
    if (prob >= o.positiveSpeechThreshold) {
      this.positiveFrames++;
      this.redemption = 0;
    } else if (prob < o.negativeSpeechThreshold) {
      this.redemption++;
      if (this.redemption >= o.redemptionFrames) return this.end(false);
    } else {
      // Between thresholds: neither confirms speech nor counts toward ending.
      this.redemption = 0;
    }

    if (this.speechFrames.length >= o.maxSpeechFrames) return this.end(true);
    return { event: 'none' };
  }

  /** Force-end the current utterance (e.g. voice mode toggled off mid-speech). Returns its
   *  audio regardless of length, or 'none' if nothing was being spoken. */
  flush(): VadEvent {
    if (!this.speaking) return { event: 'none' };
    return this.end(true);
  }

  /** Drop all buffered audio and return to idle (e.g. mic muted during TTS playback). */
  reset(): void {
    this.speaking = false;
    this.redemption = 0;
    this.positiveFrames = 0;
    this.speechFrames = [];
    this.preBuffer = [];
    this.pendingStart = [];
    this.pendingPositive = 0;
  }

  /** Finish the current utterance. When ended via redemption, trim the trailing silence and
   *  discard the whole thing if it was too short to be real speech (a misfire). A forced end
   *  (max-duration or flush) keeps the audio verbatim. */
  private end(forced: boolean): VadEvent {
    const o = this.opts;
    const positiveFrames = this.positiveFrames;
    let frames = this.speechFrames;
    if (!forced && this.redemption > 0) {
      frames = frames.slice(0, Math.max(0, frames.length - this.redemption));
    }
    this.reset();
    if (!forced && positiveFrames < o.minSpeechFrames) return { event: 'speechEnd' };
    if (!frames.length) return { event: 'speechEnd' };
    return { event: 'speechEnd', audio: concatFrames(frames) };
  }
}
