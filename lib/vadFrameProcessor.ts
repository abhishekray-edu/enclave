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
  /** Consecutive sub-threshold frames tolerated before the utterance ends (~0.5 s). */
  redemptionFrames: number;
  /** Frames of retained audio before the first speech frame (~onset padding). */
  preSpeechPadFrames: number;
  /** Utterances with fewer than this many positive frames are discarded as noise. */
  minSpeechFrames: number;
  /** Hard cap on utterance length; forces an end so a stuck VAD can't buffer forever (~30 s). */
  maxSpeechFrames: number;
}

export const DEFAULT_VAD_OPTIONS: VadOptions = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionFrames: 16,
  preSpeechPadFrames: 8,
  minSpeechFrames: 4,
  maxSpeechFrames: Math.round(30 * FRAMES_PER_SEC),
};

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
  private readonly opts: VadOptions;
  private speaking = false;
  private redemption = 0;
  private positiveFrames = 0;
  private speechFrames: Float32Array[] = [];
  /** Rolling window of the most recent frames while idle (the pre-speech pad). */
  private preBuffer: Float32Array[] = [];

  constructor(opts: Partial<VadOptions> = {}) {
    this.opts = { ...DEFAULT_VAD_OPTIONS, ...opts };
  }

  /** Feed one frame and its Silero speech probability. */
  process(frame: Float32Array, prob: number): VadEvent {
    const o = this.opts;
    if (!this.speaking) {
      if (prob >= o.positiveSpeechThreshold) {
        // Utterance = the retained pre-speech pad (prior idle frames) + this first speech frame.
        this.speaking = true;
        this.redemption = 0;
        this.positiveFrames = 1;
        this.speechFrames = [...this.preBuffer, frame];
        this.preBuffer = [];
        return { event: 'speechStart' };
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
