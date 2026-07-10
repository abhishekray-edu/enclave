import { describe, it, expect } from 'vitest';
import { VadFrameProcessor, type VadOptions } from '../vadFrameProcessor';

// Frames carry a single distinguishing value so we can assert exactly which frames end up in the
// emitted utterance audio (the processor is agnostic to frame length/content).
const f = (v: number) => Float32Array.from([v]);
const vals = (a?: Float32Array) => (a ? Array.from(a) : undefined);

const OPTS: Partial<VadOptions> = {
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.35,
  redemptionFrames: 2,
  preSpeechPadFrames: 2,
  minSpeechFrames: 2,
  maxSpeechFrames: 100,
};

describe('VadFrameProcessor', () => {
  it('emits speechStart when the probability crosses the positive threshold', () => {
    const vad = new VadFrameProcessor(OPTS);
    expect(vad.process(f(0), 0.1).event).toBe('none');
    expect(vad.process(f(1), 0.9).event).toBe('speechStart');
  });

  it('includes the pre-speech pad and trims the trailing redemption silence', () => {
    const vad = new VadFrameProcessor(OPTS);
    vad.process(f(0), 0.0); // idle → pad [0]
    vad.process(f(1), 0.0); // idle → pad [0,1]
    expect(vad.process(f(2), 0.9).event).toBe('speechStart'); // buffer [0,1,2]
    expect(vad.process(f(3), 0.9).event).toBe('none'); // buffer [0,1,2,3]
    expect(vad.process(f(4), 0.1).event).toBe('none'); // redemption 1
    const ev = vad.process(f(5), 0.1); // redemption 2 → end
    expect(ev.event).toBe('speechEnd');
    // The two redemption frames (4,5) are trimmed; the pad (0,1) is kept.
    expect(vals(ev.event === 'speechEnd' ? ev.audio : undefined)).toEqual([0, 1, 2, 3]);
  });

  it('discards a too-short utterance as a misfire (speechEnd with no audio)', () => {
    const vad = new VadFrameProcessor({ ...OPTS, minSpeechFrames: 3, preSpeechPadFrames: 0 });
    expect(vad.process(f(0), 0.9).event).toBe('speechStart'); // positive frames = 1
    expect(vad.process(f(1), 0.1).event).toBe('none'); // redemption 1
    const ev = vad.process(f(2), 0.1); // redemption 2 → end, but only 1 positive frame
    expect(ev.event).toBe('speechEnd');
    expect(ev.event === 'speechEnd' && ev.audio).toBeUndefined();
  });

  it('survives sub-threshold dips: a between-threshold frame and a positive frame reset redemption', () => {
    const vad = new VadFrameProcessor(OPTS);
    vad.process(f(0), 0.9); // start
    expect(vad.process(f(1), 0.4).event).toBe('none'); // between thresholds → no redemption
    expect(vad.process(f(2), 0.1).event).toBe('none'); // redemption 1
    expect(vad.process(f(3), 0.9).event).toBe('none'); // positive → redemption reset
    expect(vad.process(f(4), 0.1).event).toBe('none'); // redemption 1 (did NOT end at 2 dips)
    expect(vad.process(f(5), 0.1).event).toBe('speechEnd'); // redemption 2 → end
  });

  it('force-ends at the max-duration cap, keeping all audio', () => {
    const vad = new VadFrameProcessor({ ...OPTS, maxSpeechFrames: 4, redemptionFrames: 100, preSpeechPadFrames: 0 });
    vad.process(f(0), 0.9); // len 1 (start)
    vad.process(f(1), 0.9); // len 2
    vad.process(f(2), 0.9); // len 3
    const ev = vad.process(f(3), 0.9); // len 4 == max → forced end
    expect(ev.event).toBe('speechEnd');
    expect(vals(ev.event === 'speechEnd' ? ev.audio : undefined)).toEqual([0, 1, 2, 3]);
  });

  it('flush returns buffered audio even below the min-speech threshold, and reset clears state', () => {
    const vad = new VadFrameProcessor({ ...OPTS, preSpeechPadFrames: 0 });
    vad.process(f(7), 0.9); // start, single positive frame
    const ev = vad.flush();
    expect(ev.event).toBe('speechEnd');
    expect(vals(ev.event === 'speechEnd' ? ev.audio : undefined)).toEqual([7]);
    // After flush the machine is idle again.
    expect(vad.flush().event).toBe('none');
  });

  it('flush while idle is a no-op', () => {
    const vad = new VadFrameProcessor(OPTS);
    expect(vad.flush().event).toBe('none');
  });

  it('startConfirmFrames defers speechStart until enough positive frames arrive', () => {
    const vad = new VadFrameProcessor({ ...OPTS, startConfirmFrames: 3, preSpeechPadFrames: 0 });
    expect(vad.process(f(0), 0.9).event).toBe('none'); // confirming 1
    expect(vad.process(f(1), 0.9).event).toBe('none'); // confirming 2
    expect(vad.process(f(2), 0.9).event).toBe('speechStart'); // confirmed on 3rd
  });

  it('a collapsed onset (burst then silence) never emits speechStart', () => {
    const vad = new VadFrameProcessor({ ...OPTS, startConfirmFrames: 5, preSpeechPadFrames: 2 });
    expect(vad.process(f(0), 0.9).event).toBe('none'); // confirming 1
    expect(vad.process(f(1), 0.9).event).toBe('none'); // confirming 2
    expect(vad.process(f(2), 0.1).event).toBe('none'); // collapse — folds back to the pad
    // Still idle: a fresh, sustained onset must confirm from scratch.
    expect(vad.process(f(3), 0.9).event).toBe('none');
  });

  it('updateOptions applies from the next frame', () => {
    const vad = new VadFrameProcessor({ ...OPTS, startConfirmFrames: 1 });
    vad.updateOptions({ startConfirmFrames: 2 });
    expect(vad.process(f(0), 0.9).event).toBe('none'); // now needs 2 to confirm
    expect(vad.process(f(1), 0.9).event).toBe('speechStart');
  });
});
