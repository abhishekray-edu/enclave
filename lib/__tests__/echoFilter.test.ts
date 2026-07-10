import { describe, it, expect } from 'vitest';
import { echoOverlap, isLikelyEcho } from '../echoFilter';

describe('echoFilter', () => {
  it('scores an exact echo near 1', () => {
    const spoken = ['The answer is about eighteen point three six seconds.'];
    expect(echoOverlap('the answer is about eighteen point three six seconds', spoken)).toBeCloseTo(1, 5);
  });

  it('scores an unrelated question well below the echo threshold', () => {
    const spoken = ['The answer is about eighteen point three six seconds.'];
    // Only incidental stopword overlap ("is", "the") — nowhere near the 0.75 discard threshold.
    expect(echoOverlap('what is the capital of France', spoken)).toBeLessThan(0.5);
  });

  it('flags an exact echo as likely echo', () => {
    const spoken = ['Gravity pulls the ball back down to the ground.'];
    expect(isLikelyEcho('gravity pulls the ball back down to the ground', spoken)).toBe(true);
  });

  it('does not flag a genuine new question as echo', () => {
    const spoken = ['Gravity pulls the ball back down to the ground.'];
    expect(isLikelyEcho('actually can you explain air resistance too', spoken)).toBe(false);
  });

  it('does not flag a very short transcript (guard)', () => {
    const spoken = ['Yes it does.'];
    expect(isLikelyEcho('yes', spoken)).toBe(false);
  });

  it('returns 0 overlap when nothing was spoken', () => {
    expect(echoOverlap('anything here', [])).toBe(0);
  });

  it('ignores punctuation and case differences', () => {
    const spoken = ['Hello, world! How are you?'];
    expect(echoOverlap('hello world how are you', spoken)).toBeCloseTo(1, 5);
  });
});
