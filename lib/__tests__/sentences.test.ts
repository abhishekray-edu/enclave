import { describe, it, expect } from 'vitest';
import { SentenceStream } from '../sentences';

/** Feed a full string one character at a time (simulating token streaming) and collect every
 *  emitted sentence plus the final flush. */
function streamChars(text: string): string[] {
  const s = new SentenceStream();
  const out: string[] = [];
  let acc = '';
  for (const ch of text) {
    acc += ch;
    for (const piece of s.push(acc)) out.push(piece.text);
  }
  const tail = s.flush();
  if (tail) out.push(tail.text);
  return out;
}

describe('SentenceStream', () => {
  it('emits a long sentence as soon as the next one starts', () => {
    const s = new SentenceStream();
    // Not yet — no following sentence has begun.
    expect(s.push('This is a reasonably long first sentence here.')).toEqual([]);
    // The trailing space + next char completes the boundary.
    const out = s.push('This is a reasonably long first sentence here. And more');
    expect(out.map((p) => p.text)).toEqual(['This is a reasonably long first sentence here.']);
  });

  it('never re-emits an already-emitted sentence', () => {
    const s = new SentenceStream();
    s.push('A sufficiently long opening sentence to speak. Next');
    const again = s.push('A sufficiently long opening sentence to speak. The next sentence is also quite long. End');
    expect(again.map((p) => p.text)).toEqual(['The next sentence is also quite long.']);
  });

  it('does not split on a decimal point mid-number', () => {
    const out = streamChars('The value is 3.5 metres per second in total here.');
    expect(out).toEqual(['The value is 3.5 metres per second in total here.']);
  });

  it('holds a short sentence and merges it into the next', () => {
    const out = streamChars('Sure. That was a much longer follow-up sentence to speak aloud.');
    expect(out).toEqual(['Sure. That was a much longer follow-up sentence to speak aloud.']);
  });

  it('splits on a paragraph break', () => {
    const out = streamChars('First long paragraph of text goes here\n\nSecond long paragraph over here');
    expect(out).toEqual([
      'First long paragraph of text goes here',
      'Second long paragraph over here',
    ]);
  });

  it('endOffset points just past the sentence in the pushed text', () => {
    const s = new SentenceStream();
    const full = 'A long enough sentence to emit right away. Then more.';
    const out = s.push(full);
    expect(out).toHaveLength(1);
    expect(full.slice(0, out[0].endOffset)).toBe('A long enough sentence to emit right away. ');
  });

  it('flush returns the trailing remainder', () => {
    const s = new SentenceStream();
    s.push('A complete first sentence to speak now. Trailing text with no end');
    const tail = s.flush();
    expect(tail?.text).toBe('Trailing text with no end');
  });

  it('flush returns null when nothing remains', () => {
    const s = new SentenceStream();
    s.push('One long enough sentence to be emitted. ');
    // The single sentence is held pending a following char; flush releases it, then nothing left.
    s.flush();
    expect(s.flush()).toBeNull();
  });

  it('handles closing quotes after terminal punctuation', () => {
    const s = new SentenceStream();
    const out = s.push('He said "go to the shop now." Then she left the building.');
    expect(out.map((p) => p.text)).toEqual(['He said "go to the shop now."']);
  });
});
