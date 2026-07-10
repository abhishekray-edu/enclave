import { describe, it, expect } from 'vitest';
import { normalizeMathDelimiters } from '../mathText';

describe('normalizeMathDelimiters', () => {
  it('converts display \\[...\\] to $$...$$', () => {
    expect(normalizeMathDelimiters('before \\[ x = 1 \\] after')).toBe('before $$ x = 1 $$ after');
  });

  it('converts inline \\(...\\) to $$...$$', () => {
    expect(normalizeMathDelimiters('the value \\(a+b\\) here')).toBe('the value $$a+b$$ here');
  });

  it('converts multi-line display math intact', () => {
    const input = '\\[\n t = \\frac{v}{g} \n\\]';
    expect(normalizeMathDelimiters(input)).toBe('$$\n t = \\frac{v}{g} \n$$');
  });

  it('wraps bare \\boxed{...} with balanced nested braces', () => {
    expect(normalizeMathDelimiters('answer \\boxed{18.36 \\text{ s}} end')).toBe(
      'answer $$\\boxed{18.36 \\text{ s}}$$ end',
    );
  });

  it('does not double-wrap \\boxed already inside $$', () => {
    expect(normalizeMathDelimiters('$$\\boxed{5}$$')).toBe('$$\\boxed{5}$$');
  });

  it('leaves plain currency text untouched (no math markers)', () => {
    const input = 'it costs $5 and then $10 total';
    expect(normalizeMathDelimiters(input)).toBe(input);
  });

  it('does not transform delimiters inside fenced code blocks', () => {
    const input = '```\nlatex: \\[ x \\] and \\boxed{y}\n```';
    expect(normalizeMathDelimiters(input)).toBe(input);
  });

  it('does not transform delimiters inside inline code', () => {
    const input = 'use `\\(a\\)` literally';
    expect(normalizeMathDelimiters(input)).toBe(input);
  });

  it('converts math outside code while preserving code inside the same string', () => {
    const input = 'math \\(a\\) then `\\(b\\)` code';
    expect(normalizeMathDelimiters(input)).toBe('math $$a$$ then `\\(b\\)` code');
  });

  it('is a no-op when there is nothing to convert', () => {
    const input = 'just some **markdown** text';
    expect(normalizeMathDelimiters(input)).toBe(input);
  });
});
