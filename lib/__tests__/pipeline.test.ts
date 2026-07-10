import { describe, it, expect } from 'vitest';
import { estimateTokens, pageBudgetTokens, buildMessages } from '../prompt';
import { chunkPage } from '../chunking';
import { DEFAULT_SETTINGS, type PageContent, type RetrievedChunk } from '../types';

function page(partial: Partial<PageContent>): PageContent {
  return {
    title: 'T',
    url: 'https://example.com',
    textContent: '',
    excerpt: '',
    byline: '',
    siteName: '',
    selection: '',
    ...partial,
  };
}

const pad = (label: string, chars: number) => label + ' ' + 'x'.repeat(Math.max(0, chars - label.length));

describe('estimateTokens', () => {
  it('is ~chars/4 and 0 for empty', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('abcd')).toBe(1);
    expect(estimateTokens('a'.repeat(40))).toBe(10);
  });
});

describe('pageBudgetTokens', () => {
  it('is positive and shrinks as the conversation grows', () => {
    const p = page({ textContent: 'hello' });
    const big = pageBudgetTokens(DEFAULT_SETTINGS, p, [{ role: 'user', content: 'hi' }], 8192);
    const smaller = pageBudgetTokens(
      DEFAULT_SETTINGS,
      p,
      [{ role: 'user', content: 'x'.repeat(4000) }],
      8192,
    );
    expect(big).toBeGreaterThan(0);
    expect(smaller).toBeLessThan(big);
  });
});

describe('buildMessages — full (small page)', () => {
  it('sends the whole page and is not truncated', () => {
    const p = page({ textContent: 'The quick brown fox.' });
    const built = buildMessages(DEFAULT_SETTINGS, p, [{ role: 'user', content: 'q' }], 8192);
    expect(built.mode).toBe('full');
    expect(built.truncated).toBe(false);
    expect(built.messages[0].content).toContain('The quick brown fox.');
  });
});

describe('buildMessages — page context off (page: null)', () => {
  it('sends nothing from the page — no title, URL, or content sections', () => {
    const built = buildMessages(DEFAULT_SETTINGS, null, [{ role: 'user', content: 'q' }], 8192);
    const system = built.messages[0].content;
    expect(built.truncated).toBe(false);
    expect(system).not.toContain('PAGE CONTEXT');
    expect(system).not.toContain('PAGE CONTENT');
    expect(system).not.toContain('example.com');
    expect(system).toContain('Page context is turned off');
    expect(built.messages).toHaveLength(2);
  });
});

describe('buildMessages — structure-aware truncation (large page)', () => {
  it('keeps leading whole sections, drops trailing ones, never deletes the middle of a kept block', () => {
    const p = page({
      textContent: 'whole page text',
      blocks: [
        { type: 'heading', level: 1, text: 'SECTION_ONE' },
        { type: 'text', text: pad('ONEBODY', 400) },
        { type: 'heading', level: 1, text: 'SECTION_TWO' },
        { type: 'text', text: pad('TWOBODY', 400) },
        { type: 'heading', level: 1, text: 'SECTION_THREE' },
        { type: 'text', text: pad('THREEBODY', 400) },
      ],
    });
    // ctx small enough that only ~1 section fits after the ~1024-token answer reserve.
    const built = buildMessages(DEFAULT_SETTINGS, p, [], 1300);
    const body = built.messages[0].content;
    expect(built.truncated).toBe(true);
    expect(body).toContain('SECTION_ONE');
    expect(body).toContain('ONEBODY');
    expect(body).not.toContain('THREEBODY'); // trailing section dropped, not the middle
    expect(body).toContain('trimmed'); // trim marker present
  });
});

describe('buildMessages — RAG mode orders most-relevant LAST', () => {
  it('packs retrieved chunks with the highest score nearest the question', () => {
    const retrieved: RetrievedChunk[] = [
      { ordinal: 0, text: 'CHUNK_LOW', tokensEstimate: 3, score: 0.2 },
      { ordinal: 1, text: 'CHUNK_HIGH', tokensEstimate: 3, score: 0.9 },
      { ordinal: 2, text: 'CHUNK_MID', tokensEstimate: 3, score: 0.5 },
    ];
    const built = buildMessages(DEFAULT_SETTINGS, page({ textContent: 'x' }), [], 8192, { retrieved });
    const body = built.messages[0].content;
    expect(built.mode).toBe('rag');
    expect(built.usedChunks).toHaveLength(3);
    // ascending by score in the body → HIGH must appear last (nearest the user turn).
    expect(body.indexOf('CHUNK_HIGH')).toBeGreaterThan(body.indexOf('CHUNK_MID'));
    expect(body.indexOf('CHUNK_MID')).toBeGreaterThan(body.indexOf('CHUNK_LOW'));
  });
});

describe('buildMessages — includeBody false', () => {
  it('omits the page body', () => {
    const p = page({ textContent: 'SENSITIVE_BODY_TEXT' });
    const built = buildMessages(DEFAULT_SETTINGS, p, [], 8192, { includeBody: false });
    expect(built.messages[0].content).not.toContain('SENSITIVE_BODY_TEXT');
  });
});

describe('chunkPage', () => {
  it('chunks structured blocks, carries headings, keeps stable ordinals', () => {
    const p = page({
      textContent: 'doc',
      blocks: [
        { type: 'heading', level: 2, text: 'Pricing' },
        { type: 'text', text: pad('Pricing details', 1600) }, // ~400 tokens → multiple chunks
      ],
    });
    const chunks = chunkPage(p, { targetTokens: 80, maxTokens: 120, overlapTokens: 10 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c, i) => expect(c.ordinal).toBe(i));
    expect(chunks[0].heading).toBe('Pricing');
    // every chunk respects the cap (with a little slack for overlap)
    chunks.forEach((c) => expect(c.tokensEstimate).toBeLessThanOrEqual(120 + 20));
  });

  it('falls back to paragraph splitting when there are no blocks', () => {
    const p = page({ textContent: 'Para one.\n\nPara two.\n\nPara three.' });
    const chunks = chunkPage(p, { targetTokens: 4, maxTokens: 8, overlapTokens: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(1);
    expect(chunks.map((c) => c.text).join(' ')).toContain('Para');
  });
});
