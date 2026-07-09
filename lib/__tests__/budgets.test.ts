// Regression tests for the 2026-07-09 crash: a ~7k-token prompt prefilled into an 8B model
// monopolized the integrated GPU past macOS's 40s WindowServer watchdog and killed the whole
// login session. These tests pin the invariant that NO path can build a prompt bigger than
// the per-model safePromptTokens cap, for a 6,000-word page, on every model in the catalog.
import { describe, it, expect } from 'vitest';
import { buildMessages, estimateTokens, pageBudgetTokens } from '../prompt';
import { chunkPage } from '../chunking';
import { summarizeChunks } from '../summarize';
import { WEBLLM_MODELS } from '../webllmClient';
import { DEFAULT_SETTINGS, MAX_CONTEXT_TOKENS, type ChatMessage, type PageBlock, type PageContent, type RetrievedChunk } from '../types';

/** ~6,000-word page (≈36k chars ≈ 9k tokens) with headings, like a long article. */
function bigPage(selection = ''): PageContent {
  const blocks: PageBlock[] = [];
  const sentence = 'The committee reviewed the proposal and found the projected costs broadly reasonable. ';
  for (let s = 0; s < 12; s++) {
    blocks.push({ type: 'heading', level: 2, text: `Section ${s + 1}` });
    blocks.push({ type: 'text', text: sentence.repeat(35) }); // ~3k chars per section
  }
  const textContent = blocks.map((b) => b.text).join('\n\n');
  return {
    title: 'Long test article',
    url: 'https://example.com/long',
    textContent,
    excerpt: '',
    byline: '',
    siteName: '',
    selection,
    blocks,
  };
}

const conversation: ChatMessage[] = [{ role: 'user', content: 'What did the committee decide?' }];
// The system scaffold (prompt + page header) rides on top of the body; keep slack for it.
const SCAFFOLD_SLACK_TOKENS = 300;

describe('model catalog safety invariants', () => {
  it('every model declares a safe single-prompt cap no larger than 4096 tokens', () => {
    for (const m of WEBLLM_MODELS) {
      expect(m.safePromptTokens, m.id).toBeGreaterThanOrEqual(1024);
      expect(m.safePromptTokens, m.id).toBeLessThanOrEqual(4096);
      expect(m.safePromptTokens, m.id).toBeLessThan(m.maxCtx);
    }
  });

  it('8B-class models (≥5 GB) get the tightest prompt cap and a bounded KV context', () => {
    const big = WEBLLM_MODELS.filter((m) => m.approxGb >= 5);
    expect(big.length).toBeGreaterThan(0);
    for (const m of big) {
      expect(m.safePromptTokens, m.id).toBeLessThanOrEqual(1536);
      expect(m.maxCtx, m.id).toBeLessThanOrEqual(8192);
    }
  });
});

describe('buildMessages honors the per-model body cap on a 6,000-word page', () => {
  for (const m of WEBLLM_MODELS) {
    const ctx = Math.min(MAX_CONTEXT_TOKENS, m.maxCtx);

    it(`${m.label}: structure-aware (stuffing) path stays under ${m.safePromptTokens} tokens`, () => {
      const built = buildMessages(DEFAULT_SETTINGS, bigPage(), conversation, ctx, {
        maxBodyTokens: m.safePromptTokens,
      });
      expect(built.truncated).toBe(true); // 9k tokens never fits — must be marked
      expect(estimateTokens(built.messages[0].content)).toBeLessThanOrEqual(
        m.safePromptTokens + SCAFFOLD_SLACK_TOKENS,
      );
    });

    it(`${m.label}: stuffing path stays capped even WITH a selection pinned`, () => {
      const built = buildMessages(
        DEFAULT_SETTINGS,
        bigPage('An important selected passage about the committee.'),
        conversation,
        ctx,
        { maxBodyTokens: m.safePromptTokens },
      );
      expect(estimateTokens(built.messages[0].content)).toBeLessThanOrEqual(
        m.safePromptTokens + SCAFFOLD_SLACK_TOKENS,
      );
    });

    it(`${m.label}: RAG path packs retrieved chunks under the cap`, () => {
      const chunks = chunkPage(bigPage());
      const retrieved: RetrievedChunk[] = chunks.slice(0, 16).map((c, i) => ({ ...c, score: 1 - i * 0.05 }));
      const built = buildMessages(DEFAULT_SETTINGS, bigPage(), conversation, ctx, {
        retrieved,
        maxBodyTokens: m.safePromptTokens,
      });
      expect(built.mode).toBe('rag');
      expect(estimateTokens(built.messages[0].content)).toBeLessThanOrEqual(
        m.safePromptTokens + SCAFFOLD_SLACK_TOKENS,
      );
    });
  }
});

describe('RAG engages whenever the page exceeds the safe body budget', () => {
  it('a 6,000-word page always overflows every model’s cap (so ask/extract retrieve)', () => {
    const page = bigPage();
    for (const m of WEBLLM_MODELS) {
      const ctx = Math.min(MAX_CONTEXT_TOKENS, m.maxCtx);
      const budget = pageBudgetTokens(DEFAULT_SETTINGS, page, conversation, ctx);
      const bodyBudget = Math.min(budget, m.safePromptTokens);
      expect(estimateTokens(page.textContent)).toBeGreaterThan(bodyBudget);
    }
  });
});

describe('map-reduce summarization never submits an oversized prompt', () => {
  it('every map/reduce call stays under the per-model cap (8B: 1536)', async () => {
    const cap = 1536;
    const page = bigPage();
    const chunks = chunkPage(page);
    expect(chunks.length).toBeGreaterThan(8);

    const promptSizes: number[] = [];
    const runOne = async (messages: ChatMessage[]) => {
      promptSizes.push(estimateTokens(messages.map((m) => m.content).join('\n')));
      // Simulate a worst-case chatty model: ~200-token bullet summaries per call.
      return '- ' + 'the committee reviewed and approved the item under discussion. '.repeat(12);
    };

    await summarizeChunks(
      chunks,
      { ctxTokens: 8192, title: page.title, maxPromptTokens: cap },
      { runOne, onProgress: () => {}, signal: new AbortController().signal },
    );

    expect(promptSizes.length).toBeGreaterThan(chunks.length); // map + at least one reduce
    for (const size of promptSizes) {
      expect(size).toBeLessThanOrEqual(cap + SCAFFOLD_SLACK_TOKENS);
    }
  });
});
