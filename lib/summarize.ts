// Hierarchical (map-reduce) summarization for pages that exceed the context window
// (BooookScore, arXiv:2310.00785). Panel-side orchestration; the actual generation call is
// injected as `runOne`, so this module never imports @mlc-ai/web-llm.
import type { Chunk, ChatMessage } from './types';
import { estimateTokens } from './prompt';

export type SummarizePhase =
  | { phase: 'map'; index: number; total: number }
  | { phase: 'reduce'; round: number }
  | { phase: 'done' };

export interface RunOneOptions {
  temperature: number;
  maxTokens: number;
  /** Stream deltas to the UI (used for the final merge) vs return one result (map/intermediate). */
  stream?: boolean;
}

export interface SummarizeCallbacks {
  runOne: (messages: ChatMessage[], opts: RunOneOptions) => Promise<string>;
  onProgress: (p: SummarizePhase) => void;
  signal: AbortSignal;
}

export interface SummarizeOptions {
  ctxTokens: number;
  title: string;
  /** Per-model single-prompt cap (safePromptTokens); bounds every reduce input. */
  maxPromptTokens?: number;
}

const MAP_SYSTEM =
  'You summarize ONE section of a web page into 2-4 concise, faithful bullet points. ' +
  'Use only the provided text. No preamble.';
const REDUCE_SYSTEM =
  'You combine section summaries of a web page into a single coherent, de-duplicated bulleted ' +
  'summary. Use only the provided summaries. No preamble.';

const MERGE_FANOUT = 4;

/** Reduce input shouldn't exceed ~60% of the context window (leaving room for the answer),
 *  nor the per-model single-prompt cap — a merge prompt is a GPU prefill like any other. */
function reduceBudget(opts: SummarizeOptions): number {
  const cap = Math.min(Math.floor(opts.ctxTokens * 0.6), opts.maxPromptTokens ?? Infinity);
  return Math.max(512, cap);
}

function abortIf(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
}

/**
 * Summarize a chunked page by summarizing each chunk (map), then hierarchically merging the
 * summaries (reduce) until they fit, then one final coherent pass. The final pass streams.
 */
export async function summarizeChunks(
  chunks: Chunk[],
  opts: SummarizeOptions,
  cb: SummarizeCallbacks,
): Promise<string> {
  // MAP — one bounded, non-streamed call per chunk.
  let summaries: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    abortIf(cb.signal);
    cb.onProgress({ phase: 'map', index: i, total: chunks.length });
    const label = chunks[i].heading ? `Section "${chunks[i].heading}"` : `Section ${i + 1}`;
    const text = await cb.runOne(
      [
        { role: 'system', content: MAP_SYSTEM },
        { role: 'user', content: `${label} of "${opts.title}" (part ${i + 1}/${chunks.length}):\n\n${chunks[i].text}` },
      ],
      { temperature: 0.2, maxTokens: 200, stream: false },
    );
    if (text.trim()) summaries.push(text.trim());
  }

  if (summaries.length === 0) return '';
  if (summaries.length === 1) return summaries[0];

  // REDUCE — hierarchically merge groups until the combined summaries fit the budget.
  let round = 1;
  while (summaries.length > 1 && estimateTokens(summaries.join('\n')) > reduceBudget(opts)) {
    abortIf(cb.signal);
    cb.onProgress({ phase: 'reduce', round });
    const merged: string[] = [];
    for (let i = 0; i < summaries.length; i += MERGE_FANOUT) {
      abortIf(cb.signal);
      const group = summaries.slice(i, i + MERGE_FANOUT).join('\n\n');
      const out = await cb.runOne(
        [
          { role: 'system', content: REDUCE_SYSTEM },
          { role: 'user', content: `Combine these section summaries:\n\n${group}` },
        ],
        { temperature: 0.2, maxTokens: 300, stream: false },
      );
      if (out.trim()) merged.push(out.trim());
    }
    summaries = merged;
    round++;
  }

  // FINAL — one coherent merge; streams to the UI.
  abortIf(cb.signal);
  cb.onProgress({ phase: 'reduce', round });
  const final = await cb.runOne(
    [
      { role: 'system', content: REDUCE_SYSTEM },
      {
        role: 'user',
        content: `Combine these section summaries of "${opts.title}" into one coherent, de-duplicated bulleted summary:\n\n${summaries.join('\n\n')}`,
      },
    ],
    { temperature: 0.2, maxTokens: 700, stream: true },
  );
  cb.onProgress({ phase: 'done' });
  return final;
}
