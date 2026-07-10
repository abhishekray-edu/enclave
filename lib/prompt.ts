import type { ChatMessage, PageBlock, PageContent, RetrievedChunk, Settings } from './types';

/** Rough token estimate (~4 chars/token for English). Good enough for budgeting. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Reserve this many tokens of the context window for the model's answer. */
const ANSWER_RESERVE_TOKENS = 1024;

function pageHeader(page: PageContent): string {
  const lines = [`Title: ${page.title || '(untitled)'}`, `URL: ${page.url}`];
  if (page.siteName) lines.push(`Site: ${page.siteName}`);
  if (page.byline) lines.push(`Author: ${page.byline}`);
  if (page.selection.trim()) {
    lines.push('', 'The user has selected this text on the page:', `"""${page.selection.trim()}"""`);
  }
  return lines.join('\n');
}

/**
 * Tokens available for page content after the system scaffold, conversation and answer
 * reserve. Exposed so callers (e.g. the RAG threshold in the panel) can decide whether the
 * full page fits before deciding to retrieve. No artificial page cap — the page may use the
 * whole context window (which is already clamped per-model upstream).
 */
export function pageBudgetTokens(
  settings: Settings,
  page: PageContent,
  conversation: ChatMessage[],
  ctxTokens: number,
  systemPromptOverride?: string,
): number {
  const header = pageHeader(page);
  const systemPrompt = systemPromptOverride ?? settings.systemPrompt;
  const systemBase = `${systemPrompt}\n\n--- PAGE CONTEXT ---\n${header}\n\n--- PAGE CONTENT ---\n`;
  const fixed =
    estimateTokens(systemBase) +
    conversation.reduce((sum, m) => sum + estimateTokens(m.content) + 8, 0) +
    ANSWER_RESERVE_TOKENS;
  return Math.max(0, ctxTokens - fixed);
}

/** Head truncation fallback (used only when the page has no structural blocks). Keeps the
 *  start of the page and marks the cut, rather than splicing out the middle. */
function truncateHead(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return '';
  if (estimateTokens(text) <= maxTokens) return text;
  return text.slice(0, maxTokens * 4) + '\n\n[…content trimmed to fit the model context…]';
}

/** Structure-aware selection: include whole blocks (headings + paragraphs) in document order
 *  until the budget is hit, then drop the trailing sections. Never splices mid-sentence and
 *  never deletes the middle of a kept section. */
function buildBodyFromBlocks(blocks: PageBlock[], maxTokens: number): { body: string; truncated: boolean } {
  const parts: string[] = [];
  let used = 0;
  let truncated = false;
  for (const b of blocks) {
    const piece = b.type === 'heading' ? `${'#'.repeat(b.level ?? 1)} ${b.text}` : b.text;
    const t = estimateTokens(piece) + 1;
    if (used + t > maxTokens) {
      truncated = true;
      break;
    }
    parts.push(piece);
    used += t;
  }
  // Nothing fit (e.g. one giant block, no headings) → fall back to head truncation.
  if (!parts.length) {
    const joined = blocks.map((b) => b.text).join('\n');
    return { body: truncateHead(joined, maxTokens), truncated: estimateTokens(joined) > maxTokens };
  }
  let body = parts.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  if (truncated) body += '\n\n[…later sections of the page were trimmed to fit the model context…]';
  return { body, truncated };
}

/** Pack retrieved chunks within budget: keep the highest-scoring that fit, then order them
 *  most-relevant LAST (nearest the question) to counter "lost in the middle". */
function packChunks(
  chunks: RetrievedChunk[],
  maxTokens: number,
): { body: string; used: RetrievedChunk[]; dropped: boolean } {
  const label = (c: RetrievedChunk) => (c.heading ? `[Section: ${c.heading}]\n` : '');
  const chosen: RetrievedChunk[] = [];
  let used = 0;
  let dropped = false;
  for (const c of [...chunks].sort((a, b) => b.score - a.score)) {
    const t = estimateTokens(`${label(c)}${c.text}`) + 4;
    if (used + t > maxTokens) {
      dropped = true;
      continue;
    }
    chosen.push(c);
    used += t;
  }
  const ordered = chosen.sort((a, b) => a.score - b.score); // ascending → best last
  const body = ordered.map((c) => `${label(c)}${c.text}`).join('\n\n---\n\n');
  return { body, used: ordered, dropped };
}

export interface BuildOptions {
  /** Task-specific system prompt that replaces settings.systemPrompt for this run. */
  systemPromptOverride?: string;
  /** Pre-selected, ordered chunks (RAG). When present, the body is built from these. */
  retrieved?: RetrievedChunk[];
  /** Skip the page body entirely (e.g. selection-only "explain"). Defaults to true. */
  includeBody?: boolean;
  /** Hard per-model cap on body tokens (safePromptTokens): bounds the single GPU prefill
   *  submission regardless of how large the context window is. */
  maxBodyTokens?: number;
}

export interface BuiltPrompt {
  messages: ChatMessage[];
  /** True when the page did not fully fit and some content was dropped. */
  truncated: boolean;
  /** How the body was assembled: full/structure-aware text vs retrieved chunks. */
  mode: 'full' | 'rag';
  /** Chunks actually included (RAG mode), in render order, for provenance. */
  usedChunks?: RetrievedChunk[];
}

/** Appended to the system prompt when page context is toggled off, so the model doesn't
 *  hallucinate a page it was never shown. */
const NO_PAGE_NOTE =
  'Page context is turned off: you cannot see the current web page. Answer from the conversation ' +
  'and general knowledge only, and if asked about "this page", say that page context is disabled.';

/**
 * Build the message array: a system message carrying the page context, followed by the
 * conversation. The body is assembled by the best available strategy — retrieved chunks
 * (RAG) if provided, else structure-aware section selection, else head truncation — all
 * bounded by the per-model context budget. Returns coverage info for the UI.
 *
 * `page: null` means page context is disabled — nothing from the tab (not even title or
 * URL) enters the prompt.
 */
export function buildMessages(
  settings: Settings,
  page: PageContent | null,
  conversation: ChatMessage[],
  /** Context window in tokens to budget the page text against. */
  ctxTokens: number,
  opts?: BuildOptions,
): BuiltPrompt {
  const systemPromptBase = opts?.systemPromptOverride ?? settings.systemPrompt;
  if (!page) {
    return {
      messages: [{ role: 'system', content: `${systemPromptBase}\n\n${NO_PAGE_NOTE}` }, ...conversation],
      truncated: false,
      mode: 'full',
    };
  }
  const header = pageHeader(page);
  let budget = pageBudgetTokens(settings, page, conversation, ctxTokens, systemPromptBase);
  if (opts?.maxBodyTokens != null) budget = Math.min(budget, opts.maxBodyTokens);
  const includeBody = opts?.includeBody ?? true;

  let body = '';
  let truncated = false;
  let mode: 'full' | 'rag' = 'full';
  let usedChunks: RetrievedChunk[] | undefined;

  if (!includeBody) {
    body = '';
  } else if (opts?.retrieved && opts.retrieved.length) {
    mode = 'rag';
    const packed = packChunks(opts.retrieved, budget);
    body = packed.body;
    usedChunks = packed.used;
    truncated = packed.dropped;
  } else if (page.blocks && page.blocks.length) {
    const r = buildBodyFromBlocks(page.blocks, budget);
    body = r.body;
    truncated = r.truncated;
  } else {
    body = truncateHead(page.textContent, budget);
    truncated = estimateTokens(page.textContent) > budget;
  }

  const systemBase = `${systemPromptBase}\n\n--- PAGE CONTEXT ---\n${header}\n\n--- PAGE CONTENT ---\n`;
  const system: ChatMessage = { role: 'system', content: systemBase + body };
  return {
    messages: [system, ...conversation],
    truncated,
    mode,
    usedChunks,
  };
}
