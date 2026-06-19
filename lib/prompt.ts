import type { ChatMessage, PageContent, Settings } from './types';

/** Rough token estimate (~4 chars/token for English). Good enough for budgeting. */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Reserve this many tokens of the context window for the model's answer. */
const ANSWER_RESERVE_TOKENS = 1024;
const MAX_PAGE_CONTEXT_TOKENS = 4096;

/** Head/tail truncation that keeps the start and end of long pages. */
function truncateToTokens(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return '';
  if (estimateTokens(text) <= maxTokens) return text;
  const maxChars = maxTokens * 4;
  const head = Math.floor(maxChars * 0.7);
  const tail = maxChars - head;
  return (
    text.slice(0, head) +
    '\n\n[…content trimmed to fit the model context…]\n\n' +
    text.slice(text.length - tail)
  );
}

function pageHeader(page: PageContent): string {
  const lines = [`Title: ${page.title || '(untitled)'}`, `URL: ${page.url}`];
  if (page.siteName) lines.push(`Site: ${page.siteName}`);
  if (page.byline) lines.push(`Author: ${page.byline}`);
  if (page.selection.trim()) {
    lines.push('', 'The user has selected this text on the page:', `"""${page.selection.trim()}"""`);
  }
  return lines.join('\n');
}

export interface BuiltPrompt {
  messages: ChatMessage[];
  /** Total characters of page text extracted. */
  totalChars: number;
  /** Characters of page text actually sent after truncation. */
  sentChars: number;
  /** True when page extraction stopped early to stay responsive. */
  sourceTruncated: boolean;
  /** True when the page did not fit and the middle was dropped. */
  truncated: boolean;
}

/**
 * Build the message array for /api/chat: a system message carrying the page context,
 * followed by the conversation. The page text is truncated to fit the context window
 * after accounting for the system prompt, conversation and answer reserve.
 * Returns coverage info so the UI can show how much of the page the model actually saw.
 */
export function buildMessages(
  settings: Settings,
  page: PageContent,
  conversation: ChatMessage[],
  /** Context window in tokens to budget the page text against. */
  ctxTokens: number,
): BuiltPrompt {
  const header = pageHeader(page);
  const systemBase = `${settings.systemPrompt}\n\n--- PAGE CONTEXT ---\n${header}\n\n--- PAGE CONTENT ---\n`;

  const fixedTokens =
    estimateTokens(systemBase) +
    conversation.reduce((sum, m) => sum + estimateTokens(m.content) + 8, 0) +
    ANSWER_RESERVE_TOKENS;

  const budgetForPage = Math.max(0, Math.min(ctxTokens - fixedTokens, MAX_PAGE_CONTEXT_TOKENS));
  const body = truncateToTokens(page.textContent, budgetForPage);

  const system: ChatMessage = { role: 'system', content: systemBase + body };
  const totalChars = page.textContent.length;
  // body includes a trim marker when truncated; treat any shrink as truncation.
  const truncated = estimateTokens(page.textContent) > budgetForPage;
  const sentChars = truncated ? Math.min(totalChars, budgetForPage * 4) : totalChars;

  return { messages: [system, ...conversation], totalChars, sentChars, sourceTruncated: Boolean(page.sourceTruncated), truncated };
}
