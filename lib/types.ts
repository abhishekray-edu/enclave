// Shared types used across the side panel, content script and background worker.

export interface PageContent {
  title: string;
  url: string;
  /** Clean article text extracted by Readability (falls back to body innerText). */
  textContent: string;
  excerpt: string;
  byline: string;
  siteName: string;
  /** The user's current text selection on the page, if any. */
  selection: string;
}

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export type Theme = 'system' | 'light' | 'dark';

export interface Settings {
  /** UI theme. 'system' follows the OS/browser preference. */
  theme: Theme;
  /** In-browser (WebLLM) model id, e.g. "Qwen3-4B-q4f16_1-MLC". */
  webllmModel: string;
  /** In-browser context window (tokens). Higher uses more GPU memory; clamped per model. */
  webllmCtx: number;
  /** Sampling temperature; low keeps answers grounded in the page. */
  temperature: number;
  /** System prompt prepended to every conversation. */
  systemPrompt: string;
}

export const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  webllmModel: 'Qwen3-4B-q4f16_1-MLC',
  webllmCtx: 40960,
  temperature: 0.3,
  systemPrompt:
    'You are a helpful assistant embedded in a web browser. Answer the user using ONLY the ' +
    'content of the web page provided in the context. Be concise and accurate. If the answer ' +
    'is not contained in the page, say so plainly instead of guessing.',
};

// ---- Messaging protocol (runtime.sendMessage) ----

/** Sent by the side panel to a tab's content script to capture the page. */
export interface GetPageContentRequest {
  type: 'GET_PAGE_CONTENT';
}

/** A quick action queued by the background worker (e.g. from the context menu). */
export type QuickAction = 'ask' | 'summarize' | 'explain';

export interface PendingAction {
  action: QuickAction;
  /** Optional selected text captured at the time the action was triggered. */
  selection?: string;
}
