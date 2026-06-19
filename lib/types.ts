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

/** Which inference backend to use. */
export type Engine = 'ollama' | 'webllm';

export interface Settings {
  /** UI theme. 'system' follows the OS/browser preference. */
  theme: Theme;
  /** Inference backend: native Ollama, or in-browser WebGPU (WebLLM). */
  engine: Engine;
  /** Ollama base URL. */
  endpoint: string;
  /** Ollama model tag, e.g. "gemma3:4b". */
  model: string;
  /** In-browser (WebLLM) model id, e.g. "Qwen3-4B-q4f16_1-MLC". */
  webllmModel: string;
  /** In-browser context window (tokens). Higher uses more VRAM; bounded by the model build. */
  webllmCtx: number;
  /** Context window passed to Ollama (must be set or Ollama defaults to ~4K).
   *  Larger = more of the page is read, but prompt processing gets slower. */
  numCtx: number;
  /** Sampling temperature; low keeps answers grounded in the page. */
  temperature: number;
  /** System prompt prepended to every conversation. */
  systemPrompt: string;
}

export const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  engine: 'ollama',
  endpoint: 'http://localhost:11434',
  model: 'gemma3:4b',
  webllmModel: 'Qwen3-4B-q4f16_1-MLC',
  webllmCtx: 40960,
  numCtx: 32768,
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
