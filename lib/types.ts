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
  /** True when extraction stopped early to keep huge/noisy pages responsive. */
  sourceTruncated?: boolean;
  /** Lightweight document structure for chunking/RAG and structure-aware truncation.
   *  Optional: absent when produced by an older content script → callers fall back to textContent. */
  blocks?: PageBlock[];
  /** Text currently visible in the viewport at capture time (for viewport-aware ranking). */
  viewportText?: string;
}

/** A coarse structural unit of the page, in document order. */
export interface PageBlock {
  type: 'heading' | 'text';
  /** Heading level 1..6 (only for type 'heading'). */
  level?: number;
  text: string;
}

/** A chunk of page text produced by lib/chunking, used for RAG and summarization. */
export interface Chunk {
  /** Position in document order; stable id within a page. */
  ordinal: number;
  text: string;
  /** Nearest preceding heading, for provenance and prompt labeling. */
  heading?: string;
  tokensEstimate: number;
}

/** A chunk returned by retrieval, scored against a query (cosine, 0..1). */
export interface RetrievedChunk extends Chunk {
  score: number;
}

export type ChatRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  role: ChatRole;
  content: string;
  /** Structured (JSON) output for extraction tasks; content holds the raw JSON string. */
  structured?: { schemaId: string; data: unknown };
}

export type Theme = 'system' | 'light' | 'dark';

export const MIN_CONTEXT_TOKENS = 2048;
/** Upper bound on the in-browser context window the UI will request. Raised from 8192 so
 *  large pages can actually use a model's real context (most builds support up to 40960),
 *  while still being clamped per-model and bounded for GPU-memory (KV-cache) sanity. */
export const MAX_CONTEXT_TOKENS = 16384;
/** Default context window. Kept conservative (memory) and decoupled from the ceiling, so
 *  raising context to read bigger pages is an explicit, opt-in choice in Settings. */
export const DEFAULT_CONTEXT_TOKENS = 8192;

export interface Settings {
  /** UI theme. 'system' follows the OS/browser preference. */
  theme: Theme;
  /** In-browser (WebLLM) model id, e.g. "Qwen3-4B-q4f16_1-MLC". */
  webllmModel: string;
  /** In-browser context window (tokens). Higher uses more GPU memory; capped for stability. */
  webllmCtx: number;
  /** Sampling temperature; low keeps answers grounded in the page. */
  temperature: number;
  /** Hidden system prompt prepended to every conversation. */
  systemPrompt: string;
  /** Compress retrieved context with LLMLingua-2 before sending (experimental; off by default). */
  compressContext: boolean;
  /** Boost retrieved chunks currently visible in the viewport (experimental; off by default). */
  viewportBoost: boolean;
}

export const SYSTEM_PROMPT =
  'You are a helpful assistant embedded in a web browser. Answer the user based on the content of the web page provided in the context. Be concise and accurate.';

export const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  webllmModel: 'Qwen3-4B-q4f16_1-MLC',
  webllmCtx: DEFAULT_CONTEXT_TOKENS,
  temperature: 0.3,
  systemPrompt: SYSTEM_PROMPT,
  compressContext: false,
  viewportBoost: false,
};

// ---- Messaging protocol (runtime.sendMessage) ----

/** Sent by the side panel to a tab's content script to capture the page. */
export interface GetPageContentRequest {
  type: 'GET_PAGE_CONTENT';
  /** Also collect the text currently visible in the viewport (viewport-aware ranking). */
  wantViewport?: boolean;
}

/** A quick action queued by the background worker (e.g. from the context menu). */
export type QuickAction = 'ask' | 'summarize' | 'explain';

export interface PendingAction {
  action: QuickAction;
  /** Optional selected text captured at the time the action was triggered. */
  selection?: string;
}
