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
  /** This assistant message is an error notice (content is the message) — rendered with a
   *  warning icon instead of as markdown. */
  error?: boolean;
  /** Images attached to a user message (data: URLs, downscaled at attach time). Reach the
   *  engine as image_url content parts when the loaded model supports vision
   *  (lib/webllm.ts toEngineMessages). */
  images?: string[];
  /** Text files attached to a user message (content capped at attach time). Folded into the
   *  message text at the engine boundary; shown as chips in the chat. */
  files?: { name: string; text: string }[];
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

/** Hands-free "pause before responding": how long the mic must stay quiet before the utterance
 *  is sent to the model. Plumbed to the VAD's redemption window (lib/vadFrameProcessor.ts). */
export const MIN_VOICE_PAUSE_MS = 500;
export const MAX_VOICE_PAUSE_MS = 2500;
export const DEFAULT_VOICE_PAUSE_MS = 1200;

export interface Settings {
  /** UI theme. 'system' follows the OS/browser preference. */
  theme: Theme;
  /** In-browser (WebLLM) model id, e.g. "Qwen3-4B-q4f16_1-MLC". */
  webllmModel: string;
  /** In-browser context window (tokens). Higher uses more GPU memory; capped for stability. */
  webllmCtx: number;
  /** Load the chat model — and the voice models (TTS/STT), if previously downloaded — into
   *  memory when the browser starts (cache-only — never downloads). Off: models load on
   *  first use of the panel. */
  autoLoadOnStartup: boolean;
  /** Sampling temperature; low keeps answers grounded in the page. */
  temperature: number;
  /** Hidden system prompt prepended to every conversation. */
  systemPrompt: string;
  /** Send the current tab's content (title, URL, selection, text) with each question. Off:
   *  questions reach the model with no page data at all. */
  pageContext: boolean;
  /** Compress retrieved context with LLMLingua-2 before sending (experimental; off by default). */
  compressContext: boolean;
  /** Boost retrieved chunks currently visible in the viewport (experimental; off by default). */
  viewportBoost: boolean;
  /** Read each new assistant reply aloud automatically when it finishes (off by default). */
  ttsAutoRead: boolean;
  /** Push-to-talk transcripts are auto-submitted (true) vs dropped into the composer to edit
   *  (false, default). Hands-free voice mode always auto-submits regardless of this. */
  voiceAutoSend: boolean;
  /** Hands-free silence (ms) before an utterance is sent to the model (VAD redemption window). */
  voicePauseMs: number;
  /** Frequency penalty (0..1) passed to the model to curb repeated text (0 = off/default). */
  repetitionPenalty: number;
}

export const SYSTEM_PROMPT =
  'You are a helpful assistant embedded in a web browser. Answer the user based on the content of the web page provided in the context. Be concise and accurate. Format answers in GitHub-flavored Markdown. For mathematics, use LaTeX inside dollar-sign delimiters — $$...$$ for both inline and display math; never use \\[...\\] or \\(...\\) delimiters, and do not wrap answers in \\boxed{}.';

export const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  webllmModel: 'Qwen3-4B-q4f16_1-MLC',
  webllmCtx: DEFAULT_CONTEXT_TOKENS,
  autoLoadOnStartup: false,
  temperature: 0.3,
  systemPrompt: SYSTEM_PROMPT,
  pageContext: true,
  compressContext: false,
  viewportBoost: false,
  ttsAutoRead: false,
  voiceAutoSend: false,
  voicePauseMs: DEFAULT_VOICE_PAUSE_MS,
  repetitionPenalty: 0,
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
