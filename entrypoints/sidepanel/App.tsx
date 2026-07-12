import { useEffect, useRef, useState } from 'react';
import { browser } from 'wxt/browser';
import { loadSettings, saveSettings } from '@/lib/settings';
import {
  WEBLLM_MODELS,
  PORT_NAME,
  defaultModelForDevice,
  effectiveContextWindow,
  webgpuAvailable,
  webllmModel,
  ensureOffscreen,
  releaseOffscreen,
  initModel,
  prewarmModel,
  streamGenerate,
  generateOnce,
  indexPage,
  retrieveChunks,
  compressTexts,
  hashText,
  type GenerateOptions,
  type LoadProgress,
  type WebllmPort,
} from '@/lib/webllmClient';
import { isLoadInterruption } from '@/lib/modelLoader';
import { ttsLoad, ttsSpeak, ttsSpeakStream, ttsStop, ttsRelease, TTS_DOWNLOAD_MB, type TtsStreamSession } from '@/lib/ttsClient';
import { SentenceStream } from '@/lib/sentences';
import {
  sttLoad,
  sttStartListening,
  sttRelease,
  micPermissionState,
  openMicPermissionPage,
  type SttSession,
  type SttState,
} from '@/lib/sttClient';
import { isLikelyEcho } from '@/lib/echoFilter';
import { voiceReducer, type VoiceState } from '@/lib/voiceReducer';
import { buildMessages, pageBudgetTokens, estimateTokens, VOICE_REPLY_DIRECTIVE, type BuiltPrompt } from '@/lib/prompt';
import { chunkPage } from '@/lib/chunking';
import { summarizeChunks } from '@/lib/summarize';
import { TASKS, extractTask, resolveTemperature, EXTRACTION_SCHEMAS, type TaskSpec, type ExtractionSchema } from '@/lib/tasks';
import { Markdown } from './Markdown';
import { Logo } from './Logo';
import { SettingsPanel } from './SettingsPanel';
import { HexSpinner } from './hex';
import { OnboardingCard } from './OnboardingCard';
import {
  MicIcon,
  HeadphonesIcon,
  SettingsIcon,
  FileTextIcon,
  LockIcon,
  PaperclipIcon,
  PlayIcon,
  StopIcon,
  AlertTriangleIcon,
  XIcon,
} from './icons';
import { applyTheme } from '@/lib/theme';
import {
  DEFAULT_SETTINGS,
  MAX_CONTEXT_TOKENS,
  type ChatMessage,
  type GetPageContentRequest,
  type PageContent,
  type PendingAction,
  type Settings,
} from '@/lib/types';

const PENDING_KEY = 'pendingAction';
/** Human labels for the hands-free voice loop's states (see lib/voiceReducer.ts). */
const VOICE_STATE_LABEL: Record<VoiceState, string> = {
  off: '',
  listening: 'Listening…',
  speech: 'Heard you…',
  transcribing: 'Transcribing…',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
};
/** Persists which models have an interrupted (resumable) download, so the paused state
 *  survives the panel being recreated on close/reopen. */
const PAUSED_MODELS_KEY = 'pausedDownloads';
/** Persisted onboarding preference: download the voice models with the LLM (default on). */
const ONBOARD_VOICE_KEY = 'onboardVoicePref';
const PAGE_CAPTURE_TIMEOUT_MS = 8_000;
const FIRST_TOKEN_TIMEOUT_MS = 90_000;
const NEXT_TOKEN_TIMEOUT_MS = 45_000;
const AUTO_SCROLL_BOTTOM_PX = 48;

// chrome.scripting isn't in the polyfill types; access the global directly.
const scripting = (
  globalThis as unknown as {
    chrome: {
      scripting: {
        executeScript(opts: { target: { tabId: number }; files: string[] }): Promise<unknown>;
      };
    };
  }
).chrome.scripting;

function withTimeout<T>(promise: Promise<T>, ms: number, message: string, onTimeout?: () => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      onTimeout?.();
      reject(new Error(message));
    }, ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/** Capture failed because no readable page is there (chrome://, the Web Store, PDFs, local
 *  files, or no active tab). Free-form questions degrade to a page-context-off turn on this
 *  error instead of failing; page-dependent tasks still surface it. */
class PageUnreadableError extends Error {}

/** Capture the active tab's content. If the content script isn't present (e.g. the tab was
 *  open before the extension was installed/updated), inject it on demand and retry. */
async function capturePage(wantViewport = false): Promise<PageContent> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new PageUnreadableError('No active tab.');
  const tabId = tab.id;
  const ask = () =>
    browser.tabs.sendMessage(tabId, {
      type: 'GET_PAGE_CONTENT',
      wantViewport,
    } satisfies GetPageContentRequest) as Promise<PageContent>;

  try {
    return await ask();
  } catch {
    // Content script not loaded in this tab — inject it, then retry once.
    try {
      await scripting.executeScript({
        target: { tabId },
        files: ['content-scripts/content.js'],
      });
      return await ask();
    } catch {
      throw new PageUnreadableError(
        "Can't read this page. Restricted pages (chrome://, the Chrome Web Store, PDFs, and local files) can't be read.",
      );
    }
  }
}

// ---- Attachments: images (vision model only) and text files (any model) staged in the
//      composer, sent with the next free-form question. ----
interface Attachment {
  kind: 'image' | 'text';
  name: string;
  /** kind 'image': downscaled JPEG data URL, ready for the engine's image_url part. */
  dataUrl?: string;
  /** kind 'text': file contents (storage-capped; prompt-capped again per model at send). */
  text?: string;
}

/** Letterbox canvases with aspect ratios that are SAFE for Phi-3.5-vision. The engine
 *  resizes every image by ASPECT RATIO alone — scaling it up to fill a grid of 336px tiles
 *  (max 16) — and the whole image embedding must prefill in one indivisible chunk of the
 *  compiled 2048-token budget. Square and 16:9 inputs blow it (2509 / 2353 tokens →
 *  PrefillChunkSizeSmallerThanImageError). Each canvas's token cost is FIXED by its ratio
 *  (pixel count is irrelevant); the attach path picks the closest ratio, which minimizes
 *  both padding and prefill time — a phone screenshot on the 1:2 canvas costs 1357 tokens
 *  (and lags visibly less) than it would on 3:4. */
const VISION_CANVASES = [
  { w: 1344, h: 1008, tokens: 1921 }, // 4:3 — near-square and most landscape images
  { w: 1680, h: 672, tokens: 1621 }, // 5:2 — ultrawide screenshots and banners
  { w: 1008, h: 1344, tokens: 1933 }, // 3:4 — near-square portrait
  { w: 672, h: 1344, tokens: 1357 }, // 1:2 — phone screenshots
];
/** Files per message (keeps the prompt reviewable; the token cap below is the real bound). */
const MAX_ATTACHED_FILES = 4;
/** Reject files over this size before reading them (likely not text at all). */
const MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024;
/** Storage cap per file at attach time; the per-model token cap applies again at send. */
const MAX_STORED_FILE_CHARS = 120_000;
const TEXT_FILE_ACCEPT =
  '.txt,.md,.markdown,.csv,.tsv,.json,.log,.xml,.yaml,.yml,.html,.css,.js,.jsx,.ts,.tsx,.py,.rb,.go,.rs,.java,.c,.h,.cpp,.sh,.toml,.ini,.sql';

/** Decode an image file and letterbox it onto the nearest safe vision canvas, returning a
 *  JPEG data URL. See VISION_CANVASES for why the ratio is forced. */
async function imageFileToDataUrl(file: File): Promise<string> {
  const bitmap = await createImageBitmap(file);
  try {
    const ratio = bitmap.width / bitmap.height;
    // Least-padding canvas: minimize the ratio mismatch factor (always ≥ 1).
    const mismatch = (c: { w: number; h: number }) => Math.max(ratio / (c.w / c.h), c.w / c.h / ratio);
    let spec = VISION_CANVASES[0];
    for (const c of VISION_CANVASES) if (mismatch(c) < mismatch(spec)) spec = c;
    const { w, h } = spec;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas unavailable.');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    const scale = Math.min(w / bitmap.width, h / bitmap.height);
    const dw = Math.max(1, Math.round(bitmap.width * scale));
    const dh = Math.max(1, Math.round(bitmap.height * scale));
    ctx.drawImage(bitmap, Math.floor((w - dw) / 2), Math.floor((h - dh) / 2), dw, dh);
    return canvas.toDataURL('image/jpeg', 0.85);
  } finally {
    bitmap.close();
  }
}

/** Cap total attached-file text to the model's safe single-prefill budget (~4 chars/token,
 *  matching estimateTokens) — one oversized prompt on a big model can stall the GPU past the
 *  OS watchdog, the same hazard safePromptTokens guards the page body against. */
function capAttachedFiles(files: { name: string; text: string }[], maxTokens: number): { name: string; text: string }[] {
  let budget = maxTokens * 4;
  return files.map((f) => {
    const take = Math.max(0, budget);
    budget -= f.text.length;
    return f.text.length <= take
      ? f
      : { name: f.name, text: f.text.slice(0, take) + '\n…[file truncated to fit the model context]' };
  });
}

/** Remove Qwen-style <think>…</think> reasoning from displayed output.
 *  While a block is still open mid-stream, hide everything from it onward. */
function stripThink(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  const open = out.lastIndexOf('<think>');
  if (open !== -1 && !/<\/think>/i.test(out.slice(open))) out = out.slice(0, open);
  return out.replace(/^\s+/, '');
}

/** Reduce displayed markdown to plain prose for speech — drop code blocks and syntax so the
 *  TTS voice reads words, not backticks and link URLs. */
function toPlainText(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, ' ') // fenced code blocks
    .replace(/`([^`]+)`/g, '$1') // inline code
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ') // images
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') // links → link text
    .replace(/^#{1,6}\s+/gm, '') // heading markers
    .replace(/^\s*[-*+]\s+/gm, '') // bullet markers
    .replace(/^\s*\d+\.\s+/gm, '') // ordered-list markers
    .replace(/[*_~>#]/g, '') // emphasis / blockquote marks
    .replace(/\s+/g, ' ')
    .trim();
}

function contextForSettings(s: Settings): number {
  return effectiveContextWindow(s.webllmModel, s.webllmCtx);
}

function isNearBottom(el: HTMLElement): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= AUTO_SCROLL_BOTTOM_PX;
}

function ThinkingIndicator() {
  return (
    <div className="thinking-indicator" role="status" aria-label="Thinking">
      <span className="thinking-indicator__spinner" aria-hidden="true" />
      <span className="thinking-indicator__label">
        Thinking<span className="thinking-indicator__dots" aria-hidden="true" />
      </span>
      <span className="thinking-indicator__wave" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

/** Read-aloud control shown under each finished assistant reply. Shows a download percentage
 *  the first time (while the ~178 MB voice model fetches), then toggles Speak ⇄ Stop. */
function SpeakButton({
  active,
  downloading,
  downloadPct,
  disabled,
  onClick,
}: {
  active: boolean;
  downloading: boolean;
  downloadPct: number;
  disabled: boolean;
  onClick: () => void;
}) {
  const label = downloading ? `Downloading voice… ${downloadPct}%` : active ? 'Stop' : 'Speak';
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={active ? 'Stop reading' : `Read this reply aloud${!downloading ? ` (downloads a ~${TTS_DOWNLOAD_MB} MB voice on first use)` : ''}`}
      className="mt-2 inline-flex items-center gap-1 rounded border border-zinc-300 px-2 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
    >
      {active && !downloading ? <StopIcon size={11} /> : <PlayIcon size={11} />}
      {label}
    </button>
  );
}

function renderScalar(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

/** Recursively render a JSON value as nested key/value and array tables. */
function StructuredValue({ value }: { value: unknown }) {
  if (value == null) return <span className="text-zinc-400">—</span>;
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-zinc-400">—</span>;
    if (value.every((x) => x == null || typeof x !== 'object')) {
      return <span>{value.map(renderScalar).join(', ')}</span>;
    }
    const cols = Array.from(
      new Set(value.flatMap((o) => (o && typeof o === 'object' ? Object.keys(o as object) : []))),
    );
    return (
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c} className="border-b border-zinc-200 px-1.5 py-1 text-left font-medium dark:border-zinc-700">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {value.map((o, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c} className="border-b border-zinc-100 px-1.5 py-1 align-top dark:border-zinc-800">
                  {renderScalar((o as Record<string, unknown>)?.[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  if (typeof value === 'object') {
    return (
      <table className="w-full border-collapse text-xs">
        <tbody>
          {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
            <tr key={k}>
              <td className="border-b border-zinc-100 px-1.5 py-1 align-top font-medium text-zinc-600 dark:border-zinc-800 dark:text-zinc-300">
                {k}
              </td>
              <td className="border-b border-zinc-100 px-1.5 py-1 align-top dark:border-zinc-800">
                <StructuredValue value={v} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  return <span>{String(value)}</span>;
}

/** Render extracted JSON as tables with a copy affordance; falls back to raw text on parse failure. */
function StructuredResult({ data, raw }: { data: unknown; raw: string }) {
  const [copied, setCopied] = useState(false);
  if (data == null || typeof data !== 'object') {
    return <pre className="overflow-x-auto rounded bg-zinc-50 p-2 text-xs dark:bg-zinc-900/50">{raw}</pre>;
  }
  const copy = () => {
    void navigator.clipboard?.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div className="space-y-2">
      <StructuredValue value={data} />
      <button
        onClick={copy}
        className="rounded border border-zinc-300 px-2 py-0.5 text-[11px] text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
      >
        {copied ? 'Copied' : 'Copy JSON'}
      </button>
    </div>
  );
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  /** Transient attach feedback (wrong type, too big, needs the vision model). */
  const [attachNote, setAttachNote] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pageNote, setPageNote] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
  /** Model lifecycle shown in the header: needs-download → downloading → loading → ready. */
  const [modelStatus, setModelStatus] = useState<'idle' | 'needs-download' | 'downloading' | 'loading' | 'ready'>('idle');
  /** Error surfaced inside the first-run card (download failures). */
  const [onboardError, setOnboardError] = useState<string | null>(null);
  // Onboarding voice opt-in: download STT + TTS alongside the LLM so the record button works
  // instantly. `onboardVoice` (persisted) is the checkbox; the init errors + micGranted drive the
  // per-item rows and the mic-permission step in the card.
  const [onboardVoice, setOnboardVoice] = useState(true);
  const onboardVoiceRef = useRef(true);
  const [sttInitError, setSttInitError] = useState<string | null>(null);
  const [ttsInitError, setTtsInitError] = useState<string | null>(null);
  const [micGranted, setMicGranted] = useState(false);
  /** Models whose download was interrupted (superseded by switching to another model) and
   *  is resumable. Their fetched weight shards stay cached, so re-downloading resumes. */
  const [pausedModels, setPausedModels] = useState<string[]>([]);
  // Text-to-speech (pocket-tts). The voice model downloads once on first Speak; `speakingIdx`
  // marks which message is currently being read (or downloading) so its button shows Stop.
  const [ttsReady, setTtsReady] = useState(false);
  const [ttsDownloading, setTtsDownloading] = useState(false);
  const [ttsDownloadPct, setTtsDownloadPct] = useState(0);
  const [speakingIdx, setSpeakingIdx] = useState<number | null>(null);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const ttsReadyRef = useRef(false);
  const ttsDownloadingRef = useRef(false);
  const speakingIdxRef = useRef<number | null>(null);
  // Streamed speech: the active session while a reply is spoken sentence-by-sentence as it
  // generates. `voiceReveal` gates how much of the reply's text is shown so it appears in step
  // with the spoken audio (voice mode only). `spokenRef` retains recently spoken sentences for
  // the barge-in echo filter.
  const ttsStreamRef = useRef<TtsStreamSession | null>(null);
  const [voiceReveal, setVoiceReveal] = useState<{ idx: number; end: number } | null>(null);
  const revealMapRef = useRef<Map<number, number>>(new Map());
  const spokenRef = useRef<{ plain: string; at: number }[]>([]);
  // Barge-in: `bargeInRef` is set when the user talks over a reply (so the aborting turn doesn't
  // resume the idle loop); `pendingVoiceTranscriptRef` holds a transcript that arrived while the
  // prior generation was still unwinding, consumed as the next turn in submit's finally.
  const bargeInRef = useRef(false);
  const pendingVoiceTranscriptRef = useRef<string | null>(null);
  // Voice input (Moonshine STT + Silero VAD). The speech model downloads once on first use.
  // `voiceState` drives the hands-free loop (off → listening → speech → transcribing → thinking
  // → speaking → listening); `pttActive` is the separate push-to-talk mic button.
  const [voiceState, setVoiceState] = useState<VoiceState>('off');
  const voiceStateRef = useRef<VoiceState>('off');
  const [sttReady, setSttReady] = useState(false);
  const sttReadyRef = useRef(false);
  const [sttDownloading, setSttDownloading] = useState(false);
  const [sttDownloadPct, setSttDownloadPct] = useState(0);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [pttActive, setPttActive] = useState(false);
  const pttActiveRef = useRef(false);
  /** Which control is mid-first-download (drives its inline % label). */
  const [voiceLoading, setVoiceLoading] = useState<null | 'ptt' | 'voice'>(null);
  const voiceSessionRef = useRef<SttSession | null>(null);
  const suggestedModelId = defaultModelForDevice();
  /** The model card is the one place downloads happen: it shows whenever the selected
   *  model's weights aren't on disk — on first run and mid-chat alike. */
  const onboardingVisible = modelStatus === 'needs-download' || modelStatus === 'downloading';

  // Long-lived port to the offscreen document that hosts the in-browser engine.
  const portRef = useRef<WebllmPort | null>(null);
  const reqRef = useRef(0);
  const preloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const prewarmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastIndexedHashRef = useRef<string | null>(null);

  // Refs keep submit() free of stale closures (settings load async; pending action auto-runs).
  const settingsRef = useRef(settings);
  const messagesRef = useRef(messages);
  const attachmentsRef = useRef(attachments);
  settingsRef.current = settings;
  messagesRef.current = messages;
  attachmentsRef.current = attachments;
  onboardVoiceRef.current = onboardVoice;
  // Gate persisting paused models until the stored value has been loaded, so the initial
  // empty state never overwrites what's on disk.
  const pausedHydratedRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const attachNoteTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);

  // Engine operations (download, preload, prewarm, question) can overlap — e.g. switching
  // back to a cached model while another downloads. Whichever started last owns the
  // status/progress UI: older operations' writes are stale and stay silent.
  const engineOpEpochRef = useRef(0);
  const engineOpRef = useRef<number | null>(null);

  /** Claim ownership of the model status/progress UI for one engine operation. */
  function beginEngineOp(): number {
    const epoch = ++engineOpEpochRef.current;
    engineOpRef.current = epoch;
    return epoch;
  }

  function endEngineOp(epoch: number) {
    if (engineOpRef.current === epoch) engineOpRef.current = null;
  }

  /** True while no newer engine operation has started since `epoch`. */
  function engineOpCurrent(epoch: number): boolean {
    return engineOpEpochRef.current === epoch;
  }

  // Initial load: settings, then any queued context-menu action.
  useEffect(() => {
    (async () => {
      const s = await loadSettings();
      setSettings(s);
      settingsRef.current = s;

      // Rehydrate paused (interrupted) downloads before anything can persist over them, so a
      // partially-downloaded model still reads as "download paused" after a close/reopen.
      const pausedStored = await browser.storage.local.get(PAUSED_MODELS_KEY);
      const paused = pausedStored[PAUSED_MODELS_KEY];
      if (Array.isArray(paused)) {
        setPausedModels(paused.filter((id): id is string => typeof id === 'string'));
      }
      pausedHydratedRef.current = true;

      // Onboarding voice preference + current mic permission (drives the first-run voice section).
      const voiceStored = await browser.storage.local.get(ONBOARD_VOICE_KEY);
      if (typeof voiceStored[ONBOARD_VOICE_KEY] === 'boolean') {
        setOnboardVoice(voiceStored[ONBOARD_VOICE_KEY]);
        onboardVoiceRef.current = voiceStored[ONBOARD_VOICE_KEY];
      }
      try {
        if ((await micPermissionState()) === 'granted') setMicGranted(true);
      } catch {
        /* permissions API unavailable — the mic step just falls back to first-use prompt */
      }

      const stored = await browser.storage.local.get(PENDING_KEY);
      const pending = stored[PENDING_KEY] as PendingAction | undefined;
      if (pending) {
        await browser.storage.local.remove(PENDING_KEY);
        if (pending.action === 'explain') {
          void submit(TASKS.explain.label, {
            task: TASKS.explain,
            selectionOverride: pending.selection,
          });
        }
      }
      // Resolve the model card immediately — don't wait on the debounce or (slow) page
      // capture, so a fresh install sees the download card the moment the panel opens.
      void stageModel();
      // Index the current page in the background for the first question.
      schedulePrewarm(300);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // The model card renders at the end of the thread; when it appears mid-chat (a switch
  // to a not-yet-downloaded model), bring it into view.
  useEffect(() => {
    if (!onboardingVisible) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [onboardingVisible]);

  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

  // A model has a resumable (incomplete) download from the moment its download starts until it
  // finishes: mark it while downloading, clear it once ready. Any interruption in between — a
  // model switch, closing the panel, a crash — thus leaves it marked, so it reads as "paused"
  // (its fetched shards stay cached) rather than a first-time download on the next open.
  useEffect(() => {
    const model = settings.webllmModel;
    if (modelStatus === 'downloading') {
      setPausedModels((prev) => (prev.includes(model) ? prev : [...prev, model]));
    } else if (modelStatus === 'ready') {
      setPausedModels((prev) => (prev.includes(model) ? prev.filter((id) => id !== model) : prev));
    }
  }, [modelStatus, settings.webllmModel]);

  // Persist paused downloads so they survive the panel being recreated on close/reopen.
  useEffect(() => {
    if (!pausedHydratedRef.current) return;
    void browser.storage.local.set({ [PAUSED_MODELS_KEY]: pausedModels });
  }, [pausedModels]);

  useEffect(() => {
    return () => {
      cancelScheduledPreload();
    };
  }, []);

  /** Ensure the offscreen document exists and we hold a live port to it. */
  async function getWebllmPort(): Promise<WebllmPort> {
    if (portRef.current) return portRef.current;
    await ensureOffscreen();
    const port = browser.runtime.connect({ name: PORT_NAME });
    port.onDisconnect.addListener(() => {
      if (portRef.current === port) portRef.current = null;
      // The offscreen doc (and its mic capture) is gone — drop any voice session so the UI
      // doesn't show a live indicator for a mic that's no longer running.
      resetVoiceState();
    });
    portRef.current = port;
    return port;
  }

  /** Free the model from memory by closing the offscreen document. */
  async function releaseModel() {
    cancelScheduledPreload();
    const epoch = beginEngineOp();
    // Closing the offscreen doc destroys the TTS + STT workers too — reset so the next use reloads.
    ttsReadyRef.current = false;
    setTtsReady(false);
    setSpeakingIdx(null);
    speakingIdxRef.current = null;
    resetVoiceState();
    portRef.current?.disconnect();
    portRef.current = null;
    await releaseOffscreen();
    if (engineOpCurrent(epoch)) {
      setModelStatus('idle');
      setPageNote('Model released — it will reload on your next question.');
    }
    endEngineOp(epoch);
  }

  /** Proactively (re)load the model so it's ready before the next message — but only when
   *  its weights are already on disk. Switching to a not-yet-downloaded model shows the
   *  first-run card instead of silently starting a multi-GB download. An explicit switch
   *  supersedes a different model still mid-load: the newest choice wins, and a canceled
   *  download resumes later from its cached shards. */
  async function preloadModel(s: Settings) {
    if (!webgpuAvailable() || abortRef.current) return;
    const epoch = beginEngineOp();
    try {
      const port = await getWebllmPort();
      if (engineOpCurrent(epoch)) setModelStatus('loading');
      const loaded = await prewarmModel(
        port,
        s.webllmModel,
        contextForSettings(s),
        (p) => {
          if (engineOpCurrent(epoch)) setLoadProgress(p);
        },
        { supersede: true },
      );
      if (engineOpCurrent(epoch)) setModelStatus(loaded ? 'ready' : 'needs-download');
    } catch {
      if (engineOpCurrent(epoch)) setModelStatus('idle'); /* a real error surfaces on the next send */
    } finally {
      if (engineOpCurrent(epoch)) setLoadProgress(null);
      endEngineOp(epoch);
    }
  }

  /** Explicit model download from the first-run card — this and a submitted question are
   *  the only two places a multi-GB download may start. */
  async function downloadModel() {
    const s = settingsRef.current;
    const epoch = beginEngineOp();
    setOnboardError(null);
    try {
      if (!webgpuAvailable()) {
        throw new Error('This browser has no WebGPU. Use a recent Chromium browser (Chrome, Edge, or Brave).');
      }
      const port = await getWebllmPort();
      // Voice models download in parallel (independent request-id spaces → separate workers), so
      // the record button is warm the moment onboarding finishes. Best-effort; never blocks chat.
      if (onboardVoiceRef.current) void startVoiceOnboarding(port);
      if (engineOpCurrent(epoch)) setModelStatus('downloading');
      await initModel(port, s.webllmModel, contextForSettings(s), (p) => {
        if (engineOpCurrent(epoch)) setLoadProgress(p);
      });
      if (engineOpCurrent(epoch)) setModelStatus('ready');
    } catch (e) {
      // A canceled or superseded load isn't a failure — whatever replaced it owns the UI.
      if (engineOpCurrent(epoch) && !isLoadInterruption(e)) {
        setModelStatus('needs-download');
        setOnboardError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (engineOpCurrent(epoch)) setLoadProgress(null);
      endEngineOp(epoch);
    }
  }

  /** Persist the onboarding voice preference and reflect it in the ref. */
  function toggleOnboardVoice(enabled: boolean) {
    setOnboardVoice(enabled);
    onboardVoiceRef.current = enabled;
    void browser.storage.local.set({ [ONBOARD_VOICE_KEY]: enabled });
  }

  /** Download + load the speech-recognition model for onboarding (errors shown inline in the
   *  card, not the runtime strip). No-op if already loaded. */
  async function downloadSttOnboard(port: WebllmPort) {
    if (sttReadyRef.current) return;
    setSttInitError(null);
    setSttDownloading(true);
    setSttDownloadPct(0);
    try {
      await sttLoad(port, (p) => setSttDownloadPct(Math.round(p * 100)));
      sttReadyRef.current = true;
      setSttReady(true);
    } catch (e) {
      setSttInitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSttDownloading(false);
    }
  }

  /** Download + load the TTS voice for onboarding (errors shown inline in the card). */
  async function downloadTtsOnboard(port: WebllmPort) {
    if (ttsReadyRef.current) return;
    setTtsInitError(null);
    setTtsDownloading(true);
    ttsDownloadingRef.current = true;
    setTtsDownloadPct(0);
    try {
      await ttsLoad(port, (p) => setTtsDownloadPct(Math.round(p * 100)));
      ttsReadyRef.current = true;
      setTtsReady(true);
    } catch (e) {
      setTtsInitError(e instanceof Error ? e.message : String(e));
    } finally {
      setTtsDownloading(false);
      ttsDownloadingRef.current = false;
    }
  }

  /** Kick off both speech-model downloads in parallel with the LLM download, then surface the
   *  mic-permission step once they're ready. Best-effort — a failed item shows Retry and never
   *  blocks the chat. */
  async function startVoiceOnboarding(port: WebllmPort) {
    await Promise.allSettled([downloadSttOnboard(port), downloadTtsOnboard(port)]);
    try {
      if ((await micPermissionState()) === 'granted') setMicGranted(true);
    } catch {
      /* ignore — first voice use will prompt */
    }
  }

  /** Debounce reloads so editing the context number doesn't reload on every keystroke. */
  function schedulePreload(s: Settings) {
    cancelScheduledPreload();
    preloadTimerRef.current = setTimeout(() => {
      preloadTimerRef.current = null;
      void preloadModel(s);
    }, 700);
  }

  function cancelScheduledPreload() {
    if (!preloadTimerRef.current) return;
    clearTimeout(preloadTimerRef.current);
    preloadTimerRef.current = null;
  }

  /** Resolve the model's state — needs-download vs ready — and, when its weights are already
   *  cached, load it onto the GPU. This is the fast path that drives the model card, so it
   *  runs on its own and is NEVER gated behind page capture (a deep DOM walk that can take
   *  several seconds): on a fresh install the card must appear the moment the panel opens.
   *  A missing model is never downloaded here — the card (or an explicit question) does that. */
  async function stageModel() {
    const s = settingsRef.current;
    // A background warm-up never contends with a generation or another engine operation
    // (e.g. an explicit download): it must not steal their status or cancel their load.
    if (!webgpuAvailable() || abortRef.current || engineOpRef.current !== null) return;
    const epoch = beginEngineOp();
    try {
      const port = await getWebllmPort();
      if (abortRef.current || !engineOpCurrent(epoch)) return;
      setModelStatus((cur) => (cur === 'ready' ? cur : 'loading'));
      const loaded = await prewarmModel(
        port,
        s.webllmModel,
        contextForSettings(s),
        (p) => {
          if (engineOpCurrent(epoch)) setLoadProgress(p);
        },
        {
          // A download already running in the offscreen doc (e.g. the panel was reopened
          // mid-download) is resumed here: show it as downloading so the card and its
          // progress reappear instead of a bare "needs download".
          onStart: (downloading) => {
            if (downloading && engineOpCurrent(epoch)) setModelStatus('downloading');
          },
        },
      );
      if (engineOpCurrent(epoch)) setModelStatus(loaded ? 'ready' : 'needs-download');
    } catch {
      // A failed prewarm never demotes a model that is already resident.
      if (engineOpCurrent(epoch)) setModelStatus((cur) => (cur === 'ready' ? cur : 'idle'));
    } finally {
      if (engineOpCurrent(epoch)) setLoadProgress(null);
      endEngineOp(epoch);
    }
  }

  /** Index the current page for retrieval when it's over the model's single-prompt budget.
   *  Best-effort CPU work (capture + chunk + embed) that runs independently of model staging
   *  and never gates the model card or status. */
  async function indexCurrentPage() {
    const s = settingsRef.current;
    // Page context off means the page is never read — not even for background indexing.
    if (!s.pageContext || !webgpuAvailable() || abortRef.current) return;
    // Capture fails on restricted pages (chrome://, stores, or when no readable tab is
    // active) — that's fine, there's just nothing to index.
    let page: PageContent | null = null;
    try {
      page = await capturePage();
    } catch {
      return;
    }
    try {
      const hash = hashText(page.textContent);
      if (
        lastIndexedHashRef.current !== hash &&
        estimateTokens(page.textContent) > webllmModel(s.webllmModel).safePromptTokens
      ) {
        const port = await getWebllmPort();
        await indexPage(port, page.url, hash, chunkPage(page), () => {});
        lastIndexedHashRef.current = hash;
      }
    } catch {
      /* indexing is best-effort; a real error surfaces on the next question */
    }
  }

  /** Quietly prepare for the first question: resolve/stage the model (fast) and, separately,
   *  index the current page (slow). Errors are swallowed; nothing runs while a generation
   *  streams. Staging leads so the model card is never delayed by page capture. */
  async function prewarm() {
    await stageModel();
    await indexCurrentPage();
  }

  /** Debounced prewarm: tab switches and page loads arrive in bursts. */
  function schedulePrewarm(delayMs = 1200) {
    if (prewarmTimerRef.current) clearTimeout(prewarmTimerRef.current);
    prewarmTimerRef.current = setTimeout(() => {
      prewarmTimerRef.current = null;
      void prewarm();
    }, delayMs);
  }

  // Re-prewarm as the user moves between tabs / navigates, while the panel is open.
  useEffect(() => {
    const onActivated = () => schedulePrewarm();
    const onUpdated = (_tabId: number, info: { status?: string }, tab: { active?: boolean }) => {
      if (info.status === 'complete' && tab.active) schedulePrewarm();
    };
    browser.tabs.onActivated.addListener(onActivated);
    browser.tabs.onUpdated.addListener(onUpdated);
    return () => {
      browser.tabs.onActivated.removeListener(onActivated);
      browser.tabs.onUpdated.removeListener(onUpdated);
      if (prewarmTimerRef.current) clearTimeout(prewarmTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setLastAssistant(content: string) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = { role: 'assistant', content };
      return copy;
    });
  }

  /** Replace the streaming assistant slot with an error notice (rendered with a warning icon). */
  function setLastAssistantError(message: string) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = { role: 'assistant', content: message, error: true };
      return copy;
    });
  }

  function onChatScroll() {
    const el = scrollRef.current;
    if (el) shouldAutoScrollRef.current = isNearBottom(el);
  }

  /** Stop any in-flight speech (streamed session or one-shot). */
  function stopSpeaking() {
    ttsStreamRef.current?.cancel();
    ttsStreamRef.current = null;
    if (portRef.current) ttsStop(portRef.current);
    setSpeakingIdx(null);
    speakingIdxRef.current = null;
    setVoiceReveal(null);
  }

  /** Read message `idx` aloud. First call downloads the ~178 MB voice model (once). Clicking
   *  the same message while it speaks stops it; speaking a different one interrupts the first. */
  async function speakMessage(idx: number, rawText: string) {
    const plain = toPlainText(stripThink(rawText));
    if (!plain) return;
    const port = await getWebllmPort();

    // Toggle off when this same message is already speaking.
    if (speakingIdxRef.current === idx && !ttsDownloadingRef.current) {
      stopSpeaking();
      return;
    }
    // Interrupt whatever else is speaking.
    if (speakingIdxRef.current != null) ttsStop(port);

    setTtsError(null);
    setSpeakingIdx(idx);
    speakingIdxRef.current = idx;

    if (!ttsReadyRef.current) {
      if (!(await ensureTtsReady(port))) {
        if (speakingIdxRef.current === idx) {
          setSpeakingIdx(null);
          speakingIdxRef.current = null;
        }
        return;
      }
    }
    // The user may have stopped/switched during the download.
    if (speakingIdxRef.current !== idx) return;

    try {
      await ttsSpeak(port, plain);
    } catch (e) {
      setTtsError(e instanceof Error ? e.message : String(e));
    } finally {
      if (speakingIdxRef.current === idx) {
        setSpeakingIdx(null);
        speakingIdxRef.current = null;
      }
    }
  }

  /** Ensure the voice (TTS) model is downloaded (once) and loaded. Mirrors ensureSttReady. */
  async function ensureTtsReady(port: WebllmPort): Promise<boolean> {
    if (ttsReadyRef.current) return true;
    setTtsDownloading(true);
    ttsDownloadingRef.current = true;
    setTtsDownloadPct(0);
    try {
      await ttsLoad(port, (p) => setTtsDownloadPct(Math.round(p * 100)));
      ttsReadyRef.current = true;
      setTtsReady(true);
      return true;
    } catch (e) {
      setTtsError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setTtsDownloading(false);
      ttsDownloadingRef.current = false;
    }
  }

  /** Speak the just-finished reply when auto-read is enabled (skips structured/empty). */
  function autoRead(idx: number, text: string) {
    if (settingsRef.current.ttsAutoRead && text.trim()) void speakMessage(idx, text);
  }

  /** Free the voice model's sessions (Settings). The offscreen worker unloads; next Speak reloads. */
  function releaseVoiceModel() {
    if (portRef.current) ttsRelease(portRef.current);
    stopSpeaking();
    ttsReadyRef.current = false;
    setTtsReady(false);
  }

  // ---- Voice input (speech-to-text) ----

  /** Set the hands-free voice state (keeping the ref in sync for stale-closure-free reads). */
  function setVoice(next: VoiceState) {
    voiceStateRef.current = next;
    setVoiceState(next);
  }

  function dispatchVoice(action: Parameters<typeof voiceReducer>[1]) {
    setVoice(voiceReducer(voiceStateRef.current, action));
  }

  /** Ensure the speech model is downloaded (once) and loaded. Mirrors speakMessage's TTS block. */
  async function ensureSttReady(port: WebllmPort): Promise<boolean> {
    if (sttReadyRef.current) return true;
    setSttDownloading(true);
    setSttDownloadPct(0);
    try {
      await sttLoad(port, (p) => setSttDownloadPct(Math.round(p * 100)));
      sttReadyRef.current = true;
      setSttReady(true);
      return true;
    } catch (e) {
      setVoiceError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setSttDownloading(false);
    }
  }

  /** Confirm mic access, opening the one-time permission page (in a normal tab) if needed and
   *  awaiting its result. getUserMedia can't prompt from the panel or the offscreen document. */
  async function ensureMicPermission(): Promise<boolean> {
    if ((await micPermissionState()) === 'granted') return true;
    return new Promise<boolean>((resolve) => {
      let settled = false;
      const finish = (granted: boolean) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        browser.runtime.onMessage.removeListener(onMsg);
        resolve(granted);
      };
      const onMsg = (m: unknown) => {
        const msg = m as { type?: string; granted?: boolean };
        if (msg?.type === 'MIC_PERMISSION_RESULT') finish(!!msg.granted);
      };
      // Give up (as denied) if the user abandons the page without choosing.
      const timer = setTimeout(async () => {
        finish((await micPermissionState()) === 'granted');
      }, 120_000);
      browser.runtime.onMessage.addListener(onMsg);
      void openMicPermissionPage();
    });
  }

  /** Push-to-talk: click to start capturing, click again to stop + transcribe. The transcript
   *  either fills the composer (default) or is submitted (Settings → voiceAutoSend). */
  async function togglePtt() {
    if (voiceStateRef.current !== 'off') return; // hands-free mode owns the mic
    if (pttActiveRef.current) {
      voiceSessionRef.current?.stop(true);
      return;
    }
    setVoiceError(null);
    setVoiceLoading('ptt');
    try {
      if (!(await ensureMicPermission())) {
        setVoiceError('Microphone access is needed for voice input. Allow it and try again.');
        return;
      }
      const port = await getWebllmPort();
      if (!(await ensureSttReady(port))) return;
      pttActiveRef.current = true;
      setPttActive(true);
      voiceSessionRef.current = sttStartListening(port, {
        mode: 'ptt',
        onTranscript: (text) => onPttTranscript(text),
        onError: (e) => setVoiceError(e.message),
        onStopped: () => {
          pttActiveRef.current = false;
          setPttActive(false);
          voiceSessionRef.current = null;
        },
      });
    } finally {
      setVoiceLoading(null);
    }
  }

  function onPttTranscript(text: string) {
    const t = text.trim();
    if (!t) return;
    if (settingsRef.current.voiceAutoSend) void submit(t);
    else setInput((prev) => (prev.trim() ? `${prev.trim()} ${t}` : t));
  }

  /** Toggle hands-free voice mode. On: confirm mic, load STT + TTS (the loop speaks replies),
   *  then start an auto-endpointing listening session. Off: end the session and stop speaking. */
  async function toggleVoiceMode() {
    if (voiceStateRef.current !== 'off') {
      stopVoiceMode();
      return;
    }
    setVoiceError(null);
    setVoiceLoading('voice');
    try {
      if (!(await ensureMicPermission())) {
        setVoiceError('Microphone access is needed for voice mode. Allow it and try again.');
        return;
      }
      const port = await getWebllmPort();
      if (!(await ensureSttReady(port))) return;
      // The loop reads replies aloud, so make sure the TTS voice is loaded too.
      if (!(await ensureTtsReady(port))) return;
      setVoice('listening');
      voiceSessionRef.current = sttStartListening(port, {
        mode: 'auto',
        pauseMs: settingsRef.current.voicePauseMs,
        bargeIn: true,
        onState: (s) => onVoiceStateUpdate(s),
        onTranscript: (text) => void onVoiceTranscript(text),
        onError: (e) => setVoiceError(e.message),
        onStopped: () => {
          voiceSessionRef.current = null;
          // Session ended offscreen (mic failure, worker teardown) — don't leave the loop UI up.
          if (voiceStateRef.current !== 'off') setVoice('off');
        },
      });
    } finally {
      setVoiceLoading(null);
    }
  }

  function stopVoiceMode() {
    setVoice('off');
    voiceSessionRef.current?.stop();
    voiceSessionRef.current = null;
    bargeInRef.current = false;
    pendingVoiceTranscriptRef.current = null;
    stopSpeaking();
  }

  /** A status update from the STT worker. Full-duplex barge-in: if the user starts talking while
   *  the assistant is thinking or speaking, stop the reply and let the transcript flow take over. */
  function onVoiceStateUpdate(s: SttState) {
    const cur = voiceStateRef.current;
    if (s === 'speech' && (cur === 'thinking' || cur === 'speaking')) {
      bargeInRef.current = true;
      dispatchVoice({ type: 'bargeIn' }); // → speech
      stopSpeaking(); // halt TTS playback immediately
      abortRef.current?.abort(); // and cut generation if it's still running
      return;
    }
    dispatchVoice({ type: 'sttState', state: s });
  }

  /** A hands-free utterance was transcribed → send it to the LLM. submit() drives the reply back
   *  to 'speaking' → 'listening'. */
  async function onVoiceTranscript(text: string) {
    if (voiceStateRef.current === 'off') return;
    const t = text.trim();
    if (!t) {
      bargeInRef.current = false;
      dispatchVoice({ type: 'resume' });
      return;
    }
    // Barge-in echo guard: a transcript that closely matches what TTS just spoke is the
    // assistant's own voice bleeding into the mic — discard it and keep listening.
    const now = Date.now();
    const recentSpoken = spokenRef.current.filter((x) => now - x.at < 10_000).map((x) => x.plain);
    if (recentSpoken.length && isLikelyEcho(t, recentSpoken)) {
      bargeInRef.current = false;
      dispatchVoice({ type: 'resume' });
      return;
    }
    // A prior generation may still be unwinding after a barge-in abort — submit() would reject
    // while abortRef is set, so stash it and let submit's finally re-submit.
    if (abortRef.current) {
      pendingVoiceTranscriptRef.current = t;
      return;
    }
    bargeInRef.current = false;
    dispatchVoice({ type: 'transcript' }); // → thinking
    await submit(t, { fromVoice: true });
  }

  /** After a hands-free reply generates: speak it (forced), then resume listening. Any empty /
   *  errored / aborted path resumes too, so the loop is never stranded in 'thinking'. */
  function handleVoiceReply(idx: number, text: string | null) {
    if (voiceStateRef.current === 'off') return;
    if (text && text.trim()) {
      dispatchVoice({ type: 'speak' });
      void speakMessage(idx, text).finally(resumeVoiceListening);
    } else {
      resumeVoiceListening();
    }
  }

  function resumeVoiceListening() {
    if (voiceStateRef.current === 'off') return;
    setVoiceReveal(null); // reply is done — show its full text
    dispatchVoice({ type: 'resume' }); // → listening
  }

  /** Reset all voice/STT state (offscreen doc gone or model released). */
  function resetVoiceState() {
    voiceSessionRef.current = null;
    pttActiveRef.current = false;
    setPttActive(false);
    setVoice('off');
    sttReadyRef.current = false;
    setSttReady(false);
    bargeInRef.current = false;
    pendingVoiceTranscriptRef.current = null;
  }

  /** Free the speech model from memory (Settings). Next voice use reloads it. */
  function releaseVoiceInput() {
    stopVoiceMode();
    if (portRef.current) sttRelease(portRef.current);
    sttReadyRef.current = false;
    setSttReady(false);
  }

  /** Show a short-lived note under the attachment chips (wrong file type, size, model). */
  function flashAttachNote(text: string) {
    setAttachNote(text);
    if (attachNoteTimerRef.current) clearTimeout(attachNoteTimerRef.current);
    attachNoteTimerRef.current = setTimeout(() => setAttachNote(null), 6000);
  }

  /** Stage picked files: images are downscaled to data URLs (vision model only, one per
   *  message — an image costs ~2k tokens of the vision build's 4k context); anything else
   *  must read as text. Rejections surface as a transient note, never an error. */
  async function addAttachments(list: FileList | null) {
    if (!list?.length) return;
    let next = [...attachmentsRef.current];
    const visionOk = webllmModel(settingsRef.current.webllmModel).vision === true;
    for (const file of Array.from(list)) {
      if (file.type.startsWith('image/')) {
        if (!visionOk) {
          flashAttachNote('Images need the vision model — pick Phi 3.5 Vision in Settings.');
          continue;
        }
        try {
          const dataUrl = await imageFileToDataUrl(file);
          // One image per message: a new pick replaces the previous one.
          next = [...next.filter((a) => a.kind !== 'image'), { kind: 'image', name: file.name, dataUrl }];
        } catch {
          flashAttachNote(`Couldn't read ${file.name} as an image.`);
        }
      } else {
        if (next.filter((a) => a.kind === 'text').length >= MAX_ATTACHED_FILES) {
          flashAttachNote(`Up to ${MAX_ATTACHED_FILES} files per message.`);
          break;
        }
        if (file.size > MAX_TEXT_FILE_BYTES) {
          flashAttachNote(`${file.name} is too large — text files up to 2 MB.`);
          continue;
        }
        try {
          const text = await file.text();
          if (text.includes('\0')) throw new Error('binary');
          next = [...next, { kind: 'text', name: file.name, text: text.slice(0, MAX_STORED_FILE_CHARS) }];
        } catch {
          flashAttachNote(`${file.name} isn't readable text (PDFs and binaries aren't supported).`);
        }
      }
    }
    setAttachments(next);
  }

  async function submit(
    text: string,
    opts?: {
      selectionOverride?: string;
      task?: TaskSpec;
      schema?: ExtractionSchema;
      fromVoice?: boolean;
    },
  ) {
    const trimmed = text.trim();
    // abortRef (a ref, never stale) backs up the `streaming` state check: voice transcripts
    // arrive via long-lived port listeners whose captured `streaming` is frozen at registration
    // time, so a transcript landing mid-generation would otherwise slip past the guard.
    if (!trimmed || streaming || abortRef.current) {
      // A voice utterance that can't be sent right now must still release the loop.
      if (opts?.fromVoice) resumeVoiceListening();
      return;
    }

    // A new question interrupts any speech from the previous reply.
    stopSpeaking();

    const s = settingsRef.current;
    const spec = opts?.task ?? TASKS.ask;
    // Attachments ride on free-form questions only — the quick actions are about the page.
    const attached = spec.kind === 'ask' ? attachmentsRef.current : [];
    const images = attached.filter((a) => a.kind === 'image' && a.dataUrl).map((a) => a.dataUrl!);
    const files = capAttachedFiles(
      attached.filter((a) => a.kind === 'text' && a.text != null).map((a) => ({ name: a.name, text: a.text! })),
      webllmModel(s.webllmModel).safePromptTokens,
    );
    const conversation: ChatMessage[] = [
      ...messagesRef.current,
      {
        role: 'user',
        content: trimmed,
        ...(images.length ? { images } : {}),
        ...(files.length ? { files } : {}),
      },
    ];
    // Index of the assistant reply we're about to stream (for auto-read on completion).
    const assistantIdx = conversation.length;
    // In hands-free voice mode we speak this reply ourselves (forced) and resume listening — so
    // capture the final text here instead of letting autoRead speak it on the ttsAutoRead setting.
    let voiceReplyText: string | null = null;
    // When set, this reply was spoken by a streamed TTS session (sentence-by-sentence as it
    // generated); the finally block awaits its playback instead of speaking the whole text again.
    let streamSession: TtsStreamSession | null = null;
    shouldAutoScrollRef.current = true;
    setMessages([...conversation, { role: 'assistant', content: '' }]);
    setInput('');
    if (attached.length) setAttachments([]);

    const controller = new AbortController();
    abortRef.current = controller;
    const epoch = beginEngineOp();
    setStreaming(true);
    let streamIterator: AsyncIterator<string> | null = null;
    let modelReady = false;

    try {
      if (!webgpuAvailable()) {
        throw new Error('This browser has no WebGPU. Use a recent Chromium browser (Chrome, Edge, or Brave).');
      }
      const runtimeCtx = contextForSettings(s);
      const model = webllmModel(s.webllmModel);
      // A voice request is read back by TTS, so switch the model to conversational, spoken-style
      // output (no markdown/LaTeX/lists) by appending the voice directive to the system prompt.
      const baseSys = spec.systemPrompt || undefined;
      const sysOverride = opts?.fromVoice
        ? `${baseSys ?? s.systemPrompt}\n\n${VOICE_REPLY_DIRECTIVE}`
        : baseSys;
      let page: PageContent | null = null;
      let built: BuiltPrompt;
      // An image turn skips page capture: the question is about the image, and on the vision
      // build (4k context) the image embedding (~2k tokens) plus a page cannot fit anyway.
      if (s.pageContext && !images.length) {
        setPageNote('Reading page…');
        try {
          page = await withTimeout(
            capturePage(s.viewportBoost),
            PAGE_CAPTURE_TIMEOUT_MS,
            'This page took too long to read. Heavy, app-like pages can do this — select the relevant text and try again.',
          );
        } catch (e) {
          // An unreadable page (chrome://, the Web Store, PDFs…) fails a free-form question
          // softly: the turn degrades to page-context-off and the question is still answered.
          // Page-dependent tasks (summarize/explain/extract) keep the error — without the
          // page they could only invent. The capture timeout stays an error too: that page
          // IS readable, and the "select the relevant text" advice is actionable.
          if (spec.kind !== 'ask' || !(e instanceof PageUnreadableError)) throw e;
          setPageNote("This page can't be read — answering without page context.");
        }
        if (controller.signal.aborted) {
          setLastAssistant('_(stopped)_');
          return;
        }
      }
      // Page context off — or an unreadable page a question degraded past: nothing from the
      // tab is sent; the prompt is just the system scaffold plus the conversation.
      if (!page) {
        built = buildMessages(s, null, conversation, runtimeCtx, {
          systemPromptOverride: sysOverride,
        });
      } else {
        if (opts?.selectionOverride && !page.selection.trim()) {
          page.selection = opts.selectionOverride;
        }
        setPageNote('Preparing page context…');
        const budget = pageBudgetTokens(s, page, conversation, runtimeCtx, sysOverride);
        // The body a single prompt may carry: the context-window budget, hard-capped per model.
        // Big models get SMALLER prompts — one oversized prefill on an integrated GPU can starve
        // the OS compositor and take down the whole session (see docs/large-page-handling.md).
        const bodyBudget = Math.min(budget, model.safePromptTokens);
        // Retrieve from (rather than truncate) the page whenever the task allows it and the page
        // overflows what this model can safely take in one prompt. A selection no longer disables
        // retrieval — it sharpens the query instead (the selection itself rides in the header).
        const useRag = spec.allowRag && estimateTokens(page.textContent) > bodyBudget;

        if (useRag) {
          const port = await getWebllmPort();
          const chunks = chunkPage(page);
          const hash = hashText(page.textContent);
          setPageNote('Indexing page for retrieval…');
          await indexPage(port, page.url, hash, chunks, (p) => {
            if (engineOpCurrent(epoch)) setLoadProgress(p);
          });
          if (engineOpCurrent(epoch)) setLoadProgress(null);
          if (controller.signal.aborted) {
            setLastAssistant('_(stopped)_');
            return;
          }
          setPageNote('Finding the most relevant sections…');
          const topK = Math.max(3, Math.min(8, Math.floor(bodyBudget / 350)));
          // Extraction retrieves against its instruction; a question retrieves against itself,
          // sharpened by any selected text.
          const selection = page.selection.trim();
          const query =
            spec.kind === 'extract' && opts?.schema
              ? opts.schema.instruction
              : selection
                ? `${trimmed}\n\nSelected text: ${selection.slice(0, 600)}`
                : trimmed;
          // Over-fetch when boosting so on-screen chunks can rise into the final top-k.
          const fetchK = s.viewportBoost ? Math.min(topK * 2, 16) : topK;
          let results = await retrieveChunks(port, page.url, hash, query, fetchK);
          if (controller.signal.aborted) {
            setLastAssistant('_(stopped)_');
            return;
          }
          if (s.viewportBoost && page.viewportText) {
            const vp = page.viewportText.toLowerCase();
            results = results
              .map((r) => {
                const probe = r.text.replace(/\s+/g, ' ').trim().slice(0, 40).toLowerCase();
                return probe.length > 8 && vp.includes(probe) ? { ...r, score: r.score * 1.15 } : r;
              })
              .sort((a, b) => b.score - a.score);
          }
          results = results.slice(0, topK);
          if (s.compressContext && results.length) {
            setPageNote('Compressing context…');
            try {
              const compressed = await compressTexts(
                port,
                results.map((r) => r.text),
                0.5,
              );
              results = results.map((r, i) => ({
                ...r,
                text: compressed[i] ?? r.text,
              }));
            } catch {
              /* compression unavailable — proceed with the uncompressed chunks */
            }
            if (controller.signal.aborted) {
              setLastAssistant('_(stopped)_');
              return;
            }
          }
          built = buildMessages(s, page, conversation, runtimeCtx, {
            retrieved: results,
            systemPromptOverride: sysOverride,
            maxBodyTokens: model.safePromptTokens,
          });
          setPageNote(null);
        } else {
          built = buildMessages(s, page, conversation, runtimeCtx, {
            systemPromptOverride: sysOverride,
            maxBodyTokens: model.safePromptTokens,
          });
          // Coverage is never narrated — the note strip only speaks while work is in flight.
          setPageNote(spec.supportsMapReduce && built.truncated ? 'Preparing to summarize a long page…' : null);
        }
      }

      // A large page under the summarize task is handled by map-reduce, not truncation.
      const useMapReduce = page != null && spec.supportsMapReduce && built.truncated;

      // The image embeds + prefills as one indivisible GPU chunk (~1.4-1.9k tokens), so the
      // wait before the first token is noticeably longer than a text turn — say why.
      if (images.length) setPageNote('Analyzing image…');

      const port = await getWebllmPort();
      if (engineOpCurrent(epoch)) setModelStatus((cur) => (cur === 'ready' ? cur : 'loading'));
      // The abort signal makes Stop cancel the load itself — including a not-yet-downloaded
      // model's download (it resumes later) — instead of silently waiting it out.
      await initModel(
        port,
        s.webllmModel,
        runtimeCtx,
        (p) => {
          if (engineOpCurrent(epoch)) setLoadProgress(p);
        },
        controller.signal,
      );
      modelReady = true;
      if (engineOpCurrent(epoch)) {
        setModelStatus('ready');
        setLoadProgress(null);
      }
      if (controller.signal.aborted) {
        setLastAssistant('_(stopped)_');
        return;
      }

      const genOptions: GenerateOptions = {
        temperature: resolveTemperature(spec, s),
        maxTokens: spec.maxTokens,
        frequencyPenalty: s.repetitionPenalty || undefined,
      };
      // Qwen3 models reason in hidden <think> blocks by default — the user just sees the
      // spinner while tokens they'll never read stream at decode speed. Hard-disable it for
      // every task via the chat-template flag (the soft "/no_think" text hint was unreliable).
      if (/Qwen3/i.test(s.webllmModel)) genOptions.enableThinking = false;
      if (spec.outputMode === 'json' && opts?.schema) {
        genOptions.responseFormat = {
          type: 'json_object',
          schema: JSON.stringify(opts.schema.schema),
        };
      }

      if (useMapReduce && page) {
        // Whole-page summary of an over-budget page: summarize sections, then merge.
        const chunks = chunkPage(page);
        const qwen = /Qwen3/i.test(s.webllmModel);
        const runOne = async (
          messages: ChatMessage[],
          o: { temperature: number; maxTokens: number; stream?: boolean },
        ): Promise<string> => {
          const rid = ++reqRef.current;
          const ro: GenerateOptions = {
            temperature: o.temperature,
            maxTokens: o.maxTokens,
            enableThinking: qwen ? false : undefined,
            frequencyPenalty: s.repetitionPenalty || undefined,
          };
          if (o.stream) {
            const it = streamGenerate(port, rid, messages, ro, controller.signal)[Symbol.asyncIterator]();
            let acc = '';
            let got = false;
            while (true) {
              const next = await withTimeout(
                it.next(),
                got ? NEXT_TOKEN_TIMEOUT_MS : FIRST_TOKEN_TIMEOUT_MS,
                'The local model took too long, so I stopped it. Try a smaller page or model.',
                () => controller.abort(),
              );
              if (next.done) break;
              got = true;
              acc += next.value;
              setLastAssistant(acc);
            }
            return acc;
          }
          return withTimeout(
            generateOnce(port, rid, messages, { ...ro, stream: false }, controller.signal),
            FIRST_TOKEN_TIMEOUT_MS,
            'The local model took too long on a section, so I stopped it.',
            () => controller.abort(),
          );
        };
        const summary = await summarizeChunks(
          chunks,
          {
            ctxTokens: runtimeCtx,
            title: page.title || 'this page',
            maxPromptTokens: model.safePromptTokens,
          },
          {
            runOne,
            onProgress: (p) => {
              if (p.phase === 'map') setPageNote(`Summarizing section ${p.index + 1} of ${p.total}…`);
              else if (p.phase === 'reduce') setPageNote('Merging section summaries…');
            },
            signal: controller.signal,
          },
        );
        setLastAssistant(summary || (controller.signal.aborted ? '_(stopped)_' : '_(no response)_'));
        if (summary && !controller.signal.aborted) {
          if (opts?.fromVoice) voiceReplyText = summary;
          else autoRead(assistantIdx, summary);
        }
      } else if (spec.outputMode === 'json') {
        // Structured extraction: one non-streaming call, parse, store as a structured message.
        let raw: string;
        try {
          raw = await generateOnce(port, ++reqRef.current, built.messages, genOptions, controller.signal);
        } catch (e) {
          if ((e as Error)?.message === 'extract:truncated') {
            throw new Error("The extraction hit this model's output limit. Try a smaller schema or a larger model.");
          }
          throw e;
        }
        let data: unknown = null;
        try {
          data = JSON.parse(raw);
        } catch {
          /* XGrammar should guarantee valid JSON; render a raw fallback if it ever doesn't */
        }
        const schemaId = opts?.schema?.id ?? 'extract';
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { role: 'assistant', content: raw || '_(no data)_', structured: { schemaId, data } };
          }
          return copy;
        });
      } else {
        streamIterator = streamGenerate(port, ++reqRef.current, built.messages, genOptions, controller.signal)[Symbol.asyncIterator]();
        let acc = '';
        let receivedToken = false;
        // Speak the reply sentence-by-sentence AS it generates when the request is from voice or
        // auto-read is on. The session opens lazily on the first complete sentence; sentences are
        // segmented from the (think-stripped) accumulator so speech starts within a sentence or
        // two instead of after the whole reply. For auto-read we only stream when the voice is
        // already loaded — otherwise a first-use ~132 MB download would freeze the streaming text
        // mid-reply; that case falls back to speaking the whole reply after it finishes.
        const isVoiceMode = opts?.fromVoice === true;
        const wantTts = isVoiceMode || (settingsRef.current.ttsAutoRead && ttsReadyRef.current);
        const sentences = new SentenceStream();
        // In voice mode, hide the reply text from the start so it reveals in step with speech
        // (rather than flashing the first sentence before its audio is ready).
        if (isVoiceMode) setVoiceReveal({ idx: assistantIdx, end: 0 });

        const openStreamSession = async (): Promise<TtsStreamSession | null> => {
          if (!(await ensureTtsReady(port))) return null;
          const session = ttsSpeakStream(port);
          revealMapRef.current = new Map();
          spokenRef.current = [];
          session.onChunkStarted((seq) => {
            const end = revealMapRef.current.get(seq);
            if (end != null && isVoiceMode) setVoiceReveal({ idx: assistantIdx, end });
          });
          session.onError((err) => setTtsError(err.message));
          ttsStreamRef.current = session;
          setSpeakingIdx(assistantIdx);
          speakingIdxRef.current = assistantIdx;
          if (isVoiceMode) {
            dispatchVoice({ type: 'speak' });
            setVoiceReveal({ idx: assistantIdx, end: 0 }); // hide text until the first spoken sentence
          }
          return session;
        };

        const speakSentence = (session: TtsStreamSession, text: string, endOffset: number) => {
          const plain = toPlainText(text);
          if (!plain) return;
          const seq = session.append(plain);
          revealMapRef.current.set(seq, endOffset);
          spokenRef.current.push({ plain, at: Date.now() });
        };

        while (true) {
          const next = await withTimeout(
            streamIterator.next(),
            receivedToken ? NEXT_TOKEN_TIMEOUT_MS : FIRST_TOKEN_TIMEOUT_MS,
            'The local model took too long on this page, so I stopped it. Try selecting the most relevant text or asking a narrower question.',
            () => controller.abort(),
          );
          if (next.done) break;
          const delta = next.value;
          if (!receivedToken) setPageNote(null); // prefill (incl. image analysis) is done
          receivedToken = true;
          acc += delta;
          setLastAssistant(acc);
          if (wantTts && !controller.signal.aborted) {
            for (const piece of sentences.push(stripThink(acc))) {
              if (!streamSession) {
                streamSession = await openStreamSession();
                if (!streamSession) break;
              }
              speakSentence(streamSession, piece.text, piece.endOffset);
            }
          }
        }

        if (streamSession && !controller.signal.aborted) {
          const tail = sentences.flush();
          if (tail) speakSentence(streamSession, tail.text, tail.endOffset);
          streamSession.finish();
        } else if (streamSession) {
          streamSession.cancel();
          streamSession = null;
          ttsStreamRef.current = null;
          setVoiceReveal(null);
        }

        if (!acc) setLastAssistant(controller.signal.aborted ? '_(stopped)_' : '_(no response)_');
        else if (controller.signal.aborted) {
          // Barge-in kept the partial reply in the thread (and thus in context) so the follow-up
          // turn can build on it; a plain Stop just leaves the partial as-is.
          if (bargeInRef.current) setLastAssistant(acc + '\n\n_(interrupted)_');
        } else if (!streamSession) {
          // Nothing was streamed to TTS (e.g. a short single-sentence voice reply, or auto-read
          // off and not a voice turn) — fall back to the whole-reply paths and show the full text
          // (there's no sentence-level reveal to sync to here).
          if (isVoiceMode) setVoiceReveal(null);
          if (opts?.fromVoice) voiceReplyText = acc;
          else autoRead(assistantIdx, acc);
        }
      }
    } catch (e) {
      if (controller.signal.aborted) {
        try {
          await streamIterator?.return?.();
        } catch {
          /* stream already closed */
        }
      }
      // A load canceled by Stop or displaced by a model switch reads as "stopped", not a crash.
      const userStopped = (e instanceof DOMException && e.name === 'AbortError') || isLoadInterruption(e);
      if (userStopped) setLastAssistant('_(stopped)_');
      else setLastAssistantError(e instanceof Error ? e.message : String(e));
      // A streamed speech session for this reply is now orphaned — stop it.
      if (streamSession) {
        streamSession.cancel();
        streamSession = null;
        ttsStreamRef.current = null;
        setVoiceReveal(null);
      }
      // If the model never finished loading, its true state (cached? downloaded?) is
      // unknown here — re-derive the status instead of leaving a stale "Loading…".
      if (!modelReady) schedulePreload(settingsRef.current);
    } finally {
      setStreaming(false);
      if (engineOpCurrent(epoch)) setLoadProgress(null);
      setPageNote(null);
      abortRef.current = null;
      endEngineOp(epoch);
      // A barge-in captured a new utterance while this reply was in flight — send it as the next
      // turn instead of resuming the idle loop.
      const pending = pendingVoiceTranscriptRef.current;
      if (pending != null && voiceStateRef.current !== 'off') {
        pendingVoiceTranscriptRef.current = null;
        bargeInRef.current = false;
        dispatchVoice({ type: 'transcript' }); // → thinking
        void submit(pending, { fromVoice: true });
      } else if (streamSession) {
        // The reply was streamed to TTS sentence-by-sentence and its audio is still playing; wait
        // for it to finish before clearing the speaking marker / resuming the listening loop.
        const session = streamSession;
        const fromVoice = opts?.fromVoice === true;
        void session.ended.then(() => {
          if (ttsStreamRef.current === session) ttsStreamRef.current = null;
          if (speakingIdxRef.current === assistantIdx) {
            setSpeakingIdx(null);
            speakingIdxRef.current = null;
          }
          setVoiceReveal(null);
          // A barge-in is mid-flight (its transcript hasn't arrived yet) — don't resume; the
          // incoming transcript owns the next turn.
          if (fromVoice && !bargeInRef.current) resumeVoiceListening();
        });
      } else if (opts?.fromVoice && !bargeInRef.current) {
        // Hands-free loop: speak the reply (or just resume) — never leave it stuck in 'thinking'.
        handleVoiceReply(assistantIdx, voiceReplyText);
      }
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function explainSelection() {
    try {
      const page = await capturePage();
      if (!page.selection.trim()) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Select some text on the page first, then click “Explain selection”.',
          },
        ]);
        return;
      }
    } catch {
      /* submit() will surface the capture error */
    }
    void submit(TASKS.explain.label, { task: TASKS.explain });
  }

  async function onSaveSettings(patch: Partial<Settings>) {
    const previous = settingsRef.current;
    // Switching model clamps the context to that model's maximum.
    let next = patch;
    if (patch.webllmModel) {
      const max = Math.min(webllmModel(patch.webllmModel).maxCtx, MAX_CONTEXT_TOKENS);
      const cur = patch.webllmCtx ?? previous.webllmCtx;
      if (cur > max) next = { ...patch, webllmCtx: max };
    }
    const changedKeys = (Object.keys(next) as (keyof Settings)[]).filter((key) => previous[key] !== next[key]);
    if (!changedKeys.length) return;
    const merged = await saveSettings(next);
    setSettings(merged);
    settingsRef.current = merged;
    if (changedKeys.includes('webllmModel')) {
      // Reset transient download UI at the switch so the card/badge never briefly shows the
      // new model with the old model's progress. The debounced preload re-resolves the state.
      setLoadProgress(null);
      setModelStatus('loading');
    }
    if (changedKeys.includes('webllmModel') || changedKeys.includes('webllmCtx')) schedulePreload(merged);
  }

  // The selected model's weights aren't on disk yet: the in-chat card owns the flow, so
  // lock the composer and quick actions until the download finishes (status leaves the
  // needs-download / downloading states).
  const composerLocked = onboardingVisible;
  const disabled = streaming || composerLocked;

  return (
    <div className="flex h-full flex-col bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
        <Logo size={18} />
        <span className="text-sm font-semibold tracking-tight">Enclave</span>
        <select
          className="ml-auto max-w-[12rem] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
          value={settings.webllmModel}
          onChange={(e) => onSaveSettings({ webllmModel: e.target.value })}
          title="Model"
        >
          {WEBLLM_MODELS.map((m) => {
            const activelyDownloading = m.id === settings.webllmModel && modelStatus === 'downloading';
            const suffix = activelyDownloading
              ? ' · downloading'
              : pausedModels.includes(m.id)
                ? ' · download paused'
                : '';
            return (
              <option key={m.id} value={m.id}>
                {m.label} · ~{m.approxGb} GB{suffix}
              </option>
            );
          })}
        </select>
        <button
          className="flex items-center justify-center rounded p-1.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
          onClick={() => setShowSettings((v) => !v)}
          title="Settings"
          aria-label="Settings"
          aria-pressed={showSettings}
        >
          <SettingsIcon size={17} />
        </button>
      </header>

      {/* Model load progress (the model card renders its own while it is visible) */}
      {loadProgress && !onboardingVisible && (
        <div className="border-b border-zinc-200 bg-zinc-100 px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-zinc-600 dark:text-zinc-300">Loading your model… {Math.round(loadProgress.progress * 100)}%</p>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-zinc-300 dark:bg-zinc-700">
            <div
              className="h-full bg-zinc-600 transition-all dark:bg-zinc-300"
              style={{ width: `${Math.round(loadProgress.progress * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onSave={onSaveSettings}
          onRelease={releaseModel}
          busy={streaming}
          ttsLoaded={ttsReady}
          onReleaseVoice={releaseVoiceModel}
          sttLoaded={sttReady}
          onReleaseVoiceInput={releaseVoiceInput}
        />
      )}

      {/* Transient work-in-progress note (cleared whenever nothing is running) */}
      {pageNote && (
        <div className="border-b border-zinc-200 bg-zinc-100 px-3 py-1.5 text-[11px] text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
          {pageNote}
        </div>
      )}

      {/* Text-to-speech error (voice download / synthesis) */}
      {ttsError && (
        <div className="flex items-center justify-between gap-2 border-b border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
          <span>Voice: {ttsError}</span>
          <button className="shrink-0 underline" onClick={() => setTtsError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* Voice input error (mic permission / speech model) */}
      {voiceError && (
        <div className="flex items-center justify-between gap-2 border-b border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
          <span>Voice input: {voiceError}</span>
          <button className="shrink-0 underline" onClick={() => setVoiceError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* Hands-free voice-mode status strip */}
      {voiceState !== 'off' && (
        <div className="flex items-center gap-2 border-b border-indigo-200 bg-indigo-50 px-3 py-1.5 text-[11px] text-indigo-700 dark:border-indigo-900 dark:bg-indigo-950/40 dark:text-indigo-300">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-indigo-500" />
          </span>
          <span>{VOICE_STATE_LABEL[voiceState]}</span>
          <button
            className="ml-auto shrink-0 rounded border border-indigo-300 px-2 py-0.5 hover:bg-indigo-100 dark:border-indigo-700 dark:hover:bg-indigo-900/40"
            onClick={toggleVoiceMode}
          >
            End
          </button>
        </div>
      )}

      {/* Chat thread */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3" onScroll={onChatScroll}>
        {messages.length === 0 &&
          !onboardingVisible &&
          (modelStatus === 'loading' ? (
            <div className="mt-10 flex justify-center">
              <HexSpinner label="Loading your model…" />
            </div>
          ) : (
            <div className="mt-6 text-center text-xs text-zinc-400 dark:text-zinc-500">
              Ask anything about the current page. Everything runs locally on your device.
            </div>
          ))}
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          const isStreamingThis = streaming && i === messages.length - 1;
          const isPlaceholder = !m.content && isStreamingThis;
          return (
            <div
              key={i}
              className={
                isUser
                  ? 'ml-auto max-w-[90%] rounded-lg bg-zinc-200 px-3 py-2 text-sm whitespace-pre-wrap text-zinc-900 dark:bg-zinc-700 dark:text-zinc-50'
                  : 'mr-auto max-w-[95%] rounded-lg bg-white px-3 py-2 text-zinc-800 shadow-sm dark:bg-zinc-800 dark:text-zinc-100'
              }
            >
              {isUser ? (
                <>
                  {m.images?.map((src, k) => (
                    <img key={k} src={src} alt="Attached image" className="mb-1.5 max-h-44 max-w-full rounded" />
                  ))}
                  {m.files?.length ? (
                    <div className="mb-1 flex flex-wrap gap-1">
                      {m.files.map((f, k) => (
                        <span
                          key={`${f.name}-${k}`}
                          className="inline-flex items-center gap-1 rounded-full bg-zinc-300/60 px-2 py-0.5 text-[11px] dark:bg-zinc-600/60"
                        >
                          <FileTextIcon size={11} />
                          <span className="max-w-[10rem] truncate">{f.name}</span>
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {m.content}
                </>
              ) : isPlaceholder ? (
                <ThinkingIndicator />
              ) : m.error ? (
                <div className="flex items-start gap-1.5 text-amber-700 dark:text-amber-400">
                  <AlertTriangleIcon size={15} className="mt-0.5 shrink-0" />
                  <span className="text-sm">{m.content}</span>
                </div>
              ) : m.structured ? (
                <StructuredResult data={m.structured.data} raw={m.content} />
              ) : (
                (() => {
                  const full = stripThink(m.content);
                  // In voice mode, reveal only up to the sentence currently being spoken so the
                  // shown text tracks the audio. Non-voice (incl. auto-read) shows the full reply.
                  const reveal = voiceReveal && voiceReveal.idx === i ? voiceReveal.end : null;
                  const display = reveal != null && reveal < full.length ? full.slice(0, reveal) : full;
                  if (!display) return <ThinkingIndicator />;
                  return (
                    <>
                      <Markdown content={display} />
                      {/* Read aloud (pocket-tts). Hidden while this reply is still streaming. */}
                      {!isStreamingThis && (
                        <SpeakButton
                          active={speakingIdx === i}
                          downloading={ttsDownloading && speakingIdx === i}
                          downloadPct={ttsDownloadPct}
                          disabled={ttsDownloading && speakingIdx !== i}
                          onClick={() => void speakMessage(i, m.content)}
                        />
                      )}
                    </>
                  );
                })()
              )}
            </div>
          );
        })}
        {onboardingVisible && (
          <OnboardingCard
            settings={settings}
            suggestedId={suggestedModelId}
            pausedModels={pausedModels}
            busy={modelStatus === 'downloading'}
            progress={loadProgress}
            error={onboardError}
            voiceEnabled={onboardVoice}
            stt={{ downloading: sttDownloading, pct: sttDownloadPct, ready: sttReady, error: sttInitError }}
            tts={{ downloading: ttsDownloading, pct: ttsDownloadPct, ready: ttsReady, error: ttsInitError }}
            micNeeded={onboardVoice && sttReady && ttsReady && !micGranted}
            onSelect={(id) => void onSaveSettings({ webllmModel: id })}
            onDownload={() => void downloadModel()}
            onToggleVoice={toggleOnboardVoice}
            onRetryStt={() => void getWebllmPort().then(downloadSttOnboard)}
            onRetryTts={() => void getWebllmPort().then(downloadTtsOnboard)}
            onAllowMic={() => void ensureMicPermission().then((ok) => ok && setMicGranted(true))}
          />
        )}
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap items-center gap-2 border-t border-zinc-200 bg-white px-3 pt-2 dark:border-zinc-800 dark:bg-zinc-900">
        <button
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
          disabled={disabled || !settings.pageContext}
          title={settings.pageContext ? undefined : 'Turn page context on to use this'}
          onClick={() => submit(TASKS.summarize.label, { task: TASKS.summarize })}
        >
          Summarize
        </button>
        <button
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
          disabled={disabled || !settings.pageContext}
          title={settings.pageContext ? undefined : 'Turn page context on to use this'}
          onClick={explainSelection}
        >
          Explain selection
        </button>
        <select
          className="rounded-full border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800"
          disabled={disabled || !settings.pageContext}
          value=""
          onChange={(e) => {
            const schema = EXTRACTION_SCHEMAS.find((x) => x.id === e.target.value);
            if (schema)
              submit(extractTask(schema).label, {
                task: extractTask(schema),
                schema,
              });
          }}
          title="Extract structured data as JSON"
        >
          <option value="">Extract…</option>
          {EXTRACTION_SCHEMAS.map((x) => (
            <option key={x.id} value={x.id}>
              {x.label}
            </option>
          ))}
        </select>
        {/* Page-context gate: when off, NOTHING from the tab (title, URL, selection, text)
            is read or sent to the local model. */}
        <button
          type="button"
          className={
            'ml-auto inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs disabled:opacity-40 ' +
            (settings.pageContext
              ? 'border-indigo-300 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:border-indigo-500/50 dark:bg-indigo-500/15 dark:text-indigo-300 dark:hover:bg-indigo-500/25'
              : 'border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100 dark:border-amber-500/50 dark:bg-amber-500/15 dark:text-amber-300 dark:hover:bg-amber-500/25')
          }
          disabled={composerLocked}
          aria-pressed={settings.pageContext}
          title={
            settings.pageContext
              ? 'Page context is ON — this tab’s content is included in what the local model reads. Click to keep the page private.'
              : 'Page context is OFF — nothing from this tab is sent to the model. Click to include the page again.'
          }
          onClick={() => void onSaveSettings({ pageContext: !settings.pageContext })}
        >
          {settings.pageContext ? <FileTextIcon size={13} /> : <LockIcon size={13} />}
          {settings.pageContext ? 'Page: on' : 'Page: off'}
        </button>
      </div>

      {/* Attachments staged for the next message */}
      {(attachments.length > 0 || attachNote) && (
        <div className="bg-white px-3 pt-2 dark:bg-zinc-900">
          {attachments.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              {attachments.map((a, i) =>
                a.kind === 'image' ? (
                  <span key={`${a.name}-${i}`} className="relative inline-block">
                    <img
                      src={a.dataUrl}
                      alt={a.name}
                      className="h-12 w-12 rounded border border-zinc-300 object-cover dark:border-zinc-700"
                    />
                    <button
                      type="button"
                      aria-label={`Remove ${a.name}`}
                      title={`Remove ${a.name}`}
                      className="absolute -top-1.5 -right-1.5 rounded-full bg-zinc-700 p-0.5 text-white hover:bg-zinc-600"
                      onClick={() => setAttachments(attachments.filter((_, j) => j !== i))}
                    >
                      <XIcon size={10} />
                    </button>
                  </span>
                ) : (
                  <span
                    key={`${a.name}-${i}`}
                    className="inline-flex items-center gap-1 rounded-full border border-zinc-300 bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                  >
                    <FileTextIcon size={11} />
                    <span className="max-w-[10rem] truncate">{a.name}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${a.name}`}
                      className="hover:text-zinc-900 dark:hover:text-zinc-100"
                      onClick={() => setAttachments(attachments.filter((_, j) => j !== i))}
                    >
                      <XIcon size={11} />
                    </button>
                  </span>
                ),
              )}
            </div>
          )}
          {attachNote && <p className="mt-1 text-[11px] text-amber-600 dark:text-amber-400">{attachNote}</p>}
        </div>
      )}

      {/* Composer */}
      <div className="flex items-end gap-2 bg-white px-3 pt-2 pb-3 dark:bg-zinc-900">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          // Images are only offered when the selected model can see them; text files always.
          accept={webllmModel(settings.webllmModel).vision ? `image/*,${TEXT_FILE_ACCEPT}` : TEXT_FILE_ACCEPT}
          onChange={(e) => {
            void addAttachments(e.target.files);
            e.target.value = ''; // allow re-picking the same file
          }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={composerLocked}
          title={
            webllmModel(settings.webllmModel).vision
              ? 'Attach images or text files to your question'
              : 'Attach text files to your question (images need the vision model — pick Phi 3.5 Vision in Settings)'
          }
          aria-label="Attach files"
          className="flex items-center justify-center rounded bg-zinc-200 px-3 py-2 text-sm text-zinc-700 hover:bg-zinc-300 disabled:opacity-40 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600"
        >
          <PaperclipIcon size={16} />
        </button>
        <textarea
          className="max-h-32 min-h-[2.5rem] flex-1 resize-none rounded border border-zinc-300 px-2 py-1.5 text-sm focus:border-zinc-400 focus:outline-none disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-zinc-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:placeholder-zinc-500 dark:focus:border-zinc-500 dark:disabled:bg-zinc-800/60 dark:disabled:text-zinc-500"
          placeholder={
            composerLocked
              ? 'Download the model to start chatting…'
              : settings.pageContext
                ? 'Ask about this page…'
                : 'Ask anything — page context is off…'
          }
          rows={1}
          value={input}
          disabled={composerLocked}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (!composerLocked) submit(input);
            }
          }}
        />
        {/* Push-to-talk mic */}
        <button
          type="button"
          onClick={() => void togglePtt()}
          disabled={composerLocked || voiceState !== 'off' || voiceLoading !== null}
          title={pttActive ? 'Stop recording' : 'Push to talk — record a message'}
          aria-label={pttActive ? 'Stop recording' : 'Push to talk'}
          className={
            'flex items-center justify-center rounded px-3 py-2 text-sm tabular-nums ' +
            (pttActive
              ? 'bg-red-600 text-white hover:bg-red-500'
              : 'bg-zinc-200 text-zinc-700 hover:bg-zinc-300 disabled:opacity-40 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600')
          }
        >
          {voiceLoading === 'ptt' ? (
            sttDownloading ? `${sttDownloadPct}%` : '…'
          ) : pttActive ? (
            <StopIcon size={16} />
          ) : (
            <MicIcon size={16} />
          )}
        </button>
        {/* Hands-free voice mode */}
        <button
          type="button"
          onClick={() => void toggleVoiceMode()}
          disabled={composerLocked || pttActive || voiceLoading !== null}
          title={
            voiceState !== 'off' ? 'Turn off voice mode' : 'Hands-free voice mode — talk, and hear replies read back'
          }
          aria-label="Toggle hands-free voice mode"
          aria-pressed={voiceState !== 'off'}
          className={
            'flex items-center justify-center rounded px-3 py-2 text-sm tabular-nums ' +
            (voiceState !== 'off'
              ? 'bg-indigo-600 text-white hover:bg-indigo-500'
              : 'bg-zinc-200 text-zinc-700 hover:bg-zinc-300 disabled:opacity-40 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600')
          }
        >
          {voiceLoading === 'voice' ? (
            sttDownloading ? `${sttDownloadPct}%` : ttsDownloading ? `${ttsDownloadPct}%` : '…'
          ) : (
            <HeadphonesIcon size={16} />
          )}
        </button>
        {streaming ? (
          <button
            className="rounded bg-zinc-200 px-3 py-2 text-sm hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-600"
            onClick={stop}
          >
            Stop
          </button>
        ) : (
          <button
            className="rounded bg-zinc-800 px-3 py-2 text-sm text-white hover:bg-zinc-700 disabled:opacity-40 dark:bg-zinc-700 dark:hover:bg-zinc-600"
            disabled={!input.trim() || composerLocked}
            onClick={() => submit(input)}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
