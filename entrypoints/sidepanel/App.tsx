import { useEffect, useRef, useState } from 'react';
import { browser } from 'wxt/browser';
import { loadSettings, saveSettings } from '@/lib/settings';
import {
  WEBLLM_MODELS,
  PORT_NAME,
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
import { buildMessages, pageBudgetTokens, estimateTokens, type BuiltPrompt } from '@/lib/prompt';
import { chunkPage } from '@/lib/chunking';
import { summarizeChunks } from '@/lib/summarize';
import { TASKS, extractTask, resolveTemperature, EXTRACTION_SCHEMAS, type TaskSpec, type ExtractionSchema } from '@/lib/tasks';
import { Markdown } from './Markdown';
import { Logo } from './Logo';
import { applyTheme } from '@/lib/theme';
import {
  DEFAULT_SETTINGS,
  MAX_CONTEXT_TOKENS,
  MIN_CONTEXT_TOKENS,
  type ChatMessage,
  type MessageSource,
  type PageContent,
  type PendingAction,
  type Settings,
  type Theme,
} from '@/lib/types';

const PENDING_KEY = 'pendingAction';
const PAGE_CAPTURE_TIMEOUT_MS = 8_000;
const FIRST_TOKEN_TIMEOUT_MS = 90_000;
const NEXT_TOKEN_TIMEOUT_MS = 45_000;
const AUTO_SCROLL_BOTTOM_PX = 48;

// chrome.scripting isn't in the polyfill types; access the global directly.
const scripting = (globalThis as unknown as {
  chrome: { scripting: { executeScript(opts: { target: { tabId: number }; files: string[] }): Promise<unknown> } };
}).chrome.scripting;

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

/** Capture the active tab's content. If the content script isn't present (e.g. the tab was
 *  open before the extension was installed/updated), inject it on demand and retry. */
async function capturePage(wantViewport = false): Promise<PageContent> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error('No active tab.');
  const tabId = tab.id;
  const ask = () =>
    browser.tabs.sendMessage(tabId, { type: 'GET_PAGE_CONTENT', wantViewport }) as Promise<PageContent>;

  try {
    return await ask();
  } catch {
    // Content script not loaded in this tab — inject it, then retry once.
    try {
      await scripting.executeScript({ target: { tabId }, files: ['content-scripts/content.js'] });
      return await ask();
    } catch {
      throw new Error(
        "Can't read this page. Restricted pages (chrome://, the Chrome Web Store, PDFs, and local files) can't be read.",
      );
    }
  }
}

/** Ask the active tab's content script to scroll a source snippet into view and highlight it. */
async function jumpToSource(text: string) {
  try {
    const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) await browser.tabs.sendMessage(tab.id, { type: 'SCROLL_TO_TEXT', text });
  } catch {
    /* content script not reachable on this tab */
  }
}

/** Remove Qwen-style <think>…</think> reasoning from displayed output.
 *  While a block is still open mid-stream, hide everything from it onward. */
function stripThink(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  const open = out.lastIndexOf('<think>');
  if (open !== -1 && !/<\/think>/i.test(out.slice(open))) out = out.slice(0, open);
  return out.replace(/^\s+/, '');
}

function contextForSettings(s: Settings): number {
  return Math.max(MIN_CONTEXT_TOKENS, Math.min(s.webllmCtx, MAX_CONTEXT_TOKENS, webllmModel(s.webllmModel).maxCtx));
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

/** Collapsible provenance: which page sections grounded a retrieval-based answer. */
function SourcesList({ sources }: { sources: MessageSource[] }) {
  return (
    <details className="mt-2 border-t border-zinc-200 pt-1.5 text-[11px] text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
      <summary className="cursor-pointer select-none">Sources ({sources.length})</summary>
      <ul className="mt-1 space-y-1">
        {sources.map((s, i) => (
          <li key={i}>
            <button
              onClick={() => jumpToSource(s.snippet)}
              className="w-full rounded bg-zinc-50 px-2 py-1 text-left hover:bg-zinc-100 dark:bg-zinc-900/50 dark:hover:bg-zinc-800"
              title="Scroll to this section on the page"
            >
              {s.heading && <span className="font-medium text-zinc-600 dark:text-zinc-300">{s.heading}: </span>}
              <span>{s.snippet}…</span>
              <span className="ml-1 text-zinc-400">({Math.round(s.score * 100)}%)</span>
            </button>
          </li>
        ))}
      </ul>
    </details>
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
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pageNote, setPageNote] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
  /** Model lifecycle shown in the header: downloading → loading → ready. */
  const [modelStatus, setModelStatus] = useState<'idle' | 'downloading' | 'loading' | 'ready'>('idle');

  // Long-lived port to the offscreen document that hosts the in-browser engine.
  const portRef = useRef<WebllmPort | null>(null);
  const reqRef = useRef(0);
  const preloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const prewarmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastIndexedHashRef = useRef<string | null>(null);

  // Refs keep submit() free of stale closures (settings load async; pending action auto-runs).
  const settingsRef = useRef(settings);
  const messagesRef = useRef(messages);
  settingsRef.current = settings;
  messagesRef.current = messages;
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);

  // Initial load: settings, then any queued context-menu action.
  useEffect(() => {
    (async () => {
      const s = await loadSettings();
      setSettings(s);
      settingsRef.current = s;

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
      // Warm start: index the page and stage the model before the first question.
      schedulePrewarm(300);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

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
    });
    portRef.current = port;
    return port;
  }

  /** Free the model from memory by closing the offscreen document. */
  async function releaseModel() {
    cancelScheduledPreload();
    portRef.current?.disconnect();
    portRef.current = null;
    await releaseOffscreen();
    setModelStatus('idle');
    setPageNote('Model released — it will reload on your next question.');
  }

  /** Proactively (re)load the model so it's ready before the next message. */
  async function preloadModel(s: Settings) {
    if (!webgpuAvailable() || abortRef.current) return;
    try {
      const port = await getWebllmPort();
      setModelStatus('loading');
      await initModel(port, s.webllmModel, contextForSettings(s), (p) => setLoadProgress(p));
      setModelStatus('ready');
    } catch {
      setModelStatus('idle'); /* a real error surfaces on the next send */
    } finally {
      setLoadProgress(null);
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

  /** Quietly prepare for the first question: capture + chunk + embed the current page
   *  (CPU work) and load the model into the GPU if its weights are already downloaded.
   *  Best-effort — errors are swallowed and nothing runs while a generation streams. */
  async function prewarm() {
    const s = settingsRef.current;
    if (!webgpuAvailable() || abortRef.current) return;

    // Page indexing is best-effort: capture fails on restricted pages (chrome://, stores,
    // or when no readable tab is active) and that must NOT stop the model from staging.
    let page: PageContent | null = null;
    try {
      page = await capturePage();
    } catch {
      /* no readable page right now — still stage the model below */
    }
    try {
      const port = await getWebllmPort();
      if (page) {
        const hash = hashText(page.textContent);
        if (
          lastIndexedHashRef.current !== hash &&
          estimateTokens(page.textContent) > webllmModel(s.webllmModel).safePromptTokens
        ) {
          await indexPage(port, page.url, hash, chunkPage(page), () => {});
          lastIndexedHashRef.current = hash;
        }
      }
      if (abortRef.current) return;
      // Stage the model eagerly: load from disk if cached, otherwise download it now so the
      // first chat never pays for it. The header badge tracks which phase we're in.
      setModelStatus((cur) => (cur === 'ready' ? cur : 'loading'));
      const loaded = await prewarmModel(port, s.webllmModel, contextForSettings(s), (p) => setLoadProgress(p));
      if (!loaded) {
        setModelStatus('downloading');
        await initModel(port, s.webllmModel, contextForSettings(s), (p) => setLoadProgress(p));
      }
      setModelStatus('ready');
    } catch {
      // A failed prewarm never demotes a model that is already resident.
      setModelStatus((cur) => (cur === 'ready' ? cur : 'idle'));
    } finally {
      setLoadProgress(null);
    }
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

  function onChatScroll() {
    const el = scrollRef.current;
    if (el) shouldAutoScrollRef.current = isNearBottom(el);
  }

  async function submit(
    text: string,
    opts?: { selectionOverride?: string; task?: TaskSpec; schema?: ExtractionSchema },
  ) {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;

    const s = settingsRef.current;
    const spec = opts?.task ?? TASKS.ask;
    const conversation: ChatMessage[] = [...messagesRef.current, { role: 'user', content: trimmed }];
    shouldAutoScrollRef.current = true;
    setMessages([...conversation, { role: 'assistant', content: '' }]);
    setInput('');

    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true);
    let streamIterator: AsyncIterator<string> | null = null;

    try {
      if (!webgpuAvailable()) {
        throw new Error('This browser has no WebGPU. Use a recent Chromium browser (Chrome, Edge, or Brave).');
      }
      setPageNote('Reading page...');
      const page = await withTimeout(
        capturePage(s.viewportBoost),
        PAGE_CAPTURE_TIMEOUT_MS,
        'This page took too long to read. LinkedIn pages can be heavy; select the relevant text and try again.',
      );
      if (controller.signal.aborted) {
        setLastAssistant('_(stopped)_');
        return;
      }
      if (opts?.selectionOverride && !page.selection.trim()) {
        page.selection = opts.selectionOverride;
      }
      setPageNote('Preparing page context...');
      const runtimeCtx = contextForSettings(s);
      const model = webllmModel(s.webllmModel);
      const budget = pageBudgetTokens(s, page, conversation, runtimeCtx, spec.systemPrompt || undefined);
      // The body a single prompt may carry: the context-window budget, hard-capped per model.
      // Big models get SMALLER prompts — one oversized prefill on an integrated GPU can starve
      // the OS compositor and take down the whole session (see docs/large-page-handling.md).
      const bodyBudget = Math.min(budget, model.safePromptTokens);
      const words = Math.max(1, Math.round(page.textContent.length / 6));
      const sysOverride = spec.systemPrompt || undefined;
      // Retrieve from (rather than truncate) the page whenever the task allows it and the page
      // overflows what this model can safely take in one prompt. A selection no longer disables
      // retrieval — it sharpens the query instead (the selection itself rides in the header).
      const useRag = spec.allowRag && estimateTokens(page.textContent) > bodyBudget;

      let built: BuiltPrompt;
      if (useRag) {
        const port = await getWebllmPort();
        const chunks = chunkPage(page);
        const hash = hashText(page.textContent);
        setPageNote('Indexing page for retrieval…');
        const { fromCache } = await indexPage(port, page.url, hash, chunks, (p) => setLoadProgress(p));
        setLoadProgress(null);
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
            const compressed = await compressTexts(port, results.map((r) => r.text), 0.5);
            results = results.map((r, i) => ({ ...r, text: compressed[i] ?? r.text }));
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
        setPageNote(
          `Answering from the ${results.length} most relevant section${results.length === 1 ? '' : 's'} of this ~${words.toLocaleString()}-word page${fromCache ? '' : ' (indexed just now)'}.`,
        );
      } else {
        built = buildMessages(s, page, conversation, runtimeCtx, {
          systemPromptOverride: sysOverride,
          maxBodyTokens: model.safePromptTokens,
        });
        if (spec.supportsMapReduce && built.truncated) {
          setPageNote('Preparing to summarize a long page…');
        } else if (built.truncated) {
          const pct = built.totalChars ? Math.round((built.sentChars / built.totalChars) * 100) : 0;
          setPageNote(
            `⚠ Long page (~${words.toLocaleString()} words): sent the first ${pct}% as complete sections. Ask a question to search the whole page.`,
          );
        } else if (built.sourceTruncated) {
          setPageNote(`⚠ Very large page: read a bounded slice (~${words.toLocaleString()} words) so the extension stays responsive.`);
        } else {
          setPageNote(`Read the full page (~${words.toLocaleString()} words).`);
        }
      }

      // A large page under the summarize task is handled by map-reduce, not truncation.
      const useMapReduce = spec.supportsMapReduce && built.truncated;

      const port = await getWebllmPort();
      setModelStatus((cur) => (cur === 'ready' ? cur : 'loading'));
      await initModel(port, s.webllmModel, runtimeCtx, (p) => setLoadProgress(p));
      setModelStatus('ready');
      setLoadProgress(null);
      if (controller.signal.aborted) {
        setLastAssistant('_(stopped)_');
        return;
      }

      const genOptions: GenerateOptions = { temperature: resolveTemperature(spec, s), maxTokens: spec.maxTokens };
      // Qwen3 models reason in hidden <think> blocks by default — the user just sees the
      // spinner while tokens they'll never read stream at decode speed. Hard-disable it for
      // every task via the chat-template flag (the soft "/no_think" text hint was unreliable).
      if (/Qwen3/i.test(s.webllmModel)) genOptions.enableThinking = false;
      if (spec.outputMode === 'json' && opts?.schema) {
        genOptions.responseFormat = { type: 'json_object', schema: JSON.stringify(opts.schema.schema) };
      }

      if (useMapReduce) {
        // Whole-page summary of an over-budget page: summarize sections, then merge.
        const chunks = chunkPage(page);
        const qwen = /Qwen3/i.test(s.webllmModel);
        const runOne = async (
          messages: ChatMessage[],
          o: { temperature: number; maxTokens: number; stream?: boolean },
        ): Promise<string> => {
          const rid = ++reqRef.current;
          const ro: GenerateOptions = { temperature: o.temperature, maxTokens: o.maxTokens, enableThinking: qwen ? false : undefined };
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
          { ctxTokens: runtimeCtx, title: page.title || 'this page', maxPromptTokens: model.safePromptTokens },
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
        setPageNote(`Summarized this ~${words.toLocaleString()}-word page in ${chunks.length} sections.`);
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
        while (true) {
          const next = await withTimeout(
            streamIterator.next(),
            receivedToken ? NEXT_TOKEN_TIMEOUT_MS : FIRST_TOKEN_TIMEOUT_MS,
            'The local model took too long on this page, so I stopped it. Try selecting the most relevant text or asking a narrower question.',
            () => controller.abort(),
          );
          if (next.done) break;
          const delta = next.value;
          receivedToken = true;
          acc += delta;
          setLastAssistant(acc);
        }
        if (!acc) setLastAssistant(controller.signal.aborted ? '_(stopped)_' : '_(no response)_');
        if (acc && built.usedChunks?.length) {
          const sources: MessageSource[] = [...built.usedChunks]
            .sort((a, b) => b.score - a.score)
            .map((c) => ({
              heading: c.heading,
              snippet: c.text.replace(/\s+/g, ' ').trim().slice(0, 140),
              score: c.score,
              ordinal: c.ordinal,
            }));
          setMessages((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last?.role === 'assistant') copy[copy.length - 1] = { ...last, sources };
            return copy;
          });
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
      const userStopped = e instanceof DOMException && e.name === 'AbortError';
      setLastAssistant(userStopped ? '_(stopped)_' : `⚠️ ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setStreaming(false);
      setLoadProgress(null);
      abortRef.current = null;
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
          { role: 'assistant', content: 'Select some text on the page first, then click “Explain selection”.' },
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
    if (changedKeys.includes('webllmModel') || changedKeys.includes('webllmCtx')) schedulePreload(merged);
  }

  const disabled = streaming;

  return (
    <div className="flex h-full flex-col bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
        <Logo size={18} />
        <span className="text-sm font-semibold tracking-tight">Enclave</span>
        {modelStatus !== 'idle' && (
          <span
            className={
              modelStatus === 'ready'
                ? 'flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300'
                : 'flex items-center gap-1 rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400'
            }
            title={
              modelStatus === 'ready'
                ? 'Model is loaded on the GPU — answers start immediately.'
                : modelStatus === 'downloading'
                  ? 'First-time download of this model; it is cached afterwards.'
                  : 'Loading model weights onto the GPU.'
            }
          >
            <span
              className={
                modelStatus === 'ready'
                  ? 'h-1.5 w-1.5 rounded-full bg-emerald-500'
                  : 'h-1.5 w-1.5 animate-pulse rounded-full bg-zinc-400'
              }
            />
            {modelStatus === 'ready' ? 'Ready to chat' : modelStatus === 'downloading' ? 'Downloading model…' : 'Loading model…'}
          </span>
        )}
        <select
          className="ml-auto max-w-[12rem] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
          value={settings.webllmModel}
          onChange={(e) => onSaveSettings({ webllmModel: e.target.value })}
          title="Model"
        >
          {WEBLLM_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} · ~{m.approxGb} GB
            </option>
          ))}
        </select>
        <button
          className="rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
          onClick={() => setShowSettings((v) => !v)}
          title="Settings"
        >
          ⚙
        </button>
      </header>

      {/* Model download / load progress */}
      {loadProgress && (
        <div className="border-b border-zinc-200 bg-zinc-100 px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-zinc-600 dark:text-zinc-300">Loading model… {Math.round(loadProgress.progress * 100)}%</p>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-zinc-300 dark:bg-zinc-700">
            <div
              className="h-full bg-zinc-600 transition-all dark:bg-zinc-300"
              style={{ width: `${Math.round(loadProgress.progress * 100)}%` }}
            />
          </div>
          <p className="mt-1 truncate text-[10px] text-zinc-400">{loadProgress.text}</p>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && <SettingsPanel settings={settings} onSave={onSaveSettings} onRelease={releaseModel} busy={streaming} />}

      {/* Page coverage note */}
      {pageNote && (
        <div
          className={
            pageNote.startsWith('⚠')
              ? 'border-b border-amber-200 bg-amber-50 px-3 py-1.5 text-[11px] text-amber-800 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-300'
              : 'border-b border-zinc-200 bg-zinc-100 px-3 py-1.5 text-[11px] text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400'
          }
        >
          {pageNote}
        </div>
      )}

      {/* Chat thread */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3" onScroll={onChatScroll}>
        {messages.length === 0 && (
          <div className="mt-6 text-center text-xs text-zinc-400 dark:text-zinc-500">
            Ask anything about the current page. Everything runs locally on your device.
          </div>
        )}
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          const isPlaceholder = !m.content && streaming && i === messages.length - 1;
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
                m.content
              ) : isPlaceholder ? (
                <ThinkingIndicator />
              ) : m.structured ? (
                <>
                  <StructuredResult data={m.structured.data} raw={m.content} />
                  {m.sources && m.sources.length > 0 && <SourcesList sources={m.sources} />}
                </>
              ) : (
                <>
                  {(() => {
                    const display = stripThink(m.content);
                    return display ? <Markdown content={display} /> : <ThinkingIndicator />;
                  })()}
                  {m.sources && m.sources.length > 0 && <SourcesList sources={m.sources} />}
                </>
              )}
            </div>
          );
        })}
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-2 border-t border-zinc-200 bg-white px-3 pt-2 dark:border-zinc-800 dark:bg-zinc-900">
        <button
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
          disabled={disabled}
          onClick={() => submit(TASKS.summarize.label, { task: TASKS.summarize })}
        >
          Summarize
        </button>
        <button
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
          disabled={disabled}
          onClick={explainSelection}
        >
          Explain selection
        </button>
        <select
          className="rounded-full border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800"
          disabled={disabled}
          value=""
          onChange={(e) => {
            const schema = EXTRACTION_SCHEMAS.find((x) => x.id === e.target.value);
            if (schema) submit(extractTask(schema).label, { task: extractTask(schema), schema });
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
      </div>

      {/* Composer */}
      <div className="flex items-end gap-2 bg-white px-3 pt-2 pb-3 dark:bg-zinc-900">
        <textarea
          className="max-h-32 min-h-[2.5rem] flex-1 resize-none rounded border border-zinc-300 px-2 py-1.5 text-sm focus:border-zinc-400 focus:outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100 dark:placeholder-zinc-500 dark:focus:border-zinc-500"
          placeholder="Ask about this page…"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit(input);
            }
          }}
        />
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
            disabled={!input.trim()}
            onClick={() => submit(input)}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}

function SettingsPanel({
  settings,
  onSave,
  onRelease,
  busy,
}: {
  settings: Settings;
  onSave: (patch: Partial<Settings>) => void;
  onRelease: () => void;
  busy: boolean;
}) {
  const inputCls = 'rounded border border-zinc-300 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100';
  const maxCtx = Math.min(webllmModel(settings.webllmModel).maxCtx, MAX_CONTEXT_TOKENS);
  return (
    <div className="space-y-3 border-b border-zinc-200 bg-zinc-100 px-3 py-3 text-xs dark:border-zinc-800 dark:bg-zinc-900">
      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">Theme</span>
        <div className="mt-1 inline-flex overflow-hidden rounded border border-zinc-300 dark:border-zinc-700">
          {(['system', 'light', 'dark'] as Theme[]).map((t) => (
            <button
              key={t}
              className={
                'px-2.5 py-1 capitalize ' +
                (settings.theme === t
                  ? 'bg-zinc-800 text-white dark:bg-zinc-600'
                  : 'bg-white hover:bg-zinc-50 dark:bg-zinc-800 dark:hover:bg-zinc-700')
              }
              onClick={() => onSave({ theme: t })}
            >
              {t}
            </button>
          ))}
        </div>
      </label>

      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">Model</span>
        <select
          className={`mt-1 w-full ${inputCls}`}
          value={settings.webllmModel}
          onChange={(e) => onSave({ webllmModel: e.target.value })}
        >
          {WEBLLM_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} · ~{m.approxGb} GB · {m.note}
            </option>
          ))}
        </select>
        <p className="mt-1 text-[10px] text-zinc-400">
          Larger models give better answers but need more GPU memory (and a bigger one-time download). Each is cached after first use.
        </p>
      </label>

      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">Context (tokens)</span>
        <input
          type="number"
          className={`mt-1 w-full ${inputCls}`}
          value={settings.webllmCtx}
          min={MIN_CONTEXT_TOKENS}
          max={maxCtx}
          step={MIN_CONTEXT_TOKENS}
          onChange={(e) => {
            const value = Number(e.target.value);
            if (!Number.isFinite(value)) return;
            onSave({ webllmCtx: Math.max(MIN_CONTEXT_TOKENS, Math.min(maxCtx, value || DEFAULT_SETTINGS.webllmCtx)) });
          }}
        />
        <p className="mt-1 text-[10px] text-zinc-400">
          Higher reads more of the page but reserves more GPU memory up front. Enclave caps this at {maxCtx.toLocaleString()} for stability; changing it reloads the model.
        </p>
      </label>

      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">Temperature</span>
        <input
          type="number"
          className={`mt-1 w-full ${inputCls}`}
          value={settings.temperature}
          min={0}
          max={1}
          step={0.1}
          onChange={(e) => onSave({ temperature: Number(e.target.value) })}
        />
      </label>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={settings.compressContext}
          onChange={(e) => onSave({ compressContext: e.target.checked })}
        />
        <span className="text-zinc-500 dark:text-zinc-400">Compress retrieved context (experimental)</span>
      </label>

      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={settings.viewportBoost}
          onChange={(e) => onSave({ viewportBoost: e.target.checked })}
        />
        <span className="text-zinc-500 dark:text-zinc-400">Prefer on-screen content in answers (experimental)</span>
      </label>

      <button
        type="button"
        disabled={busy}
        onClick={onRelease}
        className="rounded border border-zinc-300 bg-white px-2 py-1 hover:bg-zinc-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700"
      >
        Release model from memory
      </button>
    </div>
  );
}
