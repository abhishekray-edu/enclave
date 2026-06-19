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
  streamGenerate,
  type LoadProgress,
  type WebllmPort,
} from '@/lib/webllmClient';
import { buildMessages } from '@/lib/prompt';
import { Markdown } from './Markdown';
import { Logo } from './Logo';
import { applyTheme } from '@/lib/theme';
import { DEFAULT_SETTINGS, type ChatMessage, type PageContent, type PendingAction, type Settings, type Theme } from '@/lib/types';

const PENDING_KEY = 'pendingAction';

// chrome.scripting isn't in the polyfill types; access the global directly.
const scripting = (globalThis as unknown as {
  chrome: { scripting: { executeScript(opts: { target: { tabId: number }; files: string[] }): Promise<unknown> } };
}).chrome.scripting;

/** Capture the active tab's content. If the content script isn't present (e.g. the tab was
 *  open before the extension was installed/updated), inject it on demand and retry. */
async function capturePage(): Promise<PageContent> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error('No active tab.');
  const tabId = tab.id;
  const ask = () => browser.tabs.sendMessage(tabId, { type: 'GET_PAGE_CONTENT' }) as Promise<PageContent>;

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

/** Remove Qwen-style <think>…</think> reasoning from displayed output.
 *  While a block is still open mid-stream, hide everything from it onward. */
function stripThink(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  const open = out.lastIndexOf('<think>');
  if (open !== -1 && !/<\/think>/i.test(out.slice(open))) out = out.slice(0, open);
  return out.replace(/^\s+/, '');
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pageNote, setPageNote] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);

  // Long-lived port to the offscreen document that hosts the in-browser engine.
  const portRef = useRef<WebllmPort | null>(null);
  const reqRef = useRef(0);
  const preloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refs keep submit() free of stale closures (settings load async; pending action auto-runs).
  const settingsRef = useRef(settings);
  const messagesRef = useRef(messages);
  settingsRef.current = settings;
  messagesRef.current = messages;
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

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
          void submit('Explain the selected text in simple, clear terms.', {
            selectionOverride: pending.selection,
          });
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

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
    portRef.current?.disconnect();
    portRef.current = null;
    await releaseOffscreen();
    setPageNote('Model released — it will reload on your next question.');
  }

  /** Proactively (re)load the model so it's ready before the next message. */
  async function preloadModel(s: Settings) {
    if (!webgpuAvailable() || abortRef.current) return;
    try {
      const port = await getWebllmPort();
      await initModel(port, s.webllmModel, s.webllmCtx, (p) => setLoadProgress(p));
    } catch {
      /* a real error surfaces on the next send */
    } finally {
      setLoadProgress(null);
    }
  }

  /** Debounce reloads so editing the context number doesn't reload on every keystroke. */
  function schedulePreload(s: Settings) {
    if (preloadTimerRef.current) clearTimeout(preloadTimerRef.current);
    preloadTimerRef.current = setTimeout(() => void preloadModel(s), 700);
  }

  function setLastAssistant(content: string) {
    setMessages((prev) => {
      const copy = [...prev];
      copy[copy.length - 1] = { role: 'assistant', content };
      return copy;
    });
  }

  async function submit(text: string, opts?: { selectionOverride?: string }) {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;

    const s = settingsRef.current;
    const conversation: ChatMessage[] = [...messagesRef.current, { role: 'user', content: trimmed }];
    setMessages([...conversation, { role: 'assistant', content: '' }]);
    setInput('');

    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true);

    try {
      if (!webgpuAvailable()) {
        throw new Error('This browser has no WebGPU. Use a recent Chromium browser (Chrome, Edge, or Brave).');
      }
      const page = await capturePage();
      if (opts?.selectionOverride && !page.selection.trim()) {
        page.selection = opts.selectionOverride;
      }
      const built = buildMessages(s, page, conversation, s.webllmCtx);
      const words = Math.max(1, Math.round(built.totalChars / 6));
      if (built.truncated) {
        const pct = Math.round((built.sentChars / built.totalChars) * 100);
        setPageNote(
          `⚠ Long page (~${words.toLocaleString()} words): only the first ~${pct}% was sent. Raise the context in ⚙ to read more (uses more memory).`,
        );
      } else {
        setPageNote(`Read the full page (~${words.toLocaleString()} words).`);
      }

      // Qwen3 is a hybrid reasoning model that emits <think> blocks by default.
      // Its soft switch is read from the user turn; use a COPY so the displayed message isn't altered.
      if (/Qwen3/i.test(s.webllmModel)) {
        const i = built.messages.length - 1;
        const last = built.messages[i];
        if (last.role === 'user') {
          built.messages[i] = { ...last, content: `${last.content} /no_think` };
        }
      }

      const port = await getWebllmPort();
      await initModel(port, s.webllmModel, s.webllmCtx, (p) => setLoadProgress(p));
      setLoadProgress(null);

      const stream = streamGenerate(port, ++reqRef.current, built.messages, s.temperature, controller.signal);
      let acc = '';
      for await (const delta of stream) {
        acc += delta;
        setLastAssistant(acc);
      }
      if (!acc) setLastAssistant('_(no response)_');
    } catch (e) {
      setLastAssistant(`⚠️ ${e instanceof Error ? e.message : String(e)}`);
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
    void submit('Explain the selected text in simple, clear terms.');
  }

  async function onSaveSettings(patch: Partial<Settings>) {
    // Switching model clamps the context to that model's maximum.
    let next = patch;
    if (patch.webllmModel) {
      const max = webllmModel(patch.webllmModel).maxCtx;
      const cur = patch.webllmCtx ?? settingsRef.current.webllmCtx;
      if (cur > max) next = { ...patch, webllmCtx: max };
    }
    const merged = await saveSettings(next);
    setSettings(merged);
    settingsRef.current = merged;
    if ('webllmModel' in next || 'webllmCtx' in next) schedulePreload(merged);
  }

  const disabled = streaming;

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
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-3 py-3">
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
                <span className="text-sm text-zinc-400 dark:text-zinc-500">…</span>
              ) : (
                (() => {
                  const display = stripThink(m.content);
                  return display ? (
                    <Markdown content={display} />
                  ) : (
                    <span className="text-sm text-zinc-400 dark:text-zinc-500">Thinking…</span>
                  );
                })()
              )}
            </div>
          );
        })}
      </div>

      {/* Quick actions */}
      <div className="flex gap-2 border-t border-zinc-200 bg-white px-3 pt-2 dark:border-zinc-800 dark:bg-zinc-900">
        <button
          className="rounded-full border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:hover:bg-zinc-800"
          disabled={disabled}
          onClick={() => submit('Summarize this page in a few concise bullet points.')}
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
  const maxCtx = webllmModel(settings.webllmModel).maxCtx;
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
          min={2048}
          max={maxCtx}
          step={2048}
          onChange={(e) =>
            onSave({ webllmCtx: Math.min(maxCtx, Number(e.target.value) || DEFAULT_SETTINGS.webllmCtx) })
          }
        />
        <p className="mt-1 text-[10px] text-zinc-400">
          Higher reads more of the page but reserves more GPU memory up front. This model supports up to {maxCtx.toLocaleString()}; changing it reloads the model.
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

      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">System prompt</span>
        <textarea
          className={`mt-1 h-20 w-full resize-none ${inputCls}`}
          value={settings.systemPrompt}
          onChange={(e) => onSave({ systemPrompt: e.target.value })}
        />
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
