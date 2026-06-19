import { useEffect, useRef, useState } from 'react';
import { browser } from 'wxt/browser';
import { loadSettings, saveSettings } from '@/lib/settings';
import { chatStream, listModels, OllamaConnectionError } from '@/lib/ollama';
import {
  WEBLLM_MODELS,
  PORT_NAME,
  webgpuAvailable,
  ensureOffscreen,
  releaseOffscreen,
  initModel,
  streamGenerate,
  type WebllmPort,
} from '@/lib/webllmClient';
import { buildMessages } from '@/lib/prompt';
import { Markdown } from './Markdown';
import { Logo } from './Logo';
import { applyTheme } from '@/lib/theme';
import { DEFAULT_SETTINGS, type ChatMessage, type Engine, type PageContent, type PendingAction, type Settings, type Theme } from '@/lib/types';

const PENDING_KEY = 'pendingAction';

/** Capture the active tab's content via the content script. */
async function capturePage(): Promise<PageContent> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error('No active tab.');
  try {
    return (await browser.tabs.sendMessage(tab.id, { type: 'GET_PAGE_CONTENT' })) as PageContent;
  } catch {
    throw new Error(
      "Can't read this page. Try reloading the tab — restricted pages (chrome://, the Web Store, PDFs) can't be read.",
    );
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
  const [models, setModels] = useState<string[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pageNote, setPageNote] = useState<string | null>(null);
  const [loadProgress, setLoadProgress] = useState<{ text: string; progress: number } | null>(null);

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

  // Initial load: settings, model list, then any queued context-menu action.
  useEffect(() => {
    (async () => {
      const s = await loadSettings();
      setSettings(s);
      settingsRef.current = s;
      if (s.engine === 'ollama') await refreshModels(s.endpoint);

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

  async function refreshModels(endpoint: string) {
    try {
      const list = await listModels(endpoint);
      setModels(list);
      setConnectionError(null);
    } catch (e) {
      setConnectionError(e instanceof Error ? e.message : String(e));
    }
  }

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

  /** Free the in-browser model from memory by closing the offscreen document. */
  async function releaseModel() {
    portRef.current?.disconnect();
    portRef.current = null;
    await releaseOffscreen();
    setPageNote('In-browser model released — it will reload on your next question.');
  }

  /** Proactively (re)load the in-browser model so it's ready before the next message. */
  async function preloadWebllm(s: Settings) {
    if (s.engine !== 'webllm' || !webgpuAvailable() || abortRef.current) return;
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
    preloadTimerRef.current = setTimeout(() => void preloadWebllm(s), 700);
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
      const page = await capturePage();
      if (opts?.selectionOverride && !page.selection.trim()) {
        page.selection = opts.selectionOverride;
      }
      const isWebllm = s.engine === 'webllm';
      const ctxTokens = isWebllm ? s.webllmCtx : s.numCtx;
      const built = buildMessages(s, page, conversation, ctxTokens);
      const words = Math.max(1, Math.round(built.totalChars / 6));
      if (built.truncated) {
        const pct = Math.round((built.sentChars / built.totalChars) * 100);
        const hint = isWebllm ? 'In-browser models use a small window.' : 'Raise context in ⚙ for fuller coverage (slower).';
        setPageNote(`⚠ Long page (~${words.toLocaleString()} words): only the first ~${pct}% was sent. ${hint}`);
      } else {
        setPageNote(`Read the full page (~${words.toLocaleString()} words).`);
      }

      // Qwen3 is a hybrid reasoning model that emits <think> blocks by default.
      // Its soft switch is read from the user turn, so disable thinking there.
      // Replace the slot with a COPY so the displayed message isn't altered.
      if (isWebllm && /Qwen3/i.test(s.webllmModel)) {
        const i = built.messages.length - 1;
        const last = built.messages[i];
        if (last.role === 'user') {
          built.messages[i] = { ...last, content: `${last.content} /no_think` };
        }
      }

      let stream: AsyncGenerator<string, void, unknown>;
      if (isWebllm) {
        if (!webgpuAvailable()) {
          throw new Error(
            'This browser has no WebGPU, so the in-browser engine can’t run. Use a recent Chromium browser, or switch to Ollama in ⚙.',
          );
        }
        const port = await getWebllmPort();
        await initModel(port, s.webllmModel, s.webllmCtx, (p) => setLoadProgress(p));
        setLoadProgress(null);
        stream = streamGenerate(port, ++reqRef.current, built.messages, s.temperature, controller.signal);
      } else {
        stream = chatStream({ settings: s, messages: built.messages, signal: controller.signal });
      }

      let acc = '';
      for await (const delta of stream) {
        acc += delta;
        setLastAssistant(acc);
      }
      if (!acc) setLastAssistant('_(no response)_');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLastAssistant(`⚠️ ${msg}`);
      if (e instanceof OllamaConnectionError) setConnectionError(msg);
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
    // Pull the current selection so we can warn if there is none.
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
    const next = await saveSettings(patch);
    setSettings(next);
    settingsRef.current = next;
    if (patch.engine === 'ollama') void refreshModels(next.endpoint);
    // A model-affecting change should reload the in-browser engine now, not on next send.
    if (next.engine === 'webllm' && ('engine' in patch || 'webllmModel' in patch || 'webllmCtx' in patch)) {
      schedulePreload(next);
    }
  }

  const disabled = streaming;

  return (
    <div className="flex h-full flex-col bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
        <Logo size={18} />
        <span className="text-sm font-semibold tracking-tight">Enclave</span>
        {settings.engine === 'webllm' ? (
          <select
            className="ml-auto max-w-[11rem] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
            value={settings.webllmModel}
            onChange={(e) => onSaveSettings({ webllmModel: e.target.value })}
            title="In-browser model"
          >
            {WEBLLM_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        ) : (
          <select
            className="ml-auto max-w-[11rem] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
            value={settings.model}
            onChange={(e) => onSaveSettings({ model: e.target.value })}
            title="Ollama model"
          >
            {!models.includes(settings.model) && <option value={settings.model}>{settings.model}</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        )}
        <button
          className="rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
          onClick={() => setShowSettings((v) => !v)}
          title="Settings"
        >
          ⚙
        </button>
      </header>

      {/* In-browser model download / load progress */}
      {loadProgress && (
        <div className="border-b border-zinc-200 bg-zinc-100 px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900">
          <p className="text-zinc-600 dark:text-zinc-300">
            Loading in-browser model… {Math.round(loadProgress.progress * 100)}%
          </p>
          <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-zinc-300 dark:bg-zinc-700">
            <div
              className="h-full bg-zinc-600 transition-all dark:bg-zinc-300"
              style={{ width: `${Math.round(loadProgress.progress * 100)}%` }}
            />
          </div>
          <p className="mt-1 truncate text-[10px] text-zinc-400">{loadProgress.text}</p>
        </div>
      )}

      {/* Connection / CORS banner (Ollama engine only) */}
      {connectionError && settings.engine === 'ollama' && (
        <div className="border-b border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/40 dark:text-amber-200">
          <p className="font-medium">Can’t reach Ollama.</p>
          <p className="mt-1">{connectionError}</p>
          <p className="mt-1">
            Allow this extension, then restart Ollama:
            <code className="mt-1 block rounded bg-amber-100 px-1 py-0.5 font-mono dark:bg-amber-900/40">
              launchctl setenv OLLAMA_ORIGINS "chrome-extension://*"
            </code>
          </p>
          <button
            className="mt-1 rounded bg-amber-200 px-2 py-0.5 font-medium hover:bg-amber-300 dark:bg-amber-800 dark:text-amber-100 dark:hover:bg-amber-700"
            onClick={() => refreshModels(settings.endpoint)}
          >
            Retry
          </button>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onSave={onSaveSettings}
          onRefreshModels={() => refreshModels(settings.endpoint)}
          onRelease={releaseModel}
          busy={streaming}
        />
      )}

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
            Ask anything about the current page. Everything runs locally on your Mac.
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
  onRefreshModels,
  onRelease,
  busy,
}: {
  settings: Settings;
  onSave: (patch: Partial<Settings>) => void;
  onRefreshModels: () => void;
  onRelease: () => void;
  busy: boolean;
}) {
  const inputCls =
    'rounded border border-zinc-300 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100';
  return (
    <div className="space-y-2 border-b border-zinc-200 bg-zinc-100 px-3 py-3 text-xs dark:border-zinc-800 dark:bg-zinc-900">
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
        <span className="text-zinc-500 dark:text-zinc-400">Engine</span>
        <div className="mt-1 inline-flex overflow-hidden rounded border border-zinc-300 dark:border-zinc-700">
          {([['webllm', 'In-browser'], ['ollama', 'Ollama']] as [Engine, string][]).map(([e, label]) => (
            <button
              key={e}
              className={
                'px-2.5 py-1 ' +
                (settings.engine === e
                  ? 'bg-zinc-800 text-white dark:bg-zinc-600'
                  : 'bg-white hover:bg-zinc-50 dark:bg-zinc-800 dark:hover:bg-zinc-700')
              }
              onClick={() => onSave({ engine: e })}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="mt-1 text-[10px] text-zinc-400">
          {settings.engine === 'webllm'
            ? 'Runs on your GPU in the browser — no setup. First use downloads the model (~2.5GB, cached after). Stays loaded in the background so it’s instant next time.'
            : 'Uses your local Ollama server — fastest and supports bigger models; needs Ollama running.'}
        </p>
        {settings.engine === 'webllm' && (
          <button
            type="button"
            disabled={busy}
            onClick={onRelease}
            className="mt-2 rounded border border-zinc-300 bg-white px-2 py-1 hover:bg-zinc-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700"
          >
            Release model from memory
          </button>
        )}
      </label>

      {settings.engine === 'ollama' && (
        <>
          <label className="block">
            <span className="text-zinc-500 dark:text-zinc-400">Ollama endpoint</span>
            <div className="mt-1 flex gap-2">
              <input
                className={`flex-1 ${inputCls}`}
                value={settings.endpoint}
                onChange={(e) => onSave({ endpoint: e.target.value })}
              />
              <button
                className="rounded border border-zinc-300 bg-white px-2 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700"
                onClick={onRefreshModels}
              >
                Refresh
              </button>
            </div>
          </label>
          <label className="block">
            <span className="text-zinc-500 dark:text-zinc-400">Context (num_ctx)</span>
            <input
              type="number"
              className={`mt-1 w-full ${inputCls}`}
              value={settings.numCtx}
              min={2048}
              step={2048}
              onChange={(e) => onSave({ numCtx: Number(e.target.value) || DEFAULT_SETTINGS.numCtx })}
            />
          </label>
        </>
      )}

      {settings.engine === 'webllm' && (
        <label className="block">
          <span className="text-zinc-500 dark:text-zinc-400">Context (tokens)</span>
          <input
            type="number"
            className={`mt-1 w-full ${inputCls}`}
            value={settings.webllmCtx}
            min={2048}
            max={40960}
            step={2048}
            onChange={(e) => onSave({ webllmCtx: Number(e.target.value) || DEFAULT_SETTINGS.webllmCtx })}
          />
          <p className="mt-1 text-[10px] text-zinc-400">
            Higher reads more of the page but uses more GPU memory. Qwen3 4B supports up to 40960; changing this reloads the model.
          </p>
        </label>
      )}

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
    </div>
  );
}
