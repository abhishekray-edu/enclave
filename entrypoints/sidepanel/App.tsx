import { useEffect, useRef, useState } from 'react';
import { browser } from 'wxt/browser';
import { loadSettings, saveSettings } from '@/lib/settings';
import { chatStream, listModels, OllamaConnectionError } from '@/lib/ollama';
import { buildMessages } from '@/lib/prompt';
import { Markdown } from './Markdown';
import { Logo } from './Logo';
import { applyTheme } from '@/lib/theme';
import { DEFAULT_SETTINGS, type ChatMessage, type PageContent, type PendingAction, type Settings, type Theme } from '@/lib/types';

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

export default function App() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [models, setModels] = useState<string[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pageNote, setPageNote] = useState<string | null>(null);

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
      await refreshModels(s.endpoint);

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
      const built = buildMessages(s, page, conversation);
      const words = Math.max(1, Math.round(built.totalChars / 6));
      if (built.truncated) {
        const pct = Math.round((built.sentChars / built.totalChars) * 100);
        setPageNote(
          `⚠ Long page (~${words.toLocaleString()} words): only the first ~${pct}% was sent. Raise context in ⚙ for fuller coverage (slower).`,
        );
      } else {
        setPageNote(`Read the full page (~${words.toLocaleString()} words).`);
      }

      let acc = '';
      for await (const delta of chatStream({ settings: s, messages: built.messages, signal: controller.signal })) {
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
  }

  const disabled = streaming;

  return (
    <div className="flex h-full flex-col bg-zinc-100 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100">
      {/* Header */}
      <header className="flex items-center gap-2 border-b border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
        <Logo size={18} />
        <span className="text-sm font-semibold tracking-tight">Enclave</span>
        <select
          className="ml-auto max-w-[10rem] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"
          value={settings.model}
          onChange={(e) => onSaveSettings({ model: e.target.value })}
          title="Model"
        >
          {!models.includes(settings.model) && <option value={settings.model}>{settings.model}</option>}
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
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

      {/* Connection / CORS banner */}
      {connectionError && (
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
                <Markdown content={m.content} />
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
}: {
  settings: Settings;
  onSave: (patch: Partial<Settings>) => void;
  onRefreshModels: () => void;
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
      <div className="flex gap-2">
        <label className="flex-1">
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
        <label className="flex-1">
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
      </div>
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
