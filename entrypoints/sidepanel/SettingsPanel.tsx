// Settings, grouped for scannability. Primary controls (theme, model, the voice essentials)
// stay visible; the expert knobs and memory-release actions live in a collapsed "Advanced"
// disclosure at the bottom. Every control persists immediately via `onSave` (no Save button).
import { WEBLLM_MODELS, webllmModel } from '@/lib/webllmClient';
import { TTS_DOWNLOAD_MB } from '@/lib/ttsClient';
import { STT_DOWNLOAD_MB } from '@/lib/sttClient';
import {
  DEFAULT_SETTINGS,
  MAX_CONTEXT_TOKENS,
  MIN_CONTEXT_TOKENS,
  MIN_VOICE_PAUSE_MS,
  MAX_VOICE_PAUSE_MS,
  type Settings,
  type Theme,
} from '@/lib/types';
import { ChevronDownIcon } from './icons';

const inputCls =
  'rounded border border-zinc-300 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100';
const releaseBtnCls =
  'rounded border border-zinc-300 bg-white px-2 py-1 text-left hover:bg-zinc-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:hover:bg-zinc-700';

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
      {children}
    </span>
  );
}

function ExtLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="underline hover:text-zinc-600 dark:hover:text-zinc-300"
    >
      {children}
    </a>
  );
}

function Checkbox({
  checked,
  onChange,
  label,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  title?: string;
}) {
  return (
    <label className="flex items-center gap-2" title={title}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="text-zinc-600 dark:text-zinc-300">{label}</span>
    </label>
  );
}

export function SettingsPanel({
  settings,
  onSave,
  onRelease,
  busy,
  ttsLoaded,
  onReleaseVoice,
  sttLoaded,
  onReleaseVoiceInput,
}: {
  settings: Settings;
  onSave: (patch: Partial<Settings>) => void;
  onRelease: () => void;
  busy: boolean;
  ttsLoaded: boolean;
  onReleaseVoice: () => void;
  sttLoaded: boolean;
  onReleaseVoiceInput: () => void;
}) {
  const maxCtx = Math.min(webllmModel(settings.webllmModel).maxCtx, MAX_CONTEXT_TOKENS);
  const pauseSeconds = (settings.voicePauseMs / 1000).toFixed(1);

  return (
    <div className="space-y-4 border-b border-zinc-200 bg-zinc-100 px-3 py-3 text-xs dark:border-zinc-800 dark:bg-zinc-900">
      {/* Theme */}
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

      {/* Model */}
      <label className="block">
        <span className="text-zinc-500 dark:text-zinc-400">Model</span>
        <select
          className={`mt-1 w-full ${inputCls}`}
          value={settings.webllmModel}
          onChange={(e) => onSave({ webllmModel: e.target.value })}
          title="Bigger models answer better but need more GPU memory and a larger one-time download. Each is cached after first use."
        >
          {WEBLLM_MODELS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} · ~{m.approxGb} GB · {m.note}
            </option>
          ))}
        </select>
        <p className="mt-1 text-[10px] text-zinc-400">Downloads once, then runs from cache.</p>
      </label>

      <Checkbox
        checked={settings.autoLoadOnStartup}
        onChange={(v) => onSave({ autoLoadOnStartup: v })}
        label="Load models when the browser starts"
        title="Warms the AI model — and the voice models, if you've used voice — into memory at browser launch so the first answer is instant. Only loads models already downloaded; uses memory while the browser is open."
      />

      {/* Voice */}
      <div className="space-y-2.5 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-800">
        <SectionLabel>Voice</SectionLabel>
        <Checkbox
          checked={settings.ttsAutoRead}
          onChange={(v) => onSave({ ttsAutoRead: v })}
          label="Read replies aloud"
        />
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-zinc-600 dark:text-zinc-300">Pause before responding</span>
            <span className="tabular-nums text-zinc-400">{pauseSeconds}s</span>
          </div>
          <input
            type="range"
            className="mt-1 w-full accent-indigo-600"
            min={MIN_VOICE_PAUSE_MS}
            max={MAX_VOICE_PAUSE_MS}
            step={100}
            value={settings.voicePauseMs}
            title="How long the mic waits for silence before sending your words in hands-free mode. Applies to the next hands-free session."
            onChange={(e) => onSave({ voicePauseMs: Number(e.target.value) })}
          />
        </label>
        <p className="text-[10px] text-zinc-400">
          On-device speech: <ExtLink href="https://huggingface.co/kyutai/pocket-tts">pocket-tts</ExtLink>{' '}
          (<ExtLink href="https://creativecommons.org/licenses/by/4.0/">CC-BY-4.0</ExtLink>, ~
          {TTS_DOWNLOAD_MB} MB) ·{' '}
          <ExtLink href="https://huggingface.co/UsefulSensors/moonshine">Moonshine</ExtLink> (~
          {STT_DOWNLOAD_MB} MB) ·{' '}
          <ExtLink href="https://github.com/snakers4/silero-vad">Silero VAD</ExtLink>. Downloaded once.
        </p>
      </div>

      {/* Advanced */}
      <details className="group rounded-lg border border-zinc-200 dark:border-zinc-800">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2.5 py-2 text-zinc-500 marker:hidden dark:text-zinc-400">
          <ChevronDownIcon
            size={14}
            className="-rotate-90 transition-transform group-open:rotate-0"
          />
          <SectionLabel>Advanced</SectionLabel>
        </summary>
        <div className="space-y-3 border-t border-zinc-200 px-2.5 py-3 dark:border-zinc-800">
          <label className="block">
            <span className="text-zinc-500 dark:text-zinc-400">Context (tokens)</span>
            <input
              type="number"
              className={`mt-1 w-full ${inputCls}`}
              value={settings.webllmCtx}
              min={MIN_CONTEXT_TOKENS}
              max={maxCtx}
              step={MIN_CONTEXT_TOKENS}
              title={`Higher reads more of the page but reserves more GPU memory up front (capped at ${maxCtx.toLocaleString()}). Changing it reloads the model.`}
              onChange={(e) => {
                const value = Number(e.target.value);
                if (!Number.isFinite(value)) return;
                onSave({
                  webllmCtx: Math.max(MIN_CONTEXT_TOKENS, Math.min(maxCtx, value || DEFAULT_SETTINGS.webllmCtx)),
                });
              }}
            />
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
            <span className="text-zinc-500 dark:text-zinc-400">Repetition penalty</span>
            <input
              type="number"
              className={`mt-1 w-full ${inputCls}`}
              value={settings.repetitionPenalty}
              min={0}
              max={1}
              step={0.1}
              title="Raise if the model repeats itself; 0 leaves it off."
              onChange={(e) => onSave({ repetitionPenalty: Number(e.target.value) })}
            />
          </label>

          <Checkbox
            checked={settings.compressContext}
            onChange={(v) => onSave({ compressContext: v })}
            label="Compress retrieved context (experimental)"
          />
          <Checkbox
            checked={settings.viewportBoost}
            onChange={(v) => onSave({ viewportBoost: v })}
            label="Prefer on-screen content (experimental)"
          />
          <Checkbox
            checked={settings.voiceAutoSend}
            onChange={(v) => onSave({ voiceAutoSend: v })}
            label="Send push-to-talk messages automatically"
          />

          <div className="flex flex-col gap-2 pt-1">
            <button type="button" disabled={busy} onClick={onRelease} className={releaseBtnCls}>
              Release model from memory
            </button>
            {ttsLoaded && (
              <button type="button" onClick={onReleaseVoice} className={releaseBtnCls}>
                Release voice model from memory
              </button>
            )}
            {sttLoaded && (
              <button type="button" onClick={onReleaseVoiceInput} className={releaseBtnCls}>
                Release voice-input model from memory
              </button>
            )}
          </div>
        </div>
      </details>
    </div>
  );
}
