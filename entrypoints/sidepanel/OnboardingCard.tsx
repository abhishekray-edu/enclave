// First-run setup: pick a model and download it explicitly (nothing downloads until the button
// says so). An opt-in "Enable voice chat" section — checked by default — downloads the speech
// models alongside the LLM and requests the mic, so the record button works instantly afterward.
import { WEBLLM_MODELS, webllmModel, type LoadProgress } from '@/lib/webllmClient';
import { TTS_DOWNLOAD_MB } from '@/lib/ttsClient';
import { STT_DOWNLOAD_MB } from '@/lib/sttClient';
import type { Settings } from '@/lib/types';
import { HexDraw } from './hex';
import { CheckIcon } from './icons';

const VOICE_TOTAL_MB = STT_DOWNLOAD_MB + TTS_DOWNLOAD_MB;

/** Download lifecycle of one voice model, surfaced as a row in the card. */
export interface VoiceItemState {
  downloading: boolean;
  pct: number;
  ready: boolean;
  error: string | null;
}

function VoiceRow({
  label,
  mb,
  state,
  onRetry,
}: {
  label: string;
  mb: number;
  state: VoiceItemState;
  onRetry: () => void;
}) {
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="flex-1 text-zinc-600 dark:text-zinc-300">{label}</span>
      {state.ready ? (
        <CheckIcon size={13} className="text-emerald-600 dark:text-emerald-400" />
      ) : state.error ? (
        <button
          type="button"
          onClick={onRetry}
          className="rounded border border-amber-300 px-1.5 py-px text-[10px] text-amber-700 hover:bg-amber-50 dark:border-amber-600/60 dark:text-amber-300 dark:hover:bg-amber-950/30"
        >
          Retry
        </button>
      ) : state.downloading ? (
        <span className="w-16 tabular-nums text-right text-zinc-400">{state.pct}%</span>
      ) : (
        <span className="tabular-nums text-zinc-400">~{mb} MB</span>
      )}
    </div>
  );
}

export function OnboardingCard({
  settings,
  suggestedId,
  pausedModels,
  busy,
  progress,
  error,
  voiceEnabled,
  stt,
  tts,
  micNeeded,
  onSelect,
  onDownload,
  onToggleVoice,
  onRetryStt,
  onRetryTts,
  onAllowMic,
}: {
  settings: Settings;
  suggestedId: string;
  pausedModels: string[];
  busy: boolean;
  progress: LoadProgress | null;
  error: string | null;
  voiceEnabled: boolean;
  stt: VoiceItemState;
  tts: VoiceItemState;
  micNeeded: boolean;
  onSelect: (id: string) => void;
  onDownload: () => void;
  onToggleVoice: (enabled: boolean) => void;
  onRetryStt: () => void;
  onRetryTts: () => void;
  onAllowMic: () => void;
}) {
  const current = webllmModel(settings.webllmModel);
  const pct = Math.round((progress?.progress ?? 0) * 100);
  // The selected model has a partial download to pick up where it left off.
  const resumable = pausedModels.includes(settings.webllmModel);
  const voiceAttempted = stt.downloading || tts.downloading || stt.ready || tts.ready || !!stt.error || !!tts.error;
  const downloadLabel = busy
    ? `Downloading… ${pct}%`
    : `${resumable ? 'Resume download' : 'Download'} ${current.label} · ~${current.approxGb} GB${
        voiceEnabled ? ` + voice · ~${VOICE_TOTAL_MB} MB` : ''
      }`;

  return (
    <div className="onboard mx-auto mt-4 max-w-[21rem] rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-800">
      <div className="flex flex-col items-center text-center">
        <HexDraw />
        <h2 className="mt-2 text-sm font-semibold tracking-tight">Pick your model</h2>
      </div>

      <fieldset className="mt-3 space-y-1.5" disabled={busy}>
        <legend className="sr-only">Model</legend>
        {WEBLLM_MODELS.map((m) => {
          const selected = m.id === settings.webllmModel;
          return (
            <label
              key={m.id}
              className={
                'flex cursor-pointer items-center gap-2 rounded-lg border px-2.5 py-1.5 transition-colors ' +
                (selected
                  ? 'border-zinc-800 bg-zinc-50 dark:border-zinc-300 dark:bg-zinc-900/40'
                  : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-700/40')
              }
            >
              <input
                type="radio"
                name="onboard-model"
                className="peer sr-only"
                checked={selected}
                onChange={() => onSelect(m.id)}
              />
              <span
                aria-hidden="true"
                className={
                  'h-3 w-3 shrink-0 rounded-full border-2 peer-focus-visible:ring-2 peer-focus-visible:ring-zinc-400 ' +
                  (selected
                    ? 'border-zinc-800 bg-zinc-800 dark:border-zinc-200 dark:bg-zinc-200'
                    : 'border-zinc-300 dark:border-zinc-600')
                }
              />
              <span className="flex min-w-0 flex-1 items-baseline gap-1.5 text-xs">
                <span className="font-medium">{m.label}</span>
                <span className="truncate text-[10px] text-zinc-400">{m.note}</span>
                {m.id === suggestedId && (
                  <span className="shrink-0 rounded-full bg-emerald-50 px-1.5 py-px text-[9px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
                    Suggested
                  </span>
                )}
                {pausedModels.includes(m.id) && !selected && (
                  <span className="shrink-0 rounded-full bg-amber-50 px-1.5 py-px text-[9px] font-medium text-amber-700 dark:bg-amber-950/50 dark:text-amber-300">
                    Paused
                  </span>
                )}
              </span>
              <span className="shrink-0 text-[10px] tabular-nums text-zinc-400">~{m.approxGb} GB</span>
            </label>
          );
        })}
      </fieldset>

      {/* Voice opt-in */}
      <div className="mt-3 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-700">
        <label className="flex items-start gap-2">
          <input
            type="checkbox"
            className="mt-0.5"
            checked={voiceEnabled}
            disabled={busy || voiceAttempted}
            onChange={(e) => onToggleVoice(e.target.checked)}
          />
          <span className="text-xs">
            <span className="font-medium">Enable voice chat</span>
            <span className="mt-0.5 block text-[10px] text-zinc-400">
              Talk to Enclave and hear replies — adds ~{VOICE_TOTAL_MB} MB of speech models.
            </span>
          </span>
        </label>
        {voiceEnabled && (
          <div className="mt-2 space-y-1">
            <VoiceRow label="Speech recognition" mb={STT_DOWNLOAD_MB} state={stt} onRetry={onRetryStt} />
            <VoiceRow label="Voice" mb={TTS_DOWNLOAD_MB} state={tts} onRetry={onRetryTts} />
            {micNeeded && (
              <button
                type="button"
                onClick={onAllowMic}
                className="mt-1 w-full rounded border border-indigo-300 px-2 py-1 text-[11px] text-indigo-700 hover:bg-indigo-50 dark:border-indigo-600/60 dark:text-indigo-300 dark:hover:bg-indigo-950/30"
              >
                Allow microphone
              </button>
            )}
          </div>
        )}
      </div>

      <button
        onClick={onDownload}
        disabled={busy}
        className={
          'relative mt-3 w-full overflow-hidden rounded-lg px-3 py-2 text-sm font-medium ' +
          (busy
            ? 'cursor-default bg-zinc-100 text-zinc-800 dark:bg-zinc-700 dark:text-zinc-100'
            : 'bg-zinc-900 text-white hover:bg-zinc-800 dark:bg-zinc-200 dark:text-zinc-900 dark:hover:bg-zinc-300')
        }
      >
        {/* The button is the progress bar: an emerald fill sweeps it left to right — the
            same green as the "Ready to chat" badge it resolves into. */}
        {busy && (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 left-0 bg-emerald-100 transition-[width] duration-300 dark:bg-emerald-800"
            style={{ width: `${pct}%` }}
          />
        )}
        <span className="relative">{downloadLabel}</span>
      </button>

      {error && !busy && (
        <p className="mt-1.5 text-center text-[11px] text-amber-700 dark:text-amber-400">{error}</p>
      )}
      <p className="mt-2 text-center text-[11px] text-zinc-500 dark:text-zinc-400">
        Download once. Run on your GPU. Forever free.
      </p>
    </div>
  );
}
