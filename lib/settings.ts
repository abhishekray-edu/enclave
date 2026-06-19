import { browser } from 'wxt/browser';
import { DEFAULT_SETTINGS, MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS, type Settings } from './types';

const KEY = 'settings';

function clampContext(ctx: number): number {
  if (!Number.isFinite(ctx)) return DEFAULT_SETTINGS.webllmCtx;
  return Math.max(MIN_CONTEXT_TOKENS, Math.min(MAX_CONTEXT_TOKENS, Math.round(ctx)));
}

function normalizeSettings(settings: Settings): Settings {
  return {
    ...settings,
    webllmCtx: clampContext(settings.webllmCtx),
    systemPrompt: DEFAULT_SETTINGS.systemPrompt,
  };
}

/** Load settings, filling any missing fields with defaults. */
export async function loadSettings(): Promise<Settings> {
  const stored = await browser.storage.local.get(KEY);
  return normalizeSettings({ ...DEFAULT_SETTINGS, ...(stored[KEY] as Partial<Settings> | undefined) });
}

/** Persist a partial settings patch and return the merged result. */
export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const current = await loadSettings();
  const next = normalizeSettings({ ...current, ...patch });
  await browser.storage.local.set({ [KEY]: next });
  return next;
}
