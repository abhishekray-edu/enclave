import { browser } from 'wxt/browser';
import { DEFAULT_SETTINGS, type Settings } from './types';

const KEY = 'settings';

/** Load settings, filling any missing fields with defaults. */
export async function loadSettings(): Promise<Settings> {
  const stored = await browser.storage.local.get(KEY);
  return { ...DEFAULT_SETTINGS, ...(stored[KEY] as Partial<Settings> | undefined) };
}

/** Persist a partial settings patch and return the merged result. */
export async function saveSettings(patch: Partial<Settings>): Promise<Settings> {
  const current = await loadSettings();
  const next = { ...current, ...patch };
  await browser.storage.local.set({ [KEY]: next });
  return next;
}
