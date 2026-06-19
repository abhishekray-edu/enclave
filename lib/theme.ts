import type { Theme } from './types';

let mql: MediaQueryList | null = null;
let listener: ((e: MediaQueryListEvent) => void) | null = null;

/**
 * Apply a theme by toggling the `dark` class on <html>. For 'system' it follows
 * the OS preference and keeps following it (live) until the theme changes again.
 */
export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  const setDark = (on: boolean) => root.classList.toggle('dark', on);

  // Tear down any previous system listener.
  if (mql && listener) mql.removeEventListener('change', listener);
  mql = null;
  listener = null;

  if (theme === 'system') {
    mql = window.matchMedia('(prefers-color-scheme: dark)');
    setDark(mql.matches);
    listener = (e) => setDark(e.matches);
    mql.addEventListener('change', listener);
  } else {
    setDark(theme === 'dark');
  }
}
