import type { PendingAction } from '@/lib/types';

const MENU_ID = 'explain-selection';
const PENDING_KEY = 'pendingAction';
const OFFSCREEN_URL = 'offscreen.html';

async function setPending(action: PendingAction) {
  await browser.storage.local.set({ [PENDING_KEY]: action });
}

// chrome.offscreen isn't in the polyfill types; access the global directly.
const offscreen = (globalThis as unknown as {
  chrome: {
    offscreen: {
      hasDocument(): Promise<boolean>;
      createDocument(opts: { url: string; reasons: string[]; justification: string }): Promise<void>;
      closeDocument(): Promise<void>;
    };
  };
}).chrome.offscreen;

/** Create the offscreen document (which hosts the in-browser model) if absent. */
async function ensureOffscreen() {
  if (await offscreen.hasDocument()) return;
  await offscreen.createDocument({
    url: OFFSCREEN_URL,
    // WORKERS: hosts the WebLLM engine + ML/TTS workers. AUDIO_PLAYBACK: lets the document
    // play synthesized speech (pocket-tts) without a foreground user gesture.
    reasons: ['WORKERS', 'AUDIO_PLAYBACK'],
    justification: 'Runs the local AI model so it stays loaded between uses, and plays synthesized speech.',
  });
}

/** Close the offscreen document, releasing the model from memory. */
async function releaseOffscreen() {
  if (await offscreen.hasDocument()) await offscreen.closeDocument();
}

export default defineBackground(() => {
  // Clicking the toolbar icon opens the side panel.
  browser.sidePanel
    ?.setPanelBehavior({ openPanelOnActionClick: true })
    .catch((err) => console.error('setPanelBehavior failed', err));

  // Right-click context menu on a text selection.
  browser.runtime.onInstalled.addListener(() => {
    browser.contextMenus.create({
      id: MENU_ID,
      title: 'Explain selection with local AI',
      contexts: ['selection'],
    });
  });

  browser.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId !== MENU_ID || !tab?.id || tab.windowId === undefined) return;
    // sidePanel.open() only works while the user gesture (the context-menu click) is still
    // active, so it must be called synchronously here — awaiting anything first (e.g. the
    // storage write) drops the gesture and Chrome rejects the open. Fire the pending-action
    // write without awaiting; it completes long before the panel's bundle loads and reads it.
    void setPending({ action: 'explain', selection: info.selectionText ?? '' });
    browser.sidePanel.open({ windowId: tab.windowId }).catch((err) => console.error('sidePanel.open failed', err));
  });

  // Keyboard shortcut (Cmd/Ctrl+Shift+L) opens the panel. Use the tab passed to the listener
  // rather than querying for it: awaiting a query first would drop the user gesture and Chrome
  // would reject sidePanel.open().
  browser.commands.onCommand.addListener((command, tab) => {
    if (command !== 'open-panel' || tab?.windowId === undefined) return;
    browser.sidePanel.open({ windowId: tab.windowId }).catch((err) => console.error('sidePanel.open failed', err));
  });

  // Offscreen document lifecycle (hosts the in-browser WebLLM engine).
  browser.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.type === 'ENSURE_OFFSCREEN') {
      ensureOffscreen().then(
        () => sendResponse({ ok: true }),
        (e) => sendResponse({ ok: false, error: String(e) }),
      );
      return true; // async response
    }
    if (msg?.type === 'RELEASE_OFFSCREEN') {
      releaseOffscreen().then(
        () => sendResponse({ ok: true }),
        (e) => sendResponse({ ok: false, error: String(e) }),
      );
      return true;
    }
  });
});
