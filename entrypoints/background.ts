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
    reasons: ['WORKERS'],
    justification: 'Runs the local AI model so it stays loaded between uses.',
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

  browser.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId !== MENU_ID || !tab?.id || tab.windowId === undefined) return;
    await setPending({ action: 'explain', selection: info.selectionText ?? '' });
    await browser.sidePanel.open({ windowId: tab.windowId });
  });

  // Keyboard shortcut (Cmd/Ctrl+Shift+L) opens the panel.
  browser.commands.onCommand.addListener(async (command) => {
    if (command !== 'open-panel') return;
    const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
    if (tab?.windowId !== undefined) {
      await browser.sidePanel.open({ windowId: tab.windowId });
    }
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
