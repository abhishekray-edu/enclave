import type { PendingAction } from '@/lib/types';

const MENU_ID = 'explain-selection';
const PENDING_KEY = 'pendingAction';

async function setPending(action: PendingAction) {
  await browser.storage.local.set({ [PENDING_KEY]: action });
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
});
