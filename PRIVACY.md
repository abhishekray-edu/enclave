# Privacy Policy

**Last updated:** 2026-07-10

Enclave is a browser extension that runs a large language model **entirely on your
device** so you can ask questions about the page you're viewing. This policy explains
what the extension does and does not do with your data.

## Summary

Enclave does not collect, transmit, or sell any of your data. Everything it does happens
locally in your browser.

## What Enclave accesses, and why

| Data | Why it's accessed | Where it goes |
|---|---|---|
| Active tab's page content (text, selections) | So the local model can answer questions about the page | Stays on-device; sent only to the local model running in your browser |
| Your questions and the model's answers | Core chat functionality | Stays on-device |
| Model preferences, context-window size, cached page text | So the extension remembers your settings and doesn't reprocess a page you've already visited | Stored locally via the browser's `storage` API; never leaves your machine |

Enclave never sends page content, selections, questions, or answers to any server. There
is no backend — the extension has nothing to send data to even if it wanted to.

## Network requests

The only network traffic Enclave makes is the **one-time download of model weights**
(from Hugging Face / WebLLM's public CDNs) the first time you select a model. This
downloads the model file to your machine, where it's cached by the browser for reuse.
No page content, questions, or personal information is included in these requests.

## No telemetry, analytics, or accounts

Enclave does not use analytics, crash reporting, or telemetry of any kind, and does not
require or support user accounts.

## Permissions

Enclave requests the following browser permissions, used only for the purposes below,
and only when you actively use the extension (opening the side panel, using the
right-click menu, or pressing the keyboard shortcut):

- **activeTab / scripting** — to read the content of the page you're currently viewing,
  only when you invoke the extension.
- **tabs** — to identify the active tab so the side panel is scoped to the page you're
  looking at.
- **sidePanel** — to display the Enclave chat UI.
- **contextMenus** — to add the right-click "Ask Enclave" / "Explain selection" actions.
- **storage** — to save your settings and cache processed page text locally.
- **offscreen** — to run the local model (WebGPU/WASM inference) outside the panel's
  lifecycle.

None of these permissions are used to monitor browsing activity in the background or on
tabs other than the one you're actively working with.

## Changes to this policy

If this policy changes, the update will be reflected here with a new "Last updated"
date.

## Contact

Questions about this policy or the extension can be filed as an issue at
https://github.com/abhishekray-edu/enclave/issues.
