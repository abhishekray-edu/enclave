# Local AI Page Assistant

A privacy-first Chrome side-panel extension that answers questions about the current web
page using a **local** AI model via [Ollama](https://ollama.com). Nothing leaves your
machine — the only network call the extension makes is to `localhost:11434`.

This is **v1**, the foundation for a larger local-AI browser. It does three things:
**Ask** about the page · **Summarize** the page · **Explain selection**.

## Requirements

- [Ollama](https://ollama.com) running locally with a model installed (default `gemma3:4b`).
- Node 20+ and npm (for building the extension).

## One-time setup: allow the extension to reach Ollama (CORS)

A Chrome extension has its own origin (`chrome-extension://…`), which Ollama blocks by
default. Allow it, then restart Ollama:

```bash
launchctl setenv OLLAMA_ORIGINS "chrome-extension://*"   # macOS
osascript -e 'quit app "Ollama"'; sleep 2; open -a Ollama
```

> `launchctl setenv` lasts until reboot. For a permanent setting, set `OLLAMA_ORIGINS` in
> the Ollama app's settings, or add the `launchctl setenv` line to a login script.
> For tighter security, replace `*` with your pinned extension ID once loaded.

Verify:

```bash
curl -s -i http://localhost:11434/api/tags -H "Origin: chrome-extension://x" | grep -i access-control-allow-origin
```

## Build & load

```bash
npm install
npm run build           # outputs .output/chrome-mv3
# or: npm run dev        # live-reload dev build
```

Then in Chrome: **Extensions → Manage Extensions → enable Developer mode →
Load unpacked → select `.output/chrome-mv3`** (use `.output/chrome-mv3-dev` for `dev`).

## Use

- Click the toolbar icon or press **⌘⇧L** (Ctrl+Shift+L) to open the side panel.
- Type a question, or use **Summarize** / **Explain selection**.
- Right-click selected text → **Explain selection with local AI**.
- The **⚙** panel switches model (gemma3:4b, gemma4:e4b, qwen3.5:9b, …), context size
  (`num_ctx`), temperature, and the system prompt.

## Architecture

```
content script (Readability) ──▶ side panel (React, owns the streaming fetch) ──▶ Ollama
        page text + selection            chat UI · prompt builder · model picker      /api/chat
background worker: opens panel (toolbar / ⌘⇧L / context menu)
```

- **`entrypoints/content.ts`** — extracts clean page text with Mozilla Readability
  (plain-text fallback for non-articles) plus the current selection.
- **`entrypoints/sidepanel/App.tsx`** — UI + the streaming loop (runs here, not in the
  service worker, so MV3's 30s worker timeout can't sever a response).
- **`entrypoints/background.ts`** — thin glue: panel behavior, keyboard command, context menu.
- **`lib/ollama.ts`** — `/api/chat` NDJSON streaming + abort; typed connection error.
- **`lib/prompt.ts`** — builds the system+page+conversation messages and truncates page
  text to fit `num_ctx` (gemma3:4b supports 128K, but prompt-eval gets slow on Apple
  Silicon as the window grows; default 32K ≈ 24K words. The UI shows when a page was
  trimmed). Truly long docs are what the v3 chunking/RAG step is for.
- **`lib/settings.ts` / `lib/types.ts`** — settings persisted in `chrome.storage.local`.

## Roadmap

- **v2:** persistent per-site chat history, PDF support, vision (screenshots → gemma3:4b).
- **v3:** RAG over multiple pages / notes (Ollama embeddings) — the start of the notes pillar.
```
