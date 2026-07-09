<div align="center">

<img src="public/logo.svg" width="96" height="96" alt="Enclave logo" />

# Enclave

**Run LLMs locally in your browser, on your own GPU. Nothing ever leaves your machine.**

Ask about the page you're on, summarize it, explain a selection, or extract structured data —
no accounts, no servers, no setup.

[![CI](https://github.com/abhishekray-edu/enclave/actions/workflows/ci.yml/badge.svg)](https://github.com/abhishekray-edu/enclave/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/abhishekray-edu/enclave)](https://github.com/abhishekray-edu/enclave/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## Why Enclave

Enclave runs the model on your own hardware, inside the browser via WebGPU — your browsing and
your questions stay on your machine, with nothing to install or configure.

It's a Chrome **side-panel** extension: open it beside any tab and chat about what's on screen.

<!-- TODO: screenshot / demo GIF here -->

## Features

- **Fully local & private** — inference runs in your browser on WebGPU; page content and
  questions never leave the device.
- **Zero setup** — no server, no account, no API key. Install the extension and ask.
- **Pick your model by RAM** — six options from a 1B lightweight to an 8B powerhouse.
- **Reads the real page** — robust extraction across SPAs, shadow DOM, and same-origin iframes.
- **Long pages done right** — on-device retrieval (RAG): pages are chunked, embedded, and the
  most relevant sections retrieved, with **cited sources** you can click to scroll to on the page.
  Whole-page summaries **map-reduce** over every section instead of truncating.
- **Structured extraction** — pull article metadata, key facts, or contacts as JSON, with
  schema-constrained decoding (valid JSON by construction).
- **Quick actions** — Ask, Summarize, Explain a selection (also via right-click), Extract.
- **Fast when it matters** — the page is indexed and the model staged as soon as the panel
  opens, and the model stays resident across panel open/close for instant follow-ups.
- **Clean answers** — Markdown with syntax-highlighted, copyable code blocks, in a neutral
  light/dark theme that follows your system.

## Models

Choose a model in the header (or **⚙ → Model**) based on your machine's memory. Each downloads
once on first use (cached afterwards), and you can switch anytime.

| Model | ~Memory | Best for |
|---|---|---|
| Llama 3.2 1B | ~1 GB | Lightest & fastest |
| Gemma 2 2B | ~1.9 GB | Light |
| Llama 3.2 3B | ~2.3 GB | Balanced |
| **Qwen3 4B** | ~3.4 GB | **Recommended (default)** |
| Llama 3.1 8B | ~5 GB | High quality |
| Qwen3 8B | ~5.7 GB | Best, heaviest |

Larger models give better answers but need more GPU memory. The **context window** is
adjustable in ⚙ and capped conservatively for stability.

## Privacy

- Your page content, selections, and questions are processed **entirely on-device** and are
  never sent anywhere.
- The only network traffic is the **one-time download of model weights** (from the WebLLM /
  Hugging Face CDNs), which are then cached locally.
- No telemetry, no analytics, no accounts.

## Requirements

- A Chromium browser with **WebGPU** (recent Chrome, Edge, Brave, Arc…).
- A reasonably capable machine — more RAM/GPU lets you run the larger models.

## Install (no build)

Grab the latest packaged build from the
[**Releases**](https://github.com/abhishekray-edu/enclave/releases) page — no Node or build step required.

1. Download `enclave-<version>-chrome.zip` and **unzip** it.
2. Go to `chrome://extensions` and enable **Developer mode** (top right).
3. **Load unpacked** → select the unzipped folder.
4. Click the Enclave icon (or press **⌘⇧L** / **Ctrl+Shift+L**) to open the side panel.
5. Ask a question. The first time, the model downloads with a progress bar; after that it's cached.

> Firefox users: download `enclave-<version>-firefox.zip` instead and load it via
> `about:debugging` → **This Firefox** → **Load Temporary Add-on**.

Enclave isn't on the Chrome Web Store yet, so it installs as an unpacked extension — that's expected.

## Build from source

Prefer to build it yourself, or want live reload for development? You'll need **Node 20+**.

```bash
git clone https://github.com/abhishekray-edu/enclave.git
cd enclave
npm install
npm run build        # outputs .output/chrome-mv3   (use `npm run dev` for live reload)
```

Then **Load unpacked** → select the `.output/chrome-mv3` folder, as above.

## Usage

- **⌘⇧L / Ctrl+Shift+L** or the toolbar icon — open the side panel.
- **Ask** — type any question about the current page. On long pages, answers cite the sections
  they drew from; click a source to scroll to it.
- **Summarize** — one-click page summary; long pages are summarized section-by-section, then merged.
- **Explain selection** — select text, then use the button or right-click → *Explain selection with local AI*.
- **Extract…** — pull structured JSON (article metadata, key facts, contacts) out of the page.
- **⚙** — switch model, context window, theme, temperature, and **Release model from memory**.

## How it handles large pages

A small local model can't take a 10,000-word page in one prompt — and on an integrated GPU an
oversized prompt isn't just slow, it can starve the OS compositor. So Enclave never stuffs:

```
clean → chunk → embed (MiniLM, cached in IndexedDB) → retrieve top-k → generate (+ cite sources)
```

Whole-page summaries map-reduce over all chunks, structured extraction uses grammar-constrained
JSON decoding, and every single prompt is bounded by a per-model safety cap that is pinned by
tests. The research notes and the post-mortem behind those caps live in
[docs/large-page-handling.md](docs/large-page-handling.md).

## Architecture

```
  Browser tab                      Side panel (React)                Offscreen document
 ┌──────────────┐  page text +   ┌──────────────────────┐   Port   ┌───────────────────────────────┐
 │ content      │─ structure + ─▶│ chat UI · task router │─────────▶│ webllm.worker — LLM on WebGPU │
 │ script       │  selection     │ prompt builder (RAG)  │◀─tokens──│ ml.worker — embed · retrieve  │
 └──────────────┘                └──────────────────────┘          │             · compress        │
                                                                   └───────────────────────────────┘
  Background worker: opens the panel · context menu · offscreen-document lifecycle
  (the offscreen document keeps models resident across panel open/close)
```

- **`entrypoints/content.ts`** — extracts clean page text (Mozilla Readability, plus a deep DOM
  walk that crosses shadow DOM and same-origin iframes) with document structure and the selection.
- **`entrypoints/sidepanel/`** — the React UI: chat, task routing, prompt building, markdown rendering.
- **`entrypoints/offscreen/`** — hosts the models in a persistent hidden document so they stay
  loaded across panel open/close. The LLM and the embedding/compression models each run in their
  own dedicated worker, so the UI never blocks on model work.
- **`entrypoints/background.ts`** — thin glue: panel behavior, keyboard command, context menu,
  and offscreen-document lifecycle.
- **`lib/`** — the pipeline: `prompt.ts` (context-budgeted prompt builder), `chunking.ts`
  (structure-aware chunking), `retrieval.ts` (MiniLM embeddings + IndexedDB vector cache),
  `summarize.ts` (hierarchical map-reduce), `tasks.ts` (task specs + extraction schemas),
  `compress.ts` (optional LLMLingua-2 compression), `webllm.ts` / `webllmClient.ts` (engine and
  its panel-side client + model catalog with per-model safety caps), `ortEnv.ts`, `settings.ts`,
  `theme.ts`, `types.ts`.

The heavy ML runtimes (WebLLM, Transformers.js) are isolated to the offscreen bundle, keeping
the side-panel bundle small.

The reasoning behind the core design is in a single
[Architecture Decision Record](docs/adr/0001-architecture-and-key-decisions.md); the large-page
pipeline has its own [design notes](docs/large-page-handling.md).

## Development

```bash
npm run dev        # live-reload dev build (Chrome)
npm run compile    # typecheck
npm test           # unit tests (vitest)
npm run zip        # package a release zip
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © Abhishek Ray
