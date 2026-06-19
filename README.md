<div align="center">

<img src="public/logo.svg" width="96" height="96" alt="Enclave logo" />

# Enclave

**Run LLMs locally in your browser, on your own GPU. Nothing ever leaves your machine.**

Ask about the page you're on, summarize it, or explain a selection — no accounts, no servers, no setup.

</div>

---

## Why Enclave

Enclave runs the model on your own hardware, inside the browser via WebGPU — your browsing and
your questions stay on your machine, with nothing to install or configure.

It's a Chrome **side-panel** extension: open it beside any tab and chat about what's on screen.

## Features

- **Fully local & private** — the model runs in your browser; your data never leaves the device.
- **Zero setup** — no server, no account, no API key. Install the extension and ask.
- **Pick your model by RAM** — six options from a 1B lightweight to an 8B powerhouse.
- **Reads the real page** — robust extraction across SPAs, shadow DOM, and same-origin iframes.
- **Quick actions** — Ask about the page, Summarize, or Explain a selection (also via right-click).
- **Clean answers** — Markdown with syntax-highlighted, copyable code blocks.
- **Polished UI** — neutral light/dark theme that follows your system.
- **Stays warm** — the model is kept resident in the background, so follow-up questions are instant.

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

## Requirements

- A Chromium browser with **WebGPU** (recent Chrome, Edge, Brave, Arc…).
- A reasonably capable machine — more RAM/GPU lets you run the larger models.
- Node 20+ and npm to build the extension.

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

Prefer to build it yourself, or want live reload for development?

```bash
git clone https://github.com/abhishekray-edu/enclave.git
cd enclave
npm install
npm run build        # outputs .output/chrome-mv3   (use `npm run dev` for live reload)
```

Then **Load unpacked** → select the `.output/chrome-mv3` folder, as above.

That's it — no other software to install.

## Usage

- **⌘⇧L / Ctrl+Shift+L** or the toolbar icon — open the side panel.
- **Ask** — type any question about the current page.
- **Summarize** — one-click page summary.
- **Explain selection** — select text, then use the button or right-click → *Explain selection with local AI*.
- **⚙** — switch model, context window, theme, temperature, and **Release model from memory**.

## Architecture

```
  Browser tab                         Side panel (React)              Offscreen document
 ┌──────────────┐   page text/      ┌─────────────────────┐  Port   ┌────────────────────────┐
 │ content      │── selection ─────▶│  chat UI · prompt    │───────▶ │ WebLLM engine on WebGPU │
 │ script       │  (Readability +   │  builder · markdown  │         │ stays resident in memory│
 │              │   deep DOM walk)   │  renderer            │ ◀tokens─└────────────────────────┘
 └──────────────┘                   └─────────────────────┘
   Background worker: opens the panel · context menu · offscreen-document lifecycle
```

- **`entrypoints/content.ts`** — extracts clean page text (Mozilla Readability, with a deep DOM
  walk that crosses shadow DOM and same-origin iframes) plus the current selection.
- **`entrypoints/sidepanel/`** — the React UI: chat, prompt building, markdown rendering.
- **`entrypoints/offscreen/`** — hosts the WebLLM engine in a persistent hidden document so the
  model stays loaded across panel open/close. The panel talks to it over a Port.
- **`entrypoints/background.ts`** — thin glue: panel behavior, keyboard command, context menu,
  and offscreen-document lifecycle.
- **`lib/`** — `webllm.ts` (engine, offscreen-only), `webllmClient.ts` (panel port client +
  model catalog), `prompt.ts` (context-budgeted prompt builder), `settings.ts`, `theme.ts`, `types.ts`.

The heavy WebLLM runtime is isolated to the offscreen bundle, keeping the side-panel bundle small.

The reasoning behind the core design is in a single
[Architecture Decision Record](docs/adr/0001-architecture-and-key-decisions.md).

## Privacy

All processing happens on your own hardware, inside the browser. Page content and your
questions go only to the local model. The engine downloads model weights once (from the
WebLLM/Hugging Face CDN) and caches them locally. Enclave ships with no analytics or telemetry.

## Tech stack

[WXT](https://wxt.dev) · React · TypeScript · Tailwind CSS v4 ·
[@mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) (WebGPU) ·
[@mozilla/readability](https://github.com/mozilla/readability) · react-markdown · highlight.js

## Development

```bash
npm run dev        # live-reload dev build (.output/chrome-mv3-dev)
npm run build      # production build
npm run compile    # type-check (tsc --noEmit)
npm run icons      # regenerate PNG icons from public/logo.svg
npm run zip        # package for distribution
```

## Roadmap

- **v2** — persistent per-site chat history, PDF support, vision (screenshots → multimodal models).
- **v3** — retrieval (RAG) over multiple pages and notes; the start of the broader private-workspace vision.

## License

MIT — see [LICENSE](LICENSE).
