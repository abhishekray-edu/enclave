<div align="center">

<img src="public/logo.svg" width="96" height="96" alt="Enclave logo" />

# Enclave

**A private, on-device AI that reads the page you're on — and answers.**

Ask questions about any web page, summarize it, or explain a selection — powered by a
local model. No accounts, no servers, no telemetry. Nothing ever leaves your machine.

</div>

---

## Why Enclave

Enclave runs the model **on your own hardware** — in the browser via WebGPU, or through a
local [Ollama](https://ollama.com) server — so your browsing and your questions stay private
to your machine.

It's a Chrome **side-panel** extension: open it next to any tab and chat about what's on screen.

## Features

- 🔒 **Fully local & private** — the only network calls are to your own machine.
- 🧠 **Two engines, your choice** — zero-setup in-browser (WebGPU) *or* native Ollama for bigger, faster models.
- 📄 **Reads the real page** — robust extraction across SPAs, shadow DOM, and same-origin iframes (not just article text).
- 💬 **Quick actions** — Ask about the page, Summarize, or Explain a selection (also via right-click).
- ✨ **Clean answers** — Markdown rendering with syntax-highlighted, copyable code blocks.
- 🌓 **Polished UI** — neutral light/dark theme that follows your system.
- ⚡ **Stays warm** — the in-browser model persists in the background, so follow-up questions are instant.

## Two ways to run

| | **In-browser** (WebGPU) | **Ollama** (native) |
|---|---|---|
| Setup | **None** — just install the extension | Install Ollama + one config step |
| Model | Qwen3 4B (default), Gemma 2, Llama 3.1 8B… | Any Ollama model (default `gemma3:4b`) |
| First use | One-time ~2.5 GB model download (auto, cached) | You `ollama pull` the model |
| Speed | ~80% of native; great on Apple Silicon | Full native speed |
| Best for | Trying it instantly, no dependencies | Daily driver, larger models, lower memory |

Switch anytime in **⚙ → Engine**.

## Requirements

- A Chromium browser with WebGPU (Chrome/Edge/Brave, recent versions) for the in-browser engine.
- Node 20+ and npm to build the extension.
- *(Optional, for the Ollama engine)* [Ollama](https://ollama.com) with a model installed.

## Getting started

```bash
git clone https://github.com/abhishekray-edu/enclave.git
cd enclave
npm install
npm run build        # outputs .output/chrome-mv3   (use `npm run dev` for live reload)
```

Load it in Chrome:

1. Go to `chrome://extensions` and enable **Developer mode**.
2. **Load unpacked** → select the `.output/chrome-mv3` folder.
3. Click the Enclave icon (or press **⌘⇧L** / **Ctrl+Shift+L**) to open the side panel.

### Using the in-browser engine (no setup)

Open **⚙ → Engine → In-browser**, then ask a question. The first time, the model downloads
(~2.5 GB) with a progress bar and is cached afterwards. It stays loaded in the background
(so follow-up questions are instant); free it anytime with **⚙ → Release model from memory**.

The context window defaults to the model's maximum (40,960 tokens for Qwen3 4B) and is
adjustable in **⚙ → Context** — lower it if your machine is memory-constrained. Changing the
model or context reloads the engine automatically.

### Using the Ollama engine

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull gemma3:4b
   ```
2. Allow the extension to reach Ollama (a one-time CORS step), then restart Ollama:
   ```bash
   launchctl setenv OLLAMA_ORIGINS "chrome-extension://*"   # macOS
   osascript -e 'quit app "Ollama"'; sleep 2; open -a Ollama
   ```
   > Tip: for a permanent setting, set `OLLAMA_ORIGINS` in Ollama's app settings. You can
   > also point Enclave at Ollama running on **another machine** (e.g. a home server) by
   > changing the endpoint in ⚙ — it stays private to your network.
3. In Enclave, choose **⚙ → Engine → Ollama** and pick your model.

## Usage

- **⌘⇧L / Ctrl+Shift+L** or the toolbar icon — open the side panel.
- **Ask** — type any question about the current page.
- **Summarize** — one-click page summary.
- **Explain selection** — select text, then use the button or right-click → *Explain selection with local AI*.
- **⚙** — switch engine/model/theme, system prompt, temperature, and (Ollama) context size.

## Architecture

```
  Browser tab                         Side panel (React)                Engines
 ┌──────────────┐   page text/      ┌─────────────────────┐   stream   ┌───────────────────────┐
 │ content      │── selection ─────▶│  chat UI · prompt    │──────────▶ │ Ollama  localhost:11434│
 │ script       │  (Readability +   │  builder · markdown  │            └───────────────────────┘
 │              │   deep DOM walk)   │  renderer            │   port     ┌───────────────────────┐
 └──────────────┘                   └─────────┬───────────┘──────────▶ │ Offscreen document     │
                                               │                        │  WebLLM engine (WebGPU)│
 Background worker: opens panel, manages the offscreen document         │  stays resident in RAM │
                                                                        └───────────────────────┘
```

- **`entrypoints/content.ts`** — extracts clean page text (Mozilla Readability, with a
  deep DOM walk that crosses shadow DOM and same-origin iframes) plus the current selection.
- **`entrypoints/sidepanel/`** — the React UI; owns the chat, prompt building, and markdown rendering.
- **`entrypoints/offscreen/`** — hosts the in-browser WebLLM engine in a persistent hidden
  document so the model stays loaded across panel open/close. The panel talks to it over a Port.
- **`entrypoints/background.ts`** — thin glue: panel behavior, keyboard command, context
  menu, and offscreen-document lifecycle.
- **`lib/`** — `ollama.ts` (native streaming), `webllm.ts` (engine, offscreen-only),
  `webllmClient.ts` (panel port client), `prompt.ts` (context-budgeted prompt builder),
  `settings.ts`, `theme.ts`, `types.ts`.

The heavy WebLLM runtime is isolated to the offscreen bundle, keeping the side-panel bundle small.

The reasoning behind the core design — extension vs. fork, the two engines, the offscreen
document, page extraction, and how it all connects — is captured in a single
[Architecture Decision Record](docs/adr/0001-architecture-and-key-decisions.md).

## Privacy

All processing happens on your own hardware. Page content and your questions go only to a
model running locally — in the browser, or to Ollama on `localhost`. The in-browser engine
downloads model weights once (from the WebLLM/Hugging Face CDN) and caches them locally.
Enclave ships with no analytics or telemetry.

## Tech stack

[WXT](https://wxt.dev) · React · TypeScript · Tailwind CSS v4 ·
[@mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) (WebGPU) ·
[Ollama](https://ollama.com) · [@mozilla/readability](https://github.com/mozilla/readability) ·
react-markdown · highlight.js

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

MIT
