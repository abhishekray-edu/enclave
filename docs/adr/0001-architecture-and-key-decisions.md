# ADR 0001 — In-browser (WebGPU) architecture

- **Status:** Accepted
- **Date:** 2026-06-19

Enclave runs a model **entirely in the browser, on the user's GPU** — private, on-device, and
installable with nothing else to set up. This record describes that design and how it works.

## Goal

A private page assistant a new user runs with zero setup: install the extension, ask a
question, and a local model answers — on their own hardware.

## Design

**Form factor.** A Chrome side-panel extension built on [WXT](https://wxt.dev) with React +
TypeScript + Tailwind, reusing established libraries: [WebLLM](https://github.com/mlc-ai/web-llm)
for in-browser inference, Mozilla Readability for content extraction, and react-markdown +
highlight.js for rendering.

**Model choice.** A small RAM-tiered menu (Llama 3.2 1B → Qwen3 8B, default Qwen3 4B) lets the
user pick a model that fits their machine; each downloads once and is cached. The context
window is clamped per model to its build's maximum.

**In-browser inference (WebGPU via WebLLM).** The selected model (default **Qwen3 4B**, 4-bit)
runs in the browser with GPU acceleration. On first use its weights download once (~1–6 GB
depending on the model) into the browser cache and are reused afterwards. Inference runs at
~80% of native speed on Apple Silicon. The context window is configurable, defaults to the
model build's maximum, and is clamped per model (e.g. Gemma 2 = 4096, Qwen3 = 40960). The KV
cache is pre-allocated for the configured window at load time, so it reserves GPU memory up
front regardless of how much is used; lower it on memory-constrained machines. WebGPU and a
CSP allowing `wasm-unsafe-eval` (for the WebAssembly runtime) are required.

**The model lives in an offscreen document — the key piece.** The side panel is recreated each
time it opens, so the WebLLM engine runs in a persistent **offscreen document** that stays
resident for the browser session. This keeps the model loaded across panel open/close, so
follow-up questions are instant.

- **Port protocol:** the panel is a thin client that talks to the offscreen document over a
  `runtime.Port` — `init` to load/confirm the model, `generate` to run it; the offscreen doc
  streams `progress`, then token `chunk`s, then `done`, and honors `interrupt`.
- **Lifecycle:** the background worker creates the offscreen document on demand. Changing the
  model or context window reloads the engine automatically (debounced), so it's ready before
  the next message. A **"Release model from memory"** control closes the document to reclaim
  RAM/VRAM; it reloads on the next question.
- **Bundle split:** the heavy WebLLM runtime is imported only by the offscreen document, so the
  side-panel bundle stays small (~0.5 MB).

**Reasoning models.** When a hybrid reasoning model (e.g. Qwen3) is used, its `<think>`
output is disabled (via the model's soft switch on the user turn) and stripped from the
display, keeping answers fast and clean.

**Page extraction.** A visibility-aware deep DOM walk — which also crosses shadow DOM and
same-origin iframes — captures what's on app-like and SPA pages; Mozilla Readability is used as
a cleanup pass for genuine articles. The extracted text feeds the prompt.

## How it connects

```
 Content script ── deep DOM text (shadow DOM + same-origin iframes) ──▶ Side panel (thin UI)
                                                                              │ runtime.Port
                                                                              ▼
                                                              Offscreen document
                                                              WebLLM engine on WebGPU
                                                              (downloads once, stays resident)
 Background worker: opens the panel, manages the offscreen document
```

## Properties

- Runs on the GPU; model weights are cached locally after a one-time download.
- The model stays resident for instant follow-ups, and is freed on demand via the Release button.
- Requires WebGPU (recent Chromium) and the `wasm-unsafe-eval` CSP.
- The panel bundle stays small because the WebLLM runtime lives only in the offscreen document.
