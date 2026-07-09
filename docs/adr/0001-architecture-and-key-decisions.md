# ADR 0001 — In-browser (WebGPU) architecture

- **Status:** Accepted
- **Date:** 2026-06-19 (updated 2026-07-09: large-page pipeline, dedicated workers, per-model prompt caps)

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
window is clamped per model and capped conservatively for stability.

**In-browser inference (WebGPU via WebLLM).** The selected model (default **Qwen3 4B**, 4-bit)
runs in the browser with GPU acceleration. On first use its weights download once (~1–6 GB
depending on the model) into the browser cache and are reused afterwards. Inference runs at
~80% of native speed on Apple Silicon. The context window is configurable but defaults to a
conservative 8192-token cap because the KV cache is pre-allocated for the configured window at
load time, so very large windows can reserve enough GPU memory to stall lower-resource machines.
WebGPU and a CSP allowing `wasm-unsafe-eval` (for the WebAssembly runtime) are required.

**The model lives in an offscreen document — the key piece.** The side panel is recreated each
time it opens, so the WebLLM engine runs in a persistent **offscreen document** that stays
resident for the browser session. This keeps the model loaded across panel open/close, so
follow-up questions are instant.

- **Port protocol:** the panel is a thin client that talks to the offscreen document over a
  `runtime.Port` — `init`/`prewarmModel` to load or stage the model, `generate` to run it,
  `index`/`retrieve`/`compress` for the retrieval pipeline; the offscreen doc streams
  `progress`, then token `chunk`s, then `done`, and honors `interrupt`.
- **Dedicated workers:** inside the offscreen document, the WebLLM engine runs in its own
  worker (`webllm.worker.ts`) and the embedding/compression models in another (`ml.worker.ts`).
  The offscreen document often shares the renderer main thread with the side panel, so keeping
  model work off that thread is what keeps the panel clickable during weight loading and indexing.
- **Lifecycle:** the background worker creates the offscreen document on demand. Changing the
  model or context window reloads the engine automatically (debounced), and the panel prewarms
  on open and tab switch (indexing the page, and loading the model only if its weights are
  already cached — a prewarm never triggers a multi-GB download). Loads never queue: a request
  for the model already loading joins it, and a request for a different one cancels it — the
  newest choice wins, and a canceled download loses nothing (fetched weight shards stay in the
  browser cache, so it resumes where it stopped). A background prewarm never cancels a load the
  user started explicitly. On a fresh install the panel
  shows a first-run model picker instead (the default is suggested from the device's reported
  memory) and downloads only on explicit confirmation — or when the user just asks a question.
  A **"Release model from memory"** control closes the document to reclaim RAM/VRAM; it
  reloads on the next question.
- **Bundle split:** the heavy WebLLM and Transformers.js runtimes are imported only by the
  offscreen bundles, so the side-panel bundle stays small (~0.5 MB).

**Large pages.** Small models must not be handed a huge prompt — for answer quality
("lost in the middle") and for system stability: a single oversized prefill on an integrated
GPU can starve the OS compositor (see the post-mortem in
[docs/large-page-handling.md](../large-page-handling.md)). Every model therefore declares a
hard `safePromptTokens` cap on any single prompt, and over-budget pages route through a
client-side pipeline — clean → chunk → embed (MiniLM, cached in IndexedDB) → retrieve top-k →
generate with cited sources; whole-page summaries map-reduce over all chunks; structured
extraction uses XGrammar-constrained JSON. The caps are pinned by `lib/__tests__/budgets.test.ts`.

**Reasoning models.** When a hybrid reasoning model (e.g. Qwen3) is used, its `<think>` phase
is disabled outright via the chat-template flag (`enable_thinking: false`; the soft `/no_think`
text hint proved unreliable), and any stray tags are stripped from the display — visible tokens
start when prefill ends instead of after a hidden reasoning phase.

**Page extraction.** A visibility-aware deep DOM walk — which also crosses shadow DOM and
same-origin iframes — captures what's on app-like and SPA pages; Mozilla Readability is used as
a cleanup pass for genuine articles. The extracted text feeds the prompt.

## How it connects

```
 Content script ── deep DOM text + structure (shadow DOM, same-origin iframes) ──▶ Side panel (thin UI)
                                                                                        │ runtime.Port
                                                                                        ▼
                                                                    Offscreen document (stays resident)
                                                                    ├─ webllm.worker — LLM on WebGPU
                                                                    └─ ml.worker — embeddings · retrieval
                                                                                   · compression
 Background worker: opens the panel, manages the offscreen document
```

## Properties

- Runs on the GPU; model weights are cached locally after a one-time download.
- The model stays resident for instant follow-ups, and is freed on demand via the Release button.
- Requires WebGPU (recent Chromium) and the `wasm-unsafe-eval` CSP.
- The panel bundle stays small because the WebLLM runtime lives only in the offscreen document.
