# Large pages on small local models — design notes

> **Goal:** make a small in-browser model (1B–8B, WebLLM/MLC on WebGPU) behave as close to a
> strong frontier LLM as possible when a page has 5,000+ words — for **Q&A**, **summarization**,
> and **structured extraction** — fully client-side, no server.
>
> This document records the research behind Enclave's large-page pipeline and the design that
> shipped. External claims are tagged with the confidence they earned in fact-checking
> (**high** / `UNVERIFIED`); claims that failed fact-checking are listed in §9 so they don't
> sneak back in. File links point at the modules that implement each piece.

---

## 0. TL;DR

Don't try to fit 5,000 words into the model — **put the ~500–1,500 most relevant, well-ordered
tokens in front of it.** Enclave routes every over-budget page through a layered client-side
pipeline — **clean → chunk → retrieve → (compress) → generate** — instead of truncating. Each
layer has a verified, in-browser implementation. Two tempting shortcuts are skipped deliberately:
speculative decoding is **not available** in browser WebLLM (**high confidence**), and blind
"draft-then-refine" self-correction *hurts* small models (**high confidence**).

---

## 1. The problem, from first principles

### 1.1 The token-budget math

A 5,000-word article is roughly **6,600–7,500 tokens** (≈30,000 chars; the estimator in
[lib/prompt.ts](../lib/prompt.ts) uses `chars/4`). That is more than a small model should ever
receive in one prompt — for quality *and*, as §5 shows, for safety.

The first iteration of Enclave handled overflow the way most page assistants do: clamp the page
to a fixed slice (4,096 tokens) and splice out the middle (keep the first ~70% + last ~30%).
For a how-to buried in section 4 of 9, the answer was simply *gone* before the model ran.

Current budgets, all model-aware rather than hardcoded:

| Constraint | Value | Where |
|---|---|---|
| Hard extraction cap | `MAX_EXTRACTED_CHARS = 60_000` (~15k tokens) | [entrypoints/content.ts](../entrypoints/content.ts) |
| Context-window UI ceiling | `MAX_CONTEXT_TOKENS = 16_384` (default 8,192) | [lib/types.ts](../lib/types.ts) |
| Page body budget | whole context minus scaffold, conversation, and a 1,024-token answer reserve | [lib/prompt.ts](../lib/prompt.ts) |
| **Single-prompt safety cap** | **`safePromptTokens` per model (8B: 1,536)** | [lib/webllmClient.ts](../lib/webllmClient.ts) |
| Per-model context cap | `maxCtx` (8B models: 8,192) | [lib/webllmClient.ts](../lib/webllmClient.ts) |

### 1.2 Why "just raise the context window" is not the fix

Even if the whole page fit, **"Lost in the Middle"** (Liu et al., TACL 2024, `arXiv:2307.03172`,
**high confidence**, replicated through 2026) shows a **U-shaped accuracy curve**: models use
information at the *start* and *end* of a long context well and degrade sharply (30%+ drops) for
information in the *middle*. Two corollaries:

- More raw context ≠ better answers for small models. Relevance-ordering matters more than volume.
- KV-cache memory grows with the context window and competes for the same VRAM as the model
  weights (§6) — a 40k-token window is also a memory-pressure problem in-browser.

### 1.3 The reframe

The job is not "fit 5,000 words into the model." It is **"put the ~500–1,500 most relevant,
well-ordered tokens in front of the model."** That single reframe is the spine of everything below.

---

## 2. The pipeline

Each stage shrinks or reorders content so the model sees less, but better.

```
 RAW PAGE (DOM, 5k+ words)
    │
 ┌──▼─────────────────────────────────────────────────────────────┐
 │ L0  CONTENT REDUCTION   Readability + deep DOM walk             │  cuts boilerplate, keeps structure
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L1  STRUCTURE-AWARE CHUNKING   split on headings/paragraphs     │  keeps the middle, never splices it out
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L2  RETRIEVAL (RAG)   embed → cosine top-k → order by relevance │  pick the sections that matter
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L3  COMPRESSION (optional)   LLMLingua-2 on retrieved chunks    │  ~2× fewer tokens, keep meaning
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L4  GENERATION   WebLLM + XGrammar (JSON) / map-reduce (summary)│  task-specific decoding
 └──▼─────────────────────────────────────────────────────────────┘
   ANSWER  (+ provenance: which chunks were used, click-to-scroll)
```

The crucial property: **L0/L1 always run; L2–L4 are dialed up per task and per page size.** A
short page skips retrieval entirely and goes straight to generation.

### Layer 0 — Content reduction (biggest win per unit effort)

Strip nav, ads, sidebars, footers, and comments before the model (or the embedder) ever sees
the page.

| Library | Notes | Client-side? | Confidence |
|---|---|---|---|
| **Mozilla Readability** | Firefox Reader-View extractor; article-body **F1 ≈ 0.92–0.95**; zero runtime deps | ✅ | **high** |
| **Defuddle** | Drop-in alternative; outputs cleaned HTML **or Markdown**; MV3-proven (Obsidian Web Clipper) | ✅ | **high** |
| **Trafilatura** | Best benchmarked body F1 (0.958), but Python/server — a reference ceiling only | ❌ | **high** |
| **ReaderLM-v2** (Jina, 1.5B) | HTML→Markdown model; strong vendor benchmarks, but eats WebGPU budget and is CC BY-NC (non-commercial) | ⚠️ | **high** |

**What shipped** ([entrypoints/content.ts](../entrypoints/content.ts)): a visibility-aware deep
DOM walk runs *first* — it crosses shadow DOM and same-origin iframes, which is what SPAs and
dashboards need — and Readability serves as the cleanup pass whenever the page is genuinely
article-shaped. The walk simultaneously records coarse structure (headings + text blocks) so the
chunker never has to re-discover it. Emitting Markdown from the article path (via Defuddle) is a
candidate improvement; DOM-distillation alternatives surfaced in research but **no claim
comparing them to Readability for LLM quality survived fact-checking** — a spike, not a commitment.

### Layer 1 — Structure-aware chunking

Split the cleaned text into ~320-token chunks **on natural boundaries** (headings → paragraphs →
sentences), never mid-sentence, each chunk carrying its nearest heading and a little overlap so
facts survive boundaries. This is the direct cure for the "deleted middle": instead of throwing
the middle away, index it.

**What shipped:** [lib/chunking.ts](../lib/chunking.ts) (target 320 / max 500 / overlap 50
tokens). **Late chunking** (Jina, `arXiv:2409.04701`, **high confidence**) — embed the whole
document first, then pool per chunk so each embedding carries document context — is a promising
upgrade at the embedding step, not adopted yet.

### Layer 2 — In-browser retrieval

The core upgrade for **Q&A** and large-page **extraction**. Proven end-to-end, fully
client-side, by open-source extensions (**high confidence**): `nico-martin/gemma4-browser-extension`
(the cleanest strictly-in-browser reference: Transformers.js MiniLM embeddings → IndexedDB →
cosine top-k), plus `DistiLlama` and `Lumos` (good retrieval/chunking references; their
*generation* runs on Ollama, so only the front half applies).

**What shipped** ([lib/retrieval.ts](../lib/retrieval.ts)):

- **Embedder:** `all-MiniLM-L6-v2` (~22M params) via Transformers.js — WebGPU (fp16) when
  available, WASM fallback. At 22M params its VRAM cost is negligible next to the LLM.
- **Search:** brute-force cosine over normalized vectors. A page yields tens-to-hundreds of
  chunks, so an ANN index (Voy, Orama) would be overkill; exact top-k is microseconds.
- **Cache:** IndexedDB keyed by `URL + content-hash + embedder-variant`, LRU-capped, so
  revisits skip re-embedding. Vectors from different backends/precisions are never compared.
- **Ordering:** retrieved chunks are packed **most-relevant last**, nearest the question —
  directly countering the U-curve from §1.2 ([lib/prompt.ts](../lib/prompt.ts)).
- **Isolation:** embedding runs in a dedicated worker inside the offscreen document
  ([entrypoints/offscreen/ml.worker.ts](../entrypoints/offscreen/ml.worker.ts)), so indexing
  never blocks the UI or the LLM.

**Reranking** (cross-encoder over the top-k) is a standard quality lever, but **no claim that a
cross-encoder runs client-side via Transformers.js survived fact-checking** — prototype before
depending on it.

### Layer 3 — Prompt compression (optional)

**LLMLingua-2** (`arXiv:2403.12968`, **high confidence**) drops low-information tokens
task-agnostically using a small BERT-class encoder; **LongLLMLingua** (`arXiv:2310.06839`,
ACL 2024, **high confidence**) reports ~4× compression with up to +21.4% QA on multi-doc
benchmarks — measured on GPT-3.5-class models, so the gain for 1B–8B locals is plausible but
**unquantified**.

**What shipped** ([lib/compress.ts](../lib/compress.ts)): the `@atjsh/llmlingua-2` JS port with
its smallest compressor (TinyBERT, ~57 MB), applied to retrieved chunks at rate 0.5 — **off by
default** (Settings → "Compress retrieved context") because the upstream port is explicitly
experimental. Failures degrade gracefully to the uncompressed chunks.

### Layer 4 — Task-specific generation

- **Q&A** → stuff the top-k chunks (most-relevant last) and cite which chunks were used.
- **Structured extraction** → **XGrammar constrained decoding** (`arXiv:2411.15100`, **high
  confidence**), which is already WebLLM's structured-generation backend: pass a JSON Schema via
  `response_format`, get structurally-valid JSON by construction. The guarantee is *structural*,
  not *factual* — valid JSON, not necessarily correct values. ([lib/tasks.ts](../lib/tasks.ts))
- **Whole-page summarization** → never truncate; **hierarchical merging** (map-reduce): summarize
  each chunk, then merge summaries in groups until they fit, then one final streamed pass
  (BooookScore, `arXiv:2310.00785`, ICLR 2024, **high confidence** for the pattern; that it
  beats truncation *specifically for 1B–8B models on web pages* is `UNVERIFIED` — it is the
  principled default and removes the context-window limit entirely).
  ([lib/summarize.ts](../lib/summarize.ts))

---

## 3. Per-task recipes

| Task | Pipeline | Why |
|---|---|---|
| **Q&A** | clean → chunk → **retrieve top-k** → (compress) → stuff + cite | The answer usually lives in 1–3 sections; retrieval beats stuffing for small models |
| **Summarize** | clean → chunk → **map-reduce** | Must cover the *whole* page; retrieval would drop content |
| **Extract** | clean → (retrieve if over budget) → **XGrammar JSON** | Structural correctness by construction |

Routing lives in [lib/tasks.ts](../lib/tasks.ts) (per-task retrieval/map-reduce eligibility,
system prompts, sampling) and [entrypoints/sidepanel/App.tsx](../entrypoints/sidepanel/App.tsx).

---

## 4. Small-model quality: what helps, what to avoid

| Technique | Verdict | Confidence |
|---|---|---|
| **Retrieval** over stuffing | ✅ Core strategy; counters lost-in-the-middle | high |
| **Prompt compression** (LLMLingua-2) | ✅ Fewer tokens *and* can raise QA | high (effect); unquantified for small models |
| **XGrammar constrained decoding** | ✅ Use for all structured output; already in WebLLM | high |
| **Hierarchical summarization** | ✅ Default for whole-page summaries | high (pattern) |
| **Few-shot prompting** | ➖ Standard, low-risk; no strong evidence found either way | UNVERIFIED |
| **Self-consistency** (sample N, vote) | ➖ Plausible; multiplies cost | UNVERIFIED |
| **Blind draft-then-refine self-correction** | ❌ **Avoid** — *degrades* accuracy without external feedback (`arXiv:2310.01798`) | high |
| **Verification loops with external feedback** (e.g. re-retrieval) | ✅ Still viable — only *blind* self-refine is harmful | high |
| **Speculative decoding** | ❌ **Unavailable** in browser WebLLM (native MLC backends only) | high |

> The takeaway: the wins are in **what you feed the model** (retrieval, compression, structure,
> constrained decoding), not in clever decoding-time self-talk.

---

## 5. Prompt size is a SAFETY limit — the WindowServer post-mortem (2026-07-09)

Field incident: on a 16 GB M4 MacBook Pro (Low Power Mode), asking about a 5,961-word page with
an 8B model killed the **entire macOS login session**. From the system logs:
`WindowServer … userspace_watchdog_timeout` — "40 seconds since last successful checkin" — the
display server was starved of the GPU past its watchdog, and macOS terminated it.

**Chain of causation:**

1. A text selection was present, which (at the time) disabled retrieval — so the panel *stuffed*
   ~6,800 tokens into one prompt.
2. Prefill is submitted in `prefill_chunk_size` slices **compiled into the model wasm**
   (Llama-3.1-8B: **8192** — the whole prompt becomes a single near-uninterruptible GPU
   submission; Qwen3-8B/4B and gemma-2: 2048). It cannot be overridden at runtime.
3. An 8B q4f16 prefill of ~7k tokens on a base-M4 integrated GPU (throttled further by Low Power
   Mode) monopolizes the GPU for tens of seconds. Unified memory means there is no isolation
   between Chrome's WebGPU work and the OS compositor.
4. WindowServer misses its 40 s watchdog → macOS kills it → the user is logged out.

**Rules pinned by [lib/__tests__/budgets.test.ts](../lib/__tests__/budgets.test.ts) — do not regress:**

- Every model declares `safePromptTokens` in [lib/webllmClient.ts](../lib/webllmClient.ts) — a
  hard cap on ANY single prompt body. Bigger model ⇒ smaller cap (8B: 1,536). The caps are sized
  for tolerable compositor-freeze windows, not just crash avoidance.
- Retrieval engages whenever the page exceeds the cap — **a selection no longer disables
  retrieval**; it sharpens the retrieval query instead.
- Map-reduce merge prompts respect the same cap ([lib/summarize.ts](../lib/summarize.ts)) — a
  merge prompt is a GPU prefill like any other.
- 8B `maxCtx` is 8,192: the KV cache is preallocated for the full window, and WebLLM's
  `vram_required_MB` figures assume 4,096 — a 16k window on an 8B model overcommits 16 GB machines.

Related fix, same date: onnxruntime-web dynamic-imports its wasm loader from a CDN, which the
extension CSP (`script-src 'self'`) blocks — so retrieval silently failed with "no available
backend found" in packaged builds. The runtime is now bundled (a `wxt.config.ts` build hook +
[lib/ortEnv.ts](../lib/ortEnv.ts)).

---

## 6. The hard constraint: WebGPU resource budget

**Verified (high confidence):**

- WebLLM runs fully in-browser at **~71–80% of native throughput** and ships MV3 extension examples.
- Small models are usably fast on an 8 GB laptop GPU (e.g. Llama-3.1-8B ≈ 41 tok/s on an M3 Max;
  3–4B models faster still).
- WebGPU fits **latency-tolerant, privacy-sensitive, no-server** workloads — exactly this
  profile — not latency-critical serving.

**The open risk:** every published speed is for a **single small model running alone**. No
benchmark confirms an 8B model + an embedder + a compressor co-resident in one tab on an 8 GB
GPU. Mitigations shipped: the embedder is tiny (~22M params), the compressor is opt-in, both run
sequenced around generation rather than concurrently ([entrypoints/offscreen/](../entrypoints/offscreen/)),
embeddings are cached in IndexedDB, and the per-model caps in §5 bound the worst case. Hardware
auto-tiering (detect headroom, pick the stack) remains future work.

---

## 7. Viewport-aware ranking & provenance

"Prefer what's on screen" is a real UX lever, but research turned up **no verifiable prior art**
for viewport-aware LLM reading (nothing confirmed about how Arc Max, Brave Leo, or Edge Copilot
slice pages) — so Enclave treats it as product experimentation, shipped conservatively:

- **As a retrieval prior, not a filter** (Settings → "Prefer on-screen content", off by default):
  chunks visible at capture time get a small score boost (×1.15) during ranking — off-screen
  content is never discarded, so "summarize the page" still works.
- **Provenance + scroll-to-source** (always on for retrieval answers): answers list the sections
  they drew from; clicking one scrolls the page there and highlights it. Small models earn trust
  by showing their work.

Progressive embedding (index visible chunks first, lazily embed the rest) is a natural next step.

---

## 8. Appendix — projects and papers

### Projects studied or adopted

| Project | Relevance | Fully client-side? |
|---|---|---|
| `mlc-ai/web-llm` | The engine; MV3 examples + XGrammar structured generation | ✅ |
| `nico-martin/gemma4-browser-extension` | Closest prior art: in-browser WebGPU gen + MiniLM embeddings + IndexedDB retrieval | ✅ |
| `shreyaskarnik/DistiLlama` | Transformers.js + Voy + LangChain.js retrieval; Readability cleanup | ⚠️ generation via Ollama |
| `andrewnguonly/Lumos` | Page-grounded RAG; per-domain chunking heuristics | ⚠️ generation via Ollama |
| `mozilla/readability` | Content reduction (adopted) | ✅ |
| `kepano/defuddle` | Markdown-output extractor, MV3-proven (candidate) | ✅ |
| `tantaraio/voy` / `oramasearch/orama` | WASM/TS vector search (not needed at page scale) | ✅ |
| `mlc-ai/xgrammar` | Constrained JSON/CFG decoding (adopted via WebLLM) | ✅ |
| `atjsh/llmlingua-2-js` | In-browser prompt compression (adopted, experimental, opt-in) | ✅ |
| `jina-ai/late-chunking` | Context-preserving chunk embeddings (candidate) | ✅ |

### Key papers

| Paper | arXiv | Takeaway | Confidence |
|---|---|---|---|
| Lost in the Middle | 2307.03172 | U-shaped degradation → retrieve/compress, order by relevance | high |
| LongLLMLingua | 2310.06839 | ~4× compression, +21.4% QA, targets position bias | high |
| LLMLingua-2 | 2403.12968 | Task-agnostic compression with small BERT encoders | high |
| Selective Context | 2304.12102 | Self-information pruning (no browser port exists) | high |
| BooookScore | 2310.00785 | Hierarchical-merge vs incremental-update summarization | high |
| LLMs Cannot Self-Correct Yet | 2310.01798 | Blind self-correction degrades accuracy | high |
| XGrammar | 2411.15100 | 100% structurally-correct constrained decoding | high |
| Late Chunking | 2409.04701 | Whole-document context in chunk embeddings | high |
| WebLLM | 2412.15803 | In-browser engine, ~80% native throughput | high |

---

## 9. Honest gaps & refuted claims

**Open questions (worth their own prototype pass):**

1. **Concurrent VRAM budget** — does 8B + embedder + compressor truly fit on 8 GB? (§6)
2. **In-browser cross-encoder reranking** — feasibility unverified.
3. **Markdown extraction (Defuddle) / DOM-distillation vs Readability** for LLM answer quality.
4. **Map-reduce vs truncation for *small* models specifically** — principled, not proven.
5. **Late chunking** at the embedding step.
6. **Self-consistency / few-shot** gains for 1B–8B models.

**Refuted in fact-checking — do not cite:**

- "Retrieval context actively *destroys* small-model answers (distraction effect); small models
  fail 85–100% even with oracle retrieval" — refuted; do not use it to argue against RAG.
- "Orama is a complete client-side RAG pipeline" — it is an index, not a pipeline.
- The unqualified "LLMs can never self-correct" — only the *no-external-feedback* version holds.

**Time-sensitivity:** WebLLM and MLC evolve quickly. Re-check speculative-decoding support and
the `@atjsh/llmlingua-2` port's maturity before relying on either statement above.
