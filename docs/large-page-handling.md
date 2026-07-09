# Handling Large Pages with Small Local LLMs — Research & Architecture

> **Goal:** Make a small in-browser model (1B–8B, WebLLM/MLC, WebGPU) behave as close to a
> strong frontier LLM as possible when a page has 5,000+ words — for **Q&A**, **summarization**,
> and **structured extraction**, fully client-side, no server.
>
> This document connects the dots between (a) what the research says works and (b) what your
> codebase does today. Every external claim is tagged with a confidence level from two
> adversarially-verified deep-research passes (217 sub-agents, ~3.4M tokens). Where evidence
> did **not** survive verification, it is labelled `UNVERIFIED` and treated as a hypothesis, not a fact.

---

## 0. TL;DR — the one-paragraph answer

Your instinct (compress / reformat / show-only-visible) is right, but the single highest-leverage
fix is structural: **stop stuffing the page and start retrieving from it.** Today your code force-fits
page text into a fixed 4,096-token slice and, when it overflows, **deletes the middle of the page**
([lib/prompt.ts](../lib/prompt.ts)). That collides head-on with the best-established result in this
space — "lost in the middle" (`arXiv:2307.03172`, **high confidence**): models are weakest exactly
where your truncation dumps content. The fix is a layered client-side pipeline — **clean → chunk →
retrieve → (compress) → generate** — every layer of which has a verified open-source, in-browser
implementation. Speculative decoding is **not** available in browser WebLLM (**high confidence**), and
blind "draft-then-refine" self-correction *hurts* small models (**high confidence**), so skip both.

---

## 1. The problem, from first principles

### 1.1 The token-budget math

A 5,000-word article is roughly **6,600–7,500 tokens** (≈30,000 chars; your own estimate is
`chars/4` in [lib/prompt.ts](../lib/prompt.ts)). That already exceeds what you send today.

| Constraint | Current value | File |
|---|---|---|
| Hard extraction cap | `MAX_EXTRACTED_CHARS = 60_000` (~15k tokens) | [entrypoints/content.ts](../entrypoints/content.ts) |
| **Page content sent to model** | **`MAX_PAGE_CONTEXT_TOKENS = 4096`** | [lib/prompt.ts](../lib/prompt.ts) |
| Context-window UI ceiling | `MAX_CONTEXT_TOKENS = 8192` | [lib/types.ts](../lib/types.ts) |
| Answer reservation | `ANSWER_RESERVE_TOKENS = 1024` | [lib/prompt.ts](../lib/prompt.ts) |
| Model catalog max context | up to `40960` (e.g. Qwen3-4B) | [lib/webllmClient.ts](../lib/webllmClient.ts) |

**Two bugs fall out of this table:**

1. **You under-use your own models.** Qwen3-4B advertises 40,960-token context, but the page is
   clamped to 4,096 and the UI to 8,192. Most of the model's context is unreachable.

2. **You delete the middle.** When the page overflows the budget, `truncateToTokens()` keeps the
   **first 70% + last 30%** and splices out everything between:

   ```ts
   // lib/prompt.ts — current behaviour
   const head = Math.floor(maxChars * 0.7);
   const tail = maxChars - head;
   return text.slice(0, head) + '\n\n[…content trimmed…]\n\n' + text.slice(text.length - tail);
   ```

   For a how-to buried in section 4 of 9, the answer is simply *gone* before the model runs.

### 1.2 Why "just raise the cap to 40k" is not the real fix

Even if you fed the whole page, **"Lost in the Middle"** (Liu et al., TACL 2024, `arXiv:2307.03172`,
**high confidence**, replicated through 2026) shows a **U-shaped accuracy curve**: models use
information at the *start* and *end* of a long context well and degrade sharply (30%+ drops) for
information in the *middle*. Two corollaries:

- More raw context ≠ better answers for small models. Relevance-ordering matters more than volume.
- KV-cache memory grows with context length and competes for the same VRAM as the model weights
  (see §7) — so a 40k-token prompt is also a memory-pressure problem in-browser.

> ⚠️ **What the research does NOT say.** A round-2 claim that retrieval context *actively destroys*
> small-model answers via a "distraction effect," and that small models fail 85–100% even with
> perfect retrieval (`arXiv:2603.11513`), was **REFUTED 0-3**. Do **not** use it to argue against RAG.
> RAG remains the recommended direction.

### 1.3 The reframe

The job is not "fit 5,000 words into the model." It is **"put the ~500–1,500 most relevant,
well-ordered tokens in front of the model."** That single reframe is the spine of everything below.

---

## 2. The architecture — connecting the dots

Think of it as a pipeline. Each stage shrinks or reorders content so the model sees less, but better.

```
 RAW PAGE (DOM, 5k+ words)
    │
 ┌──▼─────────────────────────────────────────────────────────────┐
 │ L0  CONTENT REDUCTION   Readability / Defuddle → clean text/MD  │  cuts boilerplate, ~30–70% tokens
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L1  STRUCTURE-AWARE CHUNKING   split on headings/paragraphs     │  keeps the middle, never splices it out
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L2  RETRIEVAL (RAG)   embed (Transformers.js) → vector search   │  pick top-k chunks for the query
 │     + late chunking + (optional) reranking                       │
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L3  COMPRESSION (optional)   LLMLingua-2 on selected chunks     │  ~2–4× fewer tokens, keep meaning
 ├──▼─────────────────────────────────────────────────────────────┤
 │ L4  GENERATION   WebLLM + XGrammar (structured) / map-reduce    │  task-specific decoding
 └──▼─────────────────────────────────────────────────────────────┘
   ANSWER  (+ provenance: which chunks were used)
```

The crucial property: **L0/L1 always run; L2–L4 are dialed up per task and per page size.** A short
page skips retrieval entirely and goes straight to generation.

---

### Layer 0 — Content reduction (biggest win per unit effort)

Strip nav, ads, sidebars, footers, comments before the model (or the embedder) ever sees the page.

| Library | What it gives you | Client-side? | Confidence |
|---|---|---|---|
| **Mozilla Readability** | Firefox Reader-View extractor; `parse()` → `textContent` (tags removed) + metadata. Article-body **F1 ≈ 0.92–0.95**. Zero runtime deps. **Already in your `package.json`.** | ✅ yes | **high** |
| **Defuddle** | Drop-in Readability alternative; outputs cleaned HTML **or Markdown**; more forgiving, better footnotes/math/code, extracts schema.org metadata. Built for the Obsidian Web Clipper MV3 extension. Zero deps. | ✅ yes | **high** |
| **Trafilatura** | Best benchmarked body **F1 0.958** (Rust port 0.97). | ❌ Python (server) — **reference ceiling only** | **high** |
| **ReaderLM-v2** (Jina, 1.5B) | Purpose-built HTML→Markdown/JSON model; beats GPT-4o/Gemini-flash on HTML→MD (ROUGE-L 0.84 vs 0.69, *vendor benchmark*). | ⚠️ yes, but eats WebGPU budget; **CC BY-NC-4.0 (non-commercial)** | **high** |

**DistiLlama's author empirically found summaries got "much better" after adding Readability** because
boilerplate was removed (**high confidence**). This is the cheapest quality win available to you.

**Your codebase today:** [entrypoints/content.ts](../entrypoints/content.ts) runs a custom `deepText`
DOM walk *first* and only falls back to Readability for article-shaped pages. That's a reasonable
hybrid (it catches SPAs/dashboards that Readability misses), but the article path should produce
**Markdown** (via Defuddle) rather than flattened text — headings/lists/tables are exactly the
structure L1 chunking and the model both benefit from.

> **Accessibility-tree / DOM-distillation alternatives** (`dom-to-semantic-markdown`, `LLMFeeder`,
> Chromium DOM Distiller) surfaced as real projects but **no claim comparing them to Readability for
> LLM quality survived verification** — treat as `UNVERIFIED` and worth a spike, not a commitment.

---

### Layer 1 — Structure-aware chunking

Split the cleaned text into ~200–500 token chunks **on natural boundaries** (headings → paragraphs →
sentences), never mid-sentence. This is the direct cure for the "deleted middle" bug: instead of
throwing the middle away, you index it.

- **Recursive/character splitting** (LangChain.js `RecursiveCharacterTextSplitter`) is the baseline
  used by DistiLlama and Lumos (**high confidence** these projects exist and use it).
- **Late chunking** (Jina, `arXiv:2409.04701`, **high confidence**): embed the *whole* document
  through the transformer first, *then* pool into chunks — so each chunk embedding carries
  whole-document context. Directly mitigates context loss across chunk boundaries on long pages.
  Apply it at the embedding step in L2, not as a separate splitter.

---

### Layer 2 — In-browser retrieval (RAG)

This is the core upgrade for **Q&A**. Embed chunks, embed the query, return the top-k most similar
chunks, and send only those. **Proven end-to-end, fully client-side**, by two open-source extensions
(**high confidence**):

- **`nico-martin/gemma4-browser-extension`** — the cleanest reference for *strictly in-browser*
  WebGPU generation: content script extracts structured content → embeds with **all-MiniLM-L6-v2
  (~22M params)** via Transformers.js → stores vectors in **IndexedDB** → cosine similarity → returns
  top sections. **This is the closest existing project to your target architecture.**
- **`DistiLlama`** — `Transformers.js` (`Xenova/jina-embeddings-v2-small-en`) + **Voy** vector store +
  LangChain.js retriever. *Caveat:* its generation step uses Ollama (local server), so borrow its
  retrieval/chunking, not its inference path.
- **`Lumos`** — page-grounded RAG with per-domain chunk sizes (e.g. Wikipedia 2000/500). Also
  Ollama-based generation; borrow the chunking heuristics.

**Building blocks (all client-side, all verified high confidence):**

| Component | Pick | Notes |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (~22M) or `jina-embeddings-v2-small` via **Transformers.js** | ONNX/WASM, optional WebGPU; normalize + cosine |
| Vector search | **Voy** (Rust/WASM k-d tree, ~75KB) | exact NN; tiny; no server |
| Vector search (alt) | **Orama** (BM25 + vector + **hybrid** in one TS lib) | in-memory; scope as an *index*, not a turnkey RAG framework — the "complete RAG pipeline" claim was **REFUTED 1-2** |
| Storage | **IndexedDB** | persist embeddings per URL to avoid re-embedding on revisit |

> **Reranking** (cross-encoder like `jina-reranker-v2`, `bge-reranker`) to reorder top-k before the
> model is a standard quality lever, and `jina-reranker-v2-base` appeared as a source — but **no claim
> that a cross-encoder reranker runs client-side via Transformers.js survived verification.** Mark
> `UNVERIFIED`; prototype before depending on it.

---

### Layer 3 — Prompt compression (optional, high upside)

After retrieval you still may have more text than you want. **Prompt compression** drops low-information
tokens while preserving meaning — and notably *also* counteracts position bias:

- **LongLLMLingua** (`arXiv:2310.06839`, ACL 2024, **high confidence**): **~4× fewer tokens** while
  **boosting QA by up to 21.4%** on multi-doc NaturalQuestions; explicitly targets the
  "lost in the middle" position-bias problem. *Caveat:* measured on GPT-3.5-Turbo with a LLaMA-7B
  compressor — the gain for 1B–8B local models is plausible but **unquantified**.
- **LLMLingua-2** (`arXiv:2403.12968`, **high confidence**): **task-agnostic** (one compressor serves
  Q&A, summarization, *and* extraction), uses small **BERT-class encoders**.
- **`@atjsh/llmlingua-2`** (npm, **high confidence**): a real JS/TS port that runs **fully in-browser**
  via Transformers.js (optional WebGPU, "server-side processing is not required by default"). Compressor
  sizes: **TinyBERT 57MB / MobileBERT 99MB** / BERT 710MB / XLM-RoBERTa 2.2GB. The 57–99MB options are
  an order of magnitude smaller than your task model and **can co-reside**. ⚠️ Labelled *experimental*,
  no unit tests, no port-measured quality numbers.

**Selective Context** (`arXiv:2304.12102`) is a real alternative but **Python-only — no browser port
exists** (**high confidence**); you'd have to port it. Lower priority than LLMLingua-2.

---

### Layer 4 — Task-specific generation

The same retrieved/compressed context is consumed differently per task:

**Q&A** → stuff top-k chunks (well-ordered: most-relevant near the *end*, per lost-in-the-middle) into
the prompt; cite which chunks were used.

**Structured extraction** → **XGrammar constrained decoding** (`arXiv:2411.15100`, **high confidence**):
guarantees **100% structurally-correct** JSON/regex/CFG output by masking invalid tokens. Crucially,
it's exposed as **`@mlc-ai/web-xgrammar`** (WASM) and is **already the structured-generation backend
integrated into WebLLM** — there's an official in-browser Structured Generation playground. This makes
reliable client-side extraction a near-free add. ⚠️ The guarantee is *structural*, not *factual* — it
ensures valid JSON, not correct values.

**Whole-page summarization** → don't truncate; use a multi-pass workflow over chunks (`arXiv:2310.00785`,
BooookScore, ICLR 2024, **high confidence**):
- **Hierarchical merging** (= map-reduce): summarize each chunk, then summarize the summaries. Higher
  coherence.
- **Incremental updating** (= refine): maintain a running summary, update per chunk. More detail,
  lower coherence.

⚠️ BooookScore studied large hosted models on book-length text; that map-reduce beats single-pass
truncation **specifically for 1B–8B models on web pages is `UNVERIFIED`** — but it's the principled
default and lets you summarize pages far beyond the context window.

---

## 3. Per-task recipes (what to actually build)

| Task | Pipeline | Why |
|---|---|---|
| **Q&A** | L0 clean → L1 chunk → **L2 retrieve top-k** → (L3 compress) → L4 stuff+cite | Answer usually lives in 1–3 sections; retrieval beats stuffing for small models |
| **Summarize** | L0 clean → L1 chunk → **L4 hierarchical map-reduce** | Must cover the *whole* page; retrieval would drop content |
| **Extract** | L0 clean → (L2 retrieve if huge) → **L4 XGrammar JSON schema** | Structural correctness guarantee; schema-constrained decoding |

---

## 4. Small-model quality: what helps, what to avoid

| Technique | Verdict | Confidence |
|---|---|---|
| **RAG / retrieval** over stuffing | ✅ Core strategy; counters lost-in-the-middle | high |
| **Prompt compression** (LLMLingua-2) | ✅ Fewer tokens *and* can raise QA | high (effect), unquantified for small models |
| **XGrammar constrained decoding** | ✅ Use for all structured output; already in WebLLM | high |
| **Hierarchical/incremental summarization** | ✅ Default for whole-doc summary | high (pattern); unverified for small models |
| **Few-shot prompting** | ➖ Standard, low-risk — but no surviving verification here | UNVERIFIED |
| **Self-consistency** (sample N, vote) | ➖ Plausible, multiplies cost; not verified here | UNVERIFIED |
| **Blind draft-then-refine / intrinsic self-correction** | ❌ **Avoid** — *degrades* accuracy without external feedback (GPT-3.5 CommonSenseQA 0.87→0.57 after one round) | high |
| **Verification/grounding loops** (refine *with* external feedback, e.g. re-retrieval) | ✅ Still viable — only *blind* self-refine is harmful | high |
| **Speculative decoding** | ❌ **Not available** in browser WebLLM (native MLC backends only) | high |

> The takeaway on "make the model smarter": the wins are in **what you feed it** (retrieval,
> compression, structure, constrained decoding), not in clever **decoding-time self-talk**.

---

## 4.5 Post-mortem (2026-07-09): the WindowServer crash — why prompt size is a SAFETY limit

Field incident: on a 16 GB M4 MacBook Pro (Low Power Mode), asking about a 5,961-word page
with an 8B model killed the entire macOS login session. Verified from system logs:
`WindowServer ... userspace_watchdog_timeout` — "40 seconds since last successful checkin" —
i.e. the display server was starved of the GPU past its watchdog and macOS terminated it.

**Chain of causation:**
1. A text selection was present (or the task was selection-driven), which disabled RAG, so the
   panel *stuffed* ~6,800 tokens (76% of the page) into one prompt.
2. Prefill is submitted in `prefill_chunk_size` slices **compiled into the model wasm**
   (Llama-3.1-8B: **8192** → the whole prompt is a single near-uninterruptible GPU submission;
   Qwen3-8B/Qwen3-4B/gemma-2: 2048). It cannot be overridden at runtime.
3. An 8B q4f16 prefill of ~7k tokens on a base-M4 integrated GPU (throttled further by Low
   Power Mode) monopolizes the GPU for tens of seconds. Unified memory means there is no
   isolation between Chrome's WebGPU work and the OS compositor.
4. WindowServer misses its 40 s watchdog → macOS kills it → user is logged out.

**Rules pinned by `lib/__tests__/budgets.test.ts` (do not regress):**
- Every model declares `safePromptTokens` in [lib/webllmClient.ts](../lib/webllmClient.ts) — a
  hard cap on ANY single prompt body. Bigger model ⇒ smaller cap (8B: 1,536).
- RAG engages whenever the page exceeds the cap — **a selection no longer disables retrieval**;
  it sharpens the retrieval query instead.
- Map-reduce merge prompts respect the same cap ([lib/summarize.ts](../lib/summarize.ts)).
- 8B `maxCtx` is 8,192: KV-cache is preallocated for the full context window, and WebLLM's
  `vram_required_MB` is quoted at 4,096 ctx — 16k ctx on an 8B model overcommits 16 GB machines.

Related fix, same date: onnxruntime-web dynamic-imports its wasm loader from a CDN, which the
extension CSP (`script-src 'self'`) blocks — so RAG indexing failed with "no available backend
found" in the packaged extension. The runtime is now bundled (`wxt.config.ts` build hook +
[lib/ortEnv.ts](../lib/ortEnv.ts)).

---

## 5. The hard constraint: WebGPU resource budget ⚠️

This is the **biggest unresolved engineering risk** and the main thing to prototype early.

**Verified (high confidence):**
- WebLLM runs fully in-browser at **~71–80% of native throughput**; ships MV3 chrome-extension examples.
- Small models are usably fast on an 8GB laptop GPU: **Qwen2.5-0.5B ≈ 51 tok/s, 1.5B ≈ 46 tok/s,
  Phi-3.5-mini-3.8B ≈ 71 tok/s, Llama-3.1-8B ≈ 41 tok/s** (M3 Max).
- WebGPU is the right fit for **latency-tolerant, privacy-sensitive, no-server** workloads (your exact
  profile) — *not* for latency-critical serving.

**NOT verified — the open risk:** every measured speed is for a **single small model running alone.**
**No benchmark confirms an 8B model + an embedding model + a vector index + (optionally) a compressor all
fit concurrently in one tab on an 8GB GPU.** KV-cache for long contexts adds further VRAM pressure.

**Design implications:**
1. **Embedder and compressor are small** (MiniLM ~22M; TinyBERT/MobileBERT 57–99MB) and run on
   WASM/CPU or share WebGPU — keep them off the critical VRAM path where possible.
2. **Run embedding/compression in the offscreen document or a Web Worker**, sequenced *around* model
   inference, not concurrently with a generation pass, on constrained hardware.
3. **Tier by hardware:** small model + RAG on 8GB; allow the full stack only when headroom is detected.
4. **Cache embeddings in IndexedDB** keyed by URL+content-hash so re-visits skip re-embedding.

---

## 6. Viewport / progressive reading (your idea) — promising but `UNVERIFIED`

You proposed "only send what's visible on screen." It's a genuinely good UX lever, **but the research
turned up no open-source extension or paper that verifiably does viewport-only / scroll-aware LLM
reading** (and nothing confirmed about how Arc Max / Brave Leo / Edge Copilot slice pages). So treat
this as **product innovation**, not a documented pattern.

How it connects to the dots above:
- **As a retrieval prior, not a hard filter.** Boost chunks currently in/near the viewport in ranking
  (a recency/proximity signal on top of cosine similarity), rather than discarding off-screen text —
  otherwise "summarize the page" or a question about a later section breaks.
- **Progressive embedding.** Embed visible chunks first for instant first-answer, then lazily embed the
  rest in the background as the user dwells/scrolls — hides the indexing latency.
- **Answer provenance + scroll-to.** Since you'll already track which chunks were used (L2), highlight
  them and offer "jump to source" — a UX frontier feature that also builds trust in a small model.

This is where you can differentiate. Just validate it behind a flag with real pages before committing.

---

## 7. Phased roadmap (mapped to your files)

Ordered by value-per-effort. Each phase is shippable on its own.

### Phase 0 — Stop deleting the middle (days, no new deps)
- **Make Readability/Defuddle the primary path for article pages** and emit **Markdown**, keeping your
  `deepText` walk as the SPA/dashboard fallback. — [entrypoints/content.ts](../entrypoints/content.ts)
- **Scale the page budget to the selected model's context**, not a hardcoded 4,096. Raise
  `MAX_PAGE_CONTEXT_TOKENS` / `MAX_CONTEXT_TOKENS` to track `maxCtx` from
  [lib/webllmClient.ts](../lib/webllmClient.ts). — [lib/prompt.ts](../lib/prompt.ts), [lib/types.ts](../lib/types.ts)
- **Replace blind head/tail truncation** with structure-aware selection (drop whole low-value sections,
  never splice mid-content). — [lib/prompt.ts](../lib/prompt.ts) `truncateToTokens()`
- Keep the existing truncation UI note; it's good. — [entrypoints/sidepanel/App.tsx](../entrypoints/sidepanel/App.tsx)

### Phase 1 — RAG for Q&A (the core upgrade)
- Add **Transformers.js** + **all-MiniLM-L6-v2**; chunk (L1), embed, store in **IndexedDB**, search
  with **Voy** (or Orama for hybrid). Run it in the **offscreen doc / Web Worker**.
- New modules: `lib/chunking.ts`, `lib/retrieval.ts`. Hook the retrieval step in
  [entrypoints/sidepanel/App.tsx](../entrypoints/sidepanel/App.tsx) `capturePage()` → before
  `buildMessages()`; extend [lib/prompt.ts](../lib/prompt.ts) to accept top-k chunks instead of one blob.
- Study `nico-martin/gemma4-browser-extension` as the reference.

### Phase 2 — Structured extraction
- Add **`@mlc-ai/web-xgrammar`**; pass JSON-schema constraints into
  `engine.chat.completions.create()`. — [lib/webllm.ts](../lib/webllm.ts)
- Add an `'extract'` task alongside the existing `QuickAction = 'ask' | 'summarize' | 'explain'`. —
  [lib/types.ts](../lib/types.ts), [entrypoints/sidepanel/App.tsx](../entrypoints/sidepanel/App.tsx)

### Phase 3 — Whole-page summarization
- Implement **hierarchical map-reduce** over chunks for `'summarize'`, so it covers pages larger than
  the context window instead of truncating. — new `lib/summarize.ts`, wired from the Summarize action.

### Phase 4 — Compression, late chunking, and the viewport UX
- Add **`@atjsh/llmlingua-2`** (TinyBERT/MobileBERT) to compress retrieved context when over budget.
- Switch embeddings to **late chunking**.
- Prototype **viewport-aware ranking + answer provenance** behind a feature flag (§6).
- (Spike) evaluate an in-browser **reranker** and **DOM-distillation** extractor.

---

## 8. Survey appendix — papers, repos, libraries

### Open-source projects to study or adopt
| Project | Relevance | Fully client-side? |
|---|---|---|
| `nico-martin/gemma4-browser-extension` | **Best reference**: in-browser WebGPU gen + Transformers.js embeddings + IndexedDB RAG | ✅ |
| `shreyaskarnik/DistiLlama` | Transformers.js + Voy + LangChain.js retrieval; Readability cleanup | ⚠️ gen via Ollama |
| `andrewnguonly/Lumos` | Page-grounded RAG, per-domain chunking heuristics | ⚠️ gen via Ollama |
| `mlc-ai/web-llm` | Your engine; ships MV3 examples + XGrammar structured-gen playground | ✅ |
| `mozilla/readability` | Content reduction (**already a dep**) | ✅ |
| `kepano/defuddle` | Markdown-output extractor, MV3-proven | ✅ |
| `tantaraio/voy` | WASM k-d-tree vector search (~75KB) | ✅ |
| `oramasearch/orama` | BM25 + vector + hybrid index | ✅ (index only) |
| `mlc-ai/xgrammar` (`@mlc-ai/web-xgrammar`) | Constrained JSON/CFG decoding | ✅ |
| `atjsh/llmlingua-2-js` | In-browser prompt compression (experimental) | ✅ |
| `jina-ai/late-chunking` | Context-preserving chunk embeddings | ✅ |
| `romansky/dom-to-semantic-markdown`, `jatinkrmalik/LLMFeeder` | DOM→Markdown distillation (unverified vs Readability) | ✅ |

### Key papers
| Paper | arXiv | Takeaway | Confidence |
|---|---|---|---|
| Lost in the Middle | 2307.03172 | U-shaped degradation → retrieve/compress, order by relevance | high |
| LongLLMLingua | 2310.06839 | ~4× compression, +21.4% QA, targets position bias | high |
| LLMLingua-2 | 2403.12968 | Task-agnostic compression, small BERT encoders | high |
| Selective Context | 2304.12102 | Self-information pruning (no browser port) | high |
| BooookScore | 2310.00785 | Hierarchical-merge vs incremental-update summarization | high |
| LLMs Cannot Self-Correct Yet | 2310.01798 | Blind self-correction degrades accuracy | high |
| XGrammar | 2411.15100 | 100% structural-correctness constrained decoding | high |
| Late Chunking | 2409.04701 | Whole-doc context in chunk embeddings | high |
| ReaderLM-v2 | 2503.01151 | Small HTML→MD/JSON model | high (vendor benchmark) |
| WebLLM | 2412.15803 | In-browser engine, ~80% native throughput | high |
| WebGPU inference benchmark | 2604.02344 | Small-model tok/s on 8GB; envelope of WebGPU | high |

---

## 9. Honest gaps & what NOT to claim

**Open questions (need their own research/prototype pass):**
1. **Concurrent VRAM budget** — does 8B + embedder + index + compressor fit on 8GB? (§5) — the key risk.
2. **Viewport/progressive reading** — no verified prior art; validate empirically (§6).
3. **In-browser cross-encoder reranking** — feasibility unverified.
4. **DOM-distillation vs Readability** for LLM quality — unverified.
5. **Map-reduce vs truncation for *small* models** specifically — unverified (principled, not proven).
6. **Self-consistency / few-shot** accuracy gains for 1B–8B — unverified here.

**Do NOT cite (refuted in verification):**
- "Retrieval context actively destroys small-model answers / distraction effect" (`2603.11513`) — **REFUTED 0-3**.
- "Small models fail 85–100% even with oracle retrieval" (`2603.11513`) — **REFUTED 0-3**.
- "Orama is a complete client-side RAG pipeline" — **REFUTED 1-2** (it's an index).
- The unqualified "LLMs can never self-correct" — only the *no-external-feedback* version holds.

**Time-sensitivity:** WebLLM (v0.2.84) and MLC evolve fast. Re-check the `web-llm` repo before relying
on "no speculative decoding," and re-verify the `@atjsh/llmlingua-2` port's maturity before shipping it.

---

*Compiled from two adversarially-verified deep-research passes (217 sub-agents) + a full read of this
repo's extraction/prompt/engine path. Confidence tags reflect verification vote outcomes, not author
opinion.*
