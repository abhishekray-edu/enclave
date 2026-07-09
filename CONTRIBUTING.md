# Contributing to Enclave

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

You'll need **Node 20+** and a Chromium browser with WebGPU.

```bash
git clone https://github.com/abhishekray-edu/enclave.git
cd enclave
npm install
npm run dev        # live-reload dev build → .output/chrome-mv3
```

Load `.output/chrome-mv3` via `chrome://extensions` → Developer mode → **Load unpacked**.

## Checks

Please make sure these pass before opening a PR (CI runs the same):

```bash
npm run compile    # typecheck
npm test           # unit tests (vitest)
npm run build      # production build (Chrome)
```

## Project layout

| Path | What lives there |
|---|---|
| `entrypoints/content.ts` | Page extraction (Readability + deep DOM walk) |
| `entrypoints/sidepanel/` | React UI: chat, task routing, markdown rendering |
| `entrypoints/offscreen/` | Model host: WebLLM worker + embeddings/compression worker |
| `entrypoints/background.ts` | Panel behavior, context menu, offscreen lifecycle |
| `lib/` | The pipeline: prompt building, chunking, retrieval, summarization, tasks |
| `docs/` | ADR + large-page design notes |

Two invariants to know before touching `lib/`:

1. **Bundle split** — `lib/webllm.ts`, `lib/retrieval.ts`, `lib/compress.ts`, and `lib/ortEnv.ts`
   import heavy ML runtimes and must only ever be imported from the offscreen entrypoints,
   never from the side panel.
2. **Prompt safety caps** — no code path may build a single prompt larger than the per-model
   `safePromptTokens` ([lib/webllmClient.ts](lib/webllmClient.ts)). This is a stability limit,
   not a tuning knob — see [docs/large-page-handling.md](docs/large-page-handling.md). It is
   pinned by `lib/__tests__/budgets.test.ts`; if your change trips those tests, rework the
   change rather than the tests.

## Pull requests

- Keep PRs focused; one logical change per PR.
- Follow the existing commit style: `feat: …`, `fix: …`, `docs: …`, `refactor: …`.
- If you change behavior, say how you tested it (page, model, machine).

## Reporting bugs

Please include: browser + version, OS, GPU/RAM, the model selected, and roughly what page you
were on (size/type — never paste sensitive content). Model-load and generation errors from the
side panel are the most useful thing to quote.
