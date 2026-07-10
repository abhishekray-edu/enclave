# Tokenizer test fixtures

`spTokenizer.test.ts` verifies our pure-JS SentencePiece Unigram tokenizer (`lib/spTokenizer.ts`)
matches the reference `sentencepiece` library id-for-id. It needs two files here:

- `tokenizer.model` — the pocket-tts English tokenizer (**gitignored**; not redistributed, as
  the pocket-tts model weights are subject to upstream Kyutai / Hugging Face terms).
- `sp_reference.json` — reference encodings (committed; just integers, no model data).

The test **auto-skips** when `tokenizer.model` is absent, so CI stays green without it.

## To run the test locally

```bash
# 1. fetch the tokenizer (browser fetch follows the HF LFS redirect; curl needs -L)
curl -sL "https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/main/onnx/english_2026-04/tokenizer.model" \
  -o lib/__tests__/fixtures/tokenizer.model

# 2. (optional) regenerate the reference with Python sentencepiece
python3 -m venv /tmp/spvenv && /tmp/spvenv/bin/pip install sentencepiece
/tmp/spvenv/bin/python - <<'PY'
import sentencepiece as spm, json
sp = spm.SentencePieceProcessor(model_file="lib/__tests__/fixtures/tokenizer.model")
tests = ["Hello world.", "Great question. I will walk you through it step by step.",
         "The quick brown fox jumps over the lazy dog.", "It's 2026 — TTS runs locally!",
         "Enclave: nothing ever leaves your machine.", "café naïve résumé"]
json.dump([{"text": t, "ids": sp.encode(t, out_type=int), "decoded": sp.decode(sp.encode(t, out_type=int))}
           for t in tests], open("lib/__tests__/fixtures/sp_reference.json", "w"), ensure_ascii=False, indent=1)
PY

npx vitest run lib/__tests__/spTokenizer.test.ts
```
