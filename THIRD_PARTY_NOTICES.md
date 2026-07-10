# Third-Party Notices

Enclave incorporates the following third-party software and models. Their licenses are
reproduced or linked below. Enclave itself is MIT-licensed (see LICENSE).

## Text-to-speech (the "Speak" feature)

### pocket-tts — model weights
- Source: https://huggingface.co/kyutai/pocket-tts (Kyutai)
- **License: CC-BY-4.0** — https://creativecommons.org/licenses/by/4.0/
- Attribution: "pocket-tts" by Kyutai, used under CC-BY-4.0.
- **Changes made:** exported from safetensors to ONNX and int8-quantized (see below); only the
  `alba` voice is used; text is tokenized with an independent pure-JS SentencePiece
  implementation.
- Use complies with Kyutai's acceptable-use policy: a single fixed preset voice reads the
  assistant's own generated replies; no voice cloning or impersonation.

### pocket-tts voice "alba"
- Source: https://huggingface.co/kyutai/tts-voices (`alba-mackenna`)
- **License: CC-BY-4.0.** Other voices in the upstream bundle (e.g. CC-BY-NC ones) are **not**
  redistributed or used.

### pocket-tts-onnx-export — ONNX export tooling & inference approach
- Source: https://github.com/KevinAHM/pocket-tts-onnx-export (KevinAHM)
- **License: MIT.** The offscreen TTS worker and audio worklet are adapted from this project.

MIT License (KevinAHM/pocket-tts-onnx-export; and original pocket-tts code © Kyutai):

```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

### onnxruntime-web
- Source: https://github.com/microsoft/onnxruntime — **License: MIT** (© Microsoft). Runs the
  TTS, STT, and VAD models on the CPU (WebAssembly).

## Speech-to-text (the microphone / "voice mode" feature)

### Moonshine — speech-recognition model
- Source: https://huggingface.co/UsefulSensors/moonshine (Useful Sensors); ONNX build
  `onnx-community/moonshine-base-ONNX`.
- **License: MIT** — used to transcribe microphone audio on-device (English). Weights are
  fetched once from the Hugging Face CDN at first use and cached in the browser; not
  redistributed by this project.

### Silero VAD — voice-activity detector
- Source: https://github.com/snakers4/silero-vad (Silero Team); ONNX mirror
  `onnx-community/silero-vad`.
- **License: MIT.** The ~2.2 MB v5 ONNX model is vendored in `public/vad/` and redistributed
  under the MIT license (refresh with `node scripts/prepare-vad-assets.mjs`).

### @ricky0123/vad-web — VAD endpointing semantics
- Source: https://github.com/ricky0123/vad — **License: MIT.** The frame-level endpointing
  state machine in `lib/vadFrameProcessor.ts` is a port of this project's logic (thresholds,
  redemption frames, pre-speech padding); no code is bundled.

## Also used elsewhere in Enclave
- @mlc-ai/web-llm (Apache-2.0) — in-browser LLM engine.
- @huggingface/transformers / Transformers.js (Apache-2.0) — RAG embeddings and Moonshine STT.
- @mozilla/readability (Apache-2.0), react (MIT), react-markdown (MIT), highlight.js (BSD-3),
  js-tiktoken (MIT), @atjsh/llmlingua-2 (MIT).
- KaTeX (MIT) — math rendering (`remark-math` / `rehype-katex`, both MIT); the KaTeX fonts are
  bundled under the SIL Open Font License 1.1.
- lucide (ISC) — the inline SVG icon geometry in `entrypoints/sidepanel/icons.tsx` is
  hand-copied from lucide's icon set; no package is bundled.

Model weights for the LLMs are downloaded from their respective providers' CDNs under their own
licenses at first use, and are not redistributed by this project.
