---
license: cc-by-4.0
tags:
  - text-to-speech
  - onnx
  - pocket-tts
---

# Enclave TTS assets (pocket-tts, ONNX)

Voice-model assets used by the [Enclave](https://github.com/abhishekray-edu/enclave) browser
extension to synthesize speech **locally, on-device**. Hosted here only so the extension can
fetch them once and cache them; nothing about how they're used is changed.

## Attribution & license

These weights are **[Kyutai pocket-tts](https://huggingface.co/kyutai/pocket-tts)**, licensed
**[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)**. This repository redistributes
them under the same license, with attribution to Kyutai.

**Changes made to the original:**
- Exported from the official safetensors weights to **ONNX** and **int8-quantized**, using
  [KevinAHM/pocket-tts-onnx-export](https://github.com/KevinAHM/pocket-tts-onnx-export)
  (export code: MIT).
- **Only the `alba` voice** is included (from
  [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) → `alba-mackenna`, CC-BY-4.0).
  The other voices from the upstream bundle — some of which are CC-BY-**NC** — are **not**
  redistributed here.

## Acceptable use

Per Kyutai's model terms: use must comply with applicable law and must not involve voice
impersonation/cloning without consent, misinformation, or other harmful use. Enclave uses a
single fixed preset voice to read the assistant's own generated replies aloud — no cloning.

## Contents

`bundle.json`, `tokenizer.model`, `bos_before_voice.npy`,
`text_conditioner_int8.onnx`, `flow_lm_main_int8.onnx`, `flow_lm_flow_int8.onnx`,
`mimi_decoder_int8.onnx`, `voices.bin` (alba only).

Regenerate with `node scripts/prepare-tts-assets.mjs` in the Enclave repo.
