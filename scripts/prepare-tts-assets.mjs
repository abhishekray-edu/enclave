// Prepares the pocket-tts voice-model assets for self-hosting in your OWN public Hugging Face
// repo (so the extension never depends on a third-party Space, and licensing is clean).
//
// It downloads the ONNX models + tokenizer + config from KevinAHM's ONNX export, and extracts
// ONLY the CC-BY-4.0 "alba" voice from voices.bin (dropping the bundle's CC-BY-NC voices).
// Output goes to ./tts-assets/, ready to upload to https://huggingface.co/<you>/enclave-tts.
//
// Licensing (see THIRD_PARTY_NOTICES.md): the model weights are Kyutai pocket-tts under
// CC-BY-4.0, which permits redistribution with attribution + indication of changes (here: ONNX
// export + int8 quantization by KevinAHM, and extraction of a single voice). The ONNX export
// code is MIT (KevinAHM). Your HF repo must carry the attribution in scripts/tts-repo-README.md.
//
// Usage:  node scripts/prepare-tts-assets.mjs
import { mkdir, writeFile, readFile, copyFile } from 'node:fs/promises';
import path from 'node:path';

const SRC = 'https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/main/onnx/english_2026-04/';
const OUT = path.resolve('tts-assets');
const VOICE = 'alba'; // CC-BY-4.0 (kyutai/tts-voices → alba-mackenna). Do NOT switch to a
                      // CC-BY-NC voice (expresso/ears) without honoring the NC restriction.

// Files copied verbatim (models, tokenizer, config). mimi_encoder is omitted (voice cloning only).
const VERBATIM = [
  'bundle.json',
  'tokenizer.model',
  'bos_before_voice.npy',
  'text_conditioner_int8.onnx',
  'flow_lm_main_int8.onnx',
  'flow_lm_flow_int8.onnx',
  'mimi_decoder_int8.onnx',
];

async function fetchBuf(file) {
  const res = await fetch(SRC + file);
  if (!res.ok) throw new Error(`fetch ${file}: ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

/** Extract a single voice from a PTVB1 voices.bin, returning a new PTVB1 buffer with just it. */
function extractVoice(buf, wanted) {
  if (buf.toString('latin1', 0, 5) !== 'PTVB1') throw new Error('bad voices.bin magic');
  let off = 5;
  const count = buf.readUInt32LE(off); off += 4;
  for (let vi = 0; vi < count; vi++) {
    const start = off;
    const nlen = buf.readUInt16LE(off); off += 2;
    const name = buf.toString('utf8', off, off + nlen); off += nlen;
    const tcount = buf.readUInt16LE(off); off += 2;
    for (let ti = 0; ti < tcount; ti++) {
      const klen = buf.readUInt16LE(off); off += 2; off += klen;
      off += 1; // dtype
      const rank = buf.readUInt8(off); off += 1;
      off += 4 * rank;
      const blen = buf.readUInt32LE(off); off += 4; off += blen;
    }
    if (name === wanted) {
      const record = buf.subarray(start, off);
      const header = Buffer.alloc(9);
      header.write('PTVB1', 0, 'latin1');
      header.writeUInt32LE(1, 5);
      return Buffer.concat([header, record]);
    }
  }
  throw new Error(`voice "${wanted}" not found (have: parsed ${count})`);
}

await mkdir(OUT, { recursive: true });
for (const f of VERBATIM) {
  process.stdout.write(`↓ ${f} … `);
  await writeFile(path.join(OUT, f), await fetchBuf(f));
  console.log('ok');
}
process.stdout.write(`↓ voices.bin (extracting "${VOICE}") … `);
const alba = extractVoice(await fetchBuf('voices.bin'), VOICE);
await writeFile(path.join(OUT, 'voices.bin'), alba);
console.log(`ok (${(alba.length / 1e6).toFixed(1)} MB, ${VOICE} only)`);

// Drop the HF repo card (README) alongside, so the upload carries its license + attribution.
await copyFile(path.resolve('scripts/tts-repo-README.md'), path.join(OUT, 'README.md'));

console.log(`\nDone → ${OUT}`);
console.log('Next: create a PUBLIC (ungated) HF repo and upload the contents of tts-assets/,');
console.log('then set TTS_ASSET_BASE in entrypoints/offscreen/tts.worker.ts to that repo URL.');
