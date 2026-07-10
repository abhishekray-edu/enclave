// Vendors the Silero VAD v5 model into public/vad/ so the extension ships it (hermetic builds —
// no runtime CDN fetch for the voice-activity detector). Patterned on prepare-tts-assets.mjs.
//
// Licensing (see THIRD_PARTY_NOTICES.md): Silero VAD is MIT (Silero Team), which permits
// redistribution. We commit the ~2.2 MB ONNX binary directly; this script just refreshes it.
//
// The companion speech model (Moonshine, MIT — Useful Sensors) is NOT vendored: transformers.js
// fetches its weights from the Hugging Face CDN on first use and caches them in the browser,
// exactly like the RAG embedding model.
//
// Usage:  node scripts/prepare-vad-assets.mjs
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

// Official Silero v5 export, mirrored by onnx-community (the input takes a raw 512-sample frame
// at 16 kHz directly and manages context internally via the `state` tensor — verified against
// the model's own I/O; see entrypoints/offscreen/stt.worker.ts).
const SRC = 'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx';
const OUT_DIR = path.resolve('public/vad');
const OUT_FILE = path.join(OUT_DIR, 'silero_vad_v5.onnx');

async function main() {
  const res = await fetch(SRC);
  if (!res.ok) throw new Error(`fetch silero_vad: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(OUT_DIR, { recursive: true });
  await writeFile(OUT_FILE, buf);
  console.log(`Wrote ${OUT_FILE} (${(buf.length / 1e6).toFixed(2)} MB)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
