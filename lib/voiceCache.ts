// Where the voice models' weights live in Cache Storage, plus cache-only probes over them.
// Source of truth for the offscreen workers' download/cache locations (they import these
// constants), and for the background worker's startup warm-up, which may only load weights
// that are already on disk — it must never start a download the user didn't ask for.
// PANEL/BACKGROUND-SAFE: imports no ONNX/transformers code (bundle-split invariant).

/** Self-hosted HF repo the pocket-tts assets are fetched from (see tts.worker.ts).
 *  NOTE: set this to YOUR public Hugging Face repo (created from tts-assets/). Update the
 *  owner if your HF username differs from the GitHub org. */
export const TTS_ASSET_BASE = 'https://huggingface.co/team-edt/enclave-tts/resolve/main/';

/** Cache Storage bucket tts.worker.ts stores the assets in. */
export const TTS_CACHE_NAME = 'enclave-tts-v1';

// mimi_encoder is intentionally absent — it's only needed for cloning a voice from audio.
// Bytes are approximate (for the download progress bar); real Content-Length is used when present.
export const TTS_ASSETS: { file: string; bytes: number }[] = [
  { file: 'bundle.json', bytes: 40_000 },
  { file: 'text_conditioner_int8.onnx', bytes: 16_400_000 },
  { file: 'flow_lm_main_int8.onnx', bytes: 76_300_000 },
  { file: 'flow_lm_flow_int8.onnx', bytes: 10_000_000 },
  { file: 'mimi_decoder_int8.onnx', bytes: 22_700_000 },
  { file: 'voices.bin', bytes: 6_200_000 }, // alba only (CC-BY-4.0); was 52 MB with all voices
  { file: 'tokenizer.model', bytes: 100_000 },
  { file: 'bos_before_voice.npy', bytes: 5_000 },
];

/** Moonshine model id (see stt.worker.ts). transformers.js fetches its files from the HF CDN
 *  and auto-caches them in its default Cache Storage bucket. */
export const MOONSHINE_MODEL = 'onnx-community/moonshine-base-ONNX';
const TRANSFORMERS_CACHE_NAME = 'transformers-cache';

/** True when EVERY pocket-tts asset is already in Cache Storage. A partial cache (interrupted
 *  download) doesn't count: loading from it would fetch the rest, and a startup warm-up must
 *  never touch the network. */
export async function ttsWeightsCached(): Promise<boolean> {
  try {
    if (!(await caches.has(TTS_CACHE_NAME))) return false;
    const cache = await caches.open(TTS_CACHE_NAME);
    const hits = await Promise.all(TTS_ASSETS.map((a) => cache.match(TTS_ASSET_BASE + a.file)));
    return hits.every(Boolean);
  } catch {
    return false;
  }
}

/** True when Moonshine's encoder+decoder weights are already in transformers.js's cache.
 *  Probes by URL substring so a dtype/file rename in stt.worker.ts fails safe: the probe
 *  returns false and the startup warm-up simply skips STT (never downloads). */
export async function sttWeightsCached(): Promise<boolean> {
  try {
    if (!(await caches.has(TRANSFORMERS_CACHE_NAME))) return false;
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const urls = (await cache.keys()).map((req) => req.url);
    const mine = urls.filter((u) => u.includes(MOONSHINE_MODEL));
    return mine.some((u) => u.includes('encoder_model')) && mine.some((u) => u.includes('decoder_model'));
  } catch {
    return false;
  }
}
