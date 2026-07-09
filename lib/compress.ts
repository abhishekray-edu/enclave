// Optional prompt compression via LLMLingua-2 (token classification with a tiny BERT-class
// encoder). OFFSCREEN-ONLY: pulls in @atjsh/llmlingua-2 + @huggingface/transformers. Gated
// behind a setting (default off) because the upstream JS port is experimental. The model
// downloads once from the HF CDN then caches, like the other models.
import { LLMLingua2 } from '@atjsh/llmlingua-2';
import { Tiktoken } from 'js-tiktoken/lite';
import o200k_base from 'js-tiktoken/ranks/o200k_base';
import { configureOrtRuntime } from './ortEnv';

// TinyBERT (~57MB) — the smallest compressor, co-resident with the embedder + LLM.
const COMPRESS_MODEL = 'atjsh/llmlingua-2-js-tinybert-meetingbank';

interface Compressor {
  compress: (context: string, opts: { rate: number }) => Promise<string>;
}

let compressorPromise: Promise<Compressor> | null = null;

function init(): Promise<Compressor> {
  configureOrtRuntime();
  const oaiTokenizer = new Tiktoken(o200k_base);
  return LLMLingua2.WithBERTMultilingual(COMPRESS_MODEL, {
    transformerJSConfig: { device: 'wasm', dtype: 'fp32' },
    oaiTokenizer,
  }).then((r) => r.promptCompressor as unknown as Compressor);
}

export function ensureCompressor(): Promise<Compressor> {
  if (!compressorPromise) {
    compressorPromise = init().catch((err) => {
      compressorPromise = null; // allow retry
      throw err;
    });
  }
  return compressorPromise;
}

/** Compress each text to roughly `rate` of its tokens, preserving meaning. Returns inputs
 *  unchanged on failure so the feature degrades gracefully. */
export async function compressTexts(texts: string[], rate: number): Promise<string[]> {
  const pc = await ensureCompressor();
  const out: string[] = [];
  for (const t of texts) {
    try {
      out.push(await pc.compress(t, { rate }));
    } catch {
      out.push(t);
    }
  }
  return out;
}
