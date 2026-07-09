// In-browser retrieval: embeds page chunks with a tiny Transformers.js model (WebGPU when
// available — at ~22M params its VRAM cost is negligible next to the LLM — else CPU/WASM),
// caches vectors in IndexedDB, and does brute-force cosine top-k. OFFSCREEN-ONLY: importing
// this pulls in @huggingface/transformers, so it must never be imported by the side panel bundle.
import { pipeline, type FeatureExtractionPipeline } from '@huggingface/transformers';
import { configureOrtRuntime } from './ortEnv';
import type { Chunk, RetrievedChunk } from './types';
import type { LoadProgress } from './webllmClient';

// all-MiniLM-L6-v2 (~22M params). Downloads once from the HF CDN then caches in the browser,
// exactly like the LLM weights. Isolated as one constant to ease swapping to jina-v2-small.
const EMBED_MODEL = 'Xenova/all-MiniLM-L6-v2';
const EMBED_DIM = 384;
const EMBED_BATCH = 16;
const MAX_CACHED_PAGES = 50;

type Progress = (p: LoadProgress) => void;
const noop: Progress = () => {};

let embedderPromise: Promise<FeatureExtractionPipeline> | null = null;
// Which backend produced the vectors. Part of the cache key: vectors from different
// precisions/backends must never be cosine-compared against each other.
let embedVariant = 'unknown';

export function ensureEmbedder(onProgress: Progress = noop): Promise<FeatureExtractionPipeline> {
  if (!embedderPromise) {
    configureOrtRuntime();
    const progress_callback = (e: unknown) => {
      const info = e as { file?: string; progress?: number };
      if (typeof info.progress === 'number') onProgress({ text: info.file ?? 'embedder', progress: info.progress / 100 });
    };
    // Prefer WebGPU (~22M params — trivial VRAM, several times faster than single-threaded
    // WASM); fall back to CPU/WASM when WebGPU is unavailable in this worker.
    embedderPromise = (async () => {
      try {
        const p = (await pipeline('feature-extraction', EMBED_MODEL, {
          device: 'webgpu',
          dtype: 'fp16',
          progress_callback,
        })) as FeatureExtractionPipeline;
        embedVariant = 'webgpu-fp16';
        return p;
      } catch {
        const p = (await pipeline('feature-extraction', EMBED_MODEL, {
          device: 'wasm',
          progress_callback,
        })) as FeatureExtractionPipeline;
        embedVariant = 'wasm-q8';
        return p;
      }
    })().catch((err) => {
      embedderPromise = null; // allow retry after a failed load
      throw err;
    });
  }
  return embedderPromise;
}

/** Cache identity for stored vectors: model + backend/precision that produced them. */
function modelKey(): string {
  return `${EMBED_MODEL}#${embedVariant}`;
}

async function embedTexts(texts: string[], onProgress: Progress = noop): Promise<Float32Array[]> {
  const extractor = await ensureEmbedder(onProgress);
  const vectors: Float32Array[] = [];
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const batch = texts.slice(i, i + EMBED_BATCH);
    const out = await extractor(batch, { pooling: 'mean', normalize: true });
    for (const v of out.tolist() as number[][]) vectors.push(Float32Array.from(v));
  }
  return vectors;
}

function flatten(vectors: Float32Array[]): Float32Array {
  const flat = new Float32Array(vectors.length * EMBED_DIM);
  vectors.forEach((v, i) => flat.set(v, i * EMBED_DIM));
  return flat;
}

// ---- IndexedDB persistence (survives panel close; only the warm embedder is lost) ----
const DB_NAME = 'enclave-rag';
const STORE = 'pages';

interface PageIndexRecord {
  cacheKey: string;
  url: string;
  contentHash: string;
  model: string;
  createdAt: number;
  chunks: Chunk[];
  vectors: Float32Array; // flat, EMBED_DIM * chunks.length
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'cacheKey' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function idbGet(cacheKey: string): Promise<PageIndexRecord | undefined> {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(cacheKey);
        req.onsuccess = () => resolve(req.result as PageIndexRecord | undefined);
        req.onerror = () => reject(req.error);
      }),
  );
}

/** Store a record, reap other hashes of the same url, and LRU-trim the store. */
async function idbPutAndReap(rec: PageIndexRecord): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    store.put(rec);
    const all = store.getAll();
    all.onsuccess = () => {
      const recs = (all.result as PageIndexRecord[]) ?? [];
      // Drop stale versions of the same page (different contentHash).
      for (const r of recs) if (r.url === rec.url && r.cacheKey !== rec.cacheKey) store.delete(r.cacheKey);
      // LRU cap by createdAt.
      const live = recs.filter((r) => !(r.url === rec.url && r.cacheKey !== rec.cacheKey));
      if (live.length > MAX_CACHED_PAGES) {
        live
          .sort((a, b) => a.createdAt - b.createdAt)
          .slice(0, live.length - MAX_CACHED_PAGES)
          .forEach((r) => store.delete(r.cacheKey));
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// In-session cache so repeated queries on the same page skip IndexedDB entirely.
const memCache = new Map<string, { chunks: Chunk[]; vectors: Float32Array }>();
const cacheKeyOf = (url: string, hash: string) => `${url}::${hash}`;

/** Ensure an embedded index exists for {url, contentHash}; reuse cache when fresh. */
export async function buildOrLoadIndex(
  url: string,
  contentHash: string,
  chunks: Chunk[],
  onProgress: Progress = noop,
): Promise<{ chunkCount: number; fromCache: boolean }> {
  const key = cacheKeyOf(url, contentHash);
  if (memCache.has(key)) return { chunkCount: memCache.get(key)!.chunks.length, fromCache: true };

  // Resolve the embedder first so modelKey() (backend + precision) is known — a record
  // embedded by a different backend must not be reused against this backend's query vectors.
  await ensureEmbedder(onProgress);
  const existing = await idbGet(key);
  if (existing && existing.model === modelKey() && existing.chunks.length) {
    memCache.set(key, { chunks: existing.chunks, vectors: existing.vectors });
    return { chunkCount: existing.chunks.length, fromCache: true };
  }

  const vectors = flatten(await embedTexts(chunks.map((c) => c.text), onProgress));
  memCache.set(key, { chunks, vectors });
  await idbPutAndReap({ cacheKey: key, url, contentHash, model: modelKey(), createdAt: Date.now(), chunks, vectors });
  return { chunkCount: chunks.length, fromCache: false };
}

/** Embed the query and return the top-k chunks by cosine similarity (vectors are normalized). */
export async function retrieve(
  url: string,
  contentHash: string,
  query: string,
  topK: number,
): Promise<RetrievedChunk[]> {
  const key = cacheKeyOf(url, contentHash);
  let entry = memCache.get(key);
  if (!entry) {
    await ensureEmbedder();
    const rec = await idbGet(key);
    if (rec && rec.model === modelKey()) {
      entry = { chunks: rec.chunks, vectors: rec.vectors };
      memCache.set(key, entry);
    }
  }
  if (!entry) throw new Error('No index for this page — build it first.');

  const [q] = await embedTexts([query]);
  const { chunks, vectors } = entry;
  const scored: RetrievedChunk[] = chunks.map((chunk, i) => {
    let dot = 0;
    const off = i * EMBED_DIM;
    for (let d = 0; d < EMBED_DIM; d++) dot += q[d] * vectors[off + d];
    return { ...chunk, score: dot };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.max(1, topK));
}
