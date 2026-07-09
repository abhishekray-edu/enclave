// Dedicated worker for the CPU-side ML: page embedding/retrieval (Transformers.js) and
// LLMLingua compression. Runs off the renderer main thread so the side panel (which usually
// shares that thread with the offscreen document) stays clickable while a page indexes.
// Speaks the same message shapes as the panel⇄offscreen port; main.ts relays verbatim.
import type { Chunk } from '@/lib/types';

interface IndexMsg {
  type: 'index';
  id: number;
  url: string;
  contentHash: string;
  chunks: Chunk[];
}
interface RetrieveMsg {
  type: 'retrieve';
  id: number;
  url: string;
  contentHash: string;
  query: string;
  topK: number;
}
interface CompressMsg {
  type: 'compress';
  id: number;
  texts: string[];
  rate: number;
}
export type MlWorkerRequest = IndexMsg | RetrieveMsg | CompressMsg;

self.onmessage = async (e: MessageEvent<MlWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'index') {
      const { buildOrLoadIndex } = await import('@/lib/retrieval');
      const { chunkCount, fromCache } = await buildOrLoadIndex(msg.url, msg.contentHash, msg.chunks, (report) =>
        self.postMessage({ type: 'embedProgress', id: msg.id, report }),
      );
      self.postMessage({ type: 'indexed', id: msg.id, chunkCount, fromCache });
    } else if (msg.type === 'retrieve') {
      const { retrieve } = await import('@/lib/retrieval');
      const results = await retrieve(msg.url, msg.contentHash, msg.query, msg.topK);
      self.postMessage({ type: 'retrieved', id: msg.id, results });
    } else if (msg.type === 'compress') {
      const { compressTexts } = await import('@/lib/compress');
      const texts = await compressTexts(msg.texts, msg.rate);
      self.postMessage({ type: 'compressed', id: msg.id, texts });
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: err instanceof Error ? err.message : String(err) });
  }
};
