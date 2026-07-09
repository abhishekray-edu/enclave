// Structure-aware page chunking for RAG and summarization. Pure + panel-safe
// (no @mlc-ai/web-llm / @huggingface/transformers imports).
import type { Chunk, PageBlock, PageContent } from './types';
import { estimateTokens } from './prompt';

export interface ChunkOptions {
  /** Preferred chunk size in tokens. */
  targetTokens?: number;
  /** Hard cap; an oversized paragraph is split to stay under this. */
  maxTokens?: number;
  /** Tokens of trailing context prepended to the next chunk so facts survive boundaries. */
  overlapTokens?: number;
}

const DEFAULTS = { targetTokens: 320, maxTokens: 500, overlapTokens: 50 };

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+/).filter((s) => s.trim());
}

/** Split an oversized paragraph into <= maxTokens pieces, on sentence then word boundaries. */
function splitOversized(text: string, maxTokens: number): string[] {
  if (estimateTokens(text) <= maxTokens) return [text];
  const out: string[] = [];
  let cur = '';
  const flush = () => {
    if (cur.trim()) out.push(cur.trim());
    cur = '';
  };
  for (const sentence of splitSentences(text)) {
    if (estimateTokens(sentence) > maxTokens) {
      flush();
      let w = '';
      for (const word of sentence.split(/\s+/)) {
        // A single word longer than the cap (long URL, base64, minified code) is hard-split.
        if (estimateTokens(word) > maxTokens) {
          if (w) {
            out.push(w);
            w = '';
          }
          const size = maxTokens * 4;
          for (let i = 0; i < word.length; i += size) out.push(word.slice(i, i + size));
          continue;
        }
        if (w && estimateTokens(`${w} ${word}`) > maxTokens) {
          out.push(w);
          w = word;
        } else {
          w = w ? `${w} ${word}` : word;
        }
      }
      if (w) out.push(w);
      continue;
    }
    if (cur && estimateTokens(`${cur} ${sentence}`) > maxTokens) flush();
    cur = cur ? `${cur} ${sentence}` : sentence;
  }
  flush();
  return out;
}

/** Last <= n tokens of `text`, on sentence boundaries (for inter-chunk overlap). */
function tailTokens(text: string, n: number): string {
  if (n <= 0) return '';
  const sentences = splitSentences(text);
  let tail = '';
  for (let i = sentences.length - 1; i >= 0; i--) {
    const next = tail ? `${sentences[i]} ${tail}` : sentences[i];
    if (estimateTokens(next) > n) break;
    tail = next;
  }
  return tail;
}

interface Unit {
  text: string;
  heading?: string;
}

function unitsFromBlocks(blocks: PageBlock[]): Unit[] {
  const units: Unit[] = [];
  let heading: string | undefined;
  for (const b of blocks) {
    if (b.type === 'heading') heading = b.text;
    else if (b.text.trim()) units.push({ text: b.text.trim(), heading });
  }
  return units;
}

function unitsFromText(text: string): Unit[] {
  return text
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((t) => ({ text: t }));
}

/**
 * Split a page into ~targetTokens chunks on natural boundaries (headings → paragraphs →
 * sentences), each carrying its nearest heading and a little overlap. Falls back to
 * paragraph-splitting page.textContent when structural blocks are absent.
 */
export function chunkPage(page: PageContent, opts?: ChunkOptions): Chunk[] {
  const { targetTokens, maxTokens, overlapTokens } = { ...DEFAULTS, ...opts };
  const units = page.blocks?.length ? unitsFromBlocks(page.blocks) : unitsFromText(page.textContent);

  const chunks: Chunk[] = [];
  let buf = '';
  let bufHeading: string | undefined;
  let ordinal = 0;
  let prevTail = '';

  const emit = () => {
    const core = buf.trim();
    buf = '';
    if (!core) return;
    const text = prevTail ? `${prevTail}\n${core}` : core;
    chunks.push({
      ordinal: ordinal++,
      text,
      heading: bufHeading,
      tokensEstimate: estimateTokens(text),
    });
    prevTail = tailTokens(core, overlapTokens);
    bufHeading = undefined;
  };

  for (const unit of units) {
    for (const piece of splitOversized(unit.text, maxTokens)) {
      if (!buf) bufHeading = unit.heading;
      if (buf && estimateTokens(`${buf}\n${piece}`) > targetTokens) {
        emit();
        bufHeading = unit.heading;
        buf = piece;
      } else {
        buf = buf ? `${buf}\n${piece}` : piece;
      }
    }
  }
  emit();
  return chunks;
}
