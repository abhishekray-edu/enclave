// Incremental sentence segmentation over a growing string. Fed the full text-so-far on every
// LLM token, it returns only the sentences newly completed since the last call — so the panel
// can hand each finished sentence to TTS while generation continues, instead of waiting for the
// whole reply (see App.tsx streaming loop + lib/ttsClient.ts ttsSpeakStream).
//
// This is deliberately NOT the TTS worker's splitIntoBestSentences (tts.worker.ts): that one is
// tokenizer-coupled and runs on a complete string. Here we need cheap, prefix-stable, streaming
// boundary detection with a frontier that never re-emits text.

const MERGE_MIN_CHARS = 25; // hold sentences shorter than this and merge them into the next one
const CLOSERS = `"')]”’»`; // closing quotes/brackets that may trail terminal punctuation

/** A completed sentence plus the offset (into the pushed text) just past its end. `endOffset`
 *  lets the caller map a spoken sentence back to how much of the displayed text it covers. */
export interface SentencePiece {
  text: string;
  endOffset: number;
}

export class SentenceStream {
  private frontier = 0; // index in the text up to which sentences have been consumed
  private held = ''; // a too-short sentence buffered to merge with the next
  private lastText = '';

  /** Feed the full text so far; returns sentences completed since the previous call. */
  push(fullText: string): SentencePiece[] {
    this.lastText = fullText;
    const out: SentencePiece[] = [];
    const n = fullText.length;
    let segStart = this.frontier;
    let i = this.frontier;

    while (i < n) {
      const c = fullText[i];
      const isTerminal = c === '.' || c === '!' || c === '?' || c === '…';
      if (isTerminal) {
        let j = i + 1;
        while (j < n && CLOSERS.includes(fullText[j])) j++;
        // A boundary requires whitespace AND a following non-space char: this defers the last
        // sentence until the next one starts (or flush()), and keeps a period inside "3.5" or an
        // abbreviation from splitting mid-token (no whitespace immediately follows).
        if (j < n && isSpace(fullText[j])) {
          let k = j;
          while (k < n && isSpace(fullText[k])) k++;
          if (k < n) {
            this.accept(fullText.slice(segStart, k), k, out);
            segStart = k;
            i = k;
            continue;
          }
          break; // trailing whitespace with nothing after yet — wait for more tokens
        }
      }
      // A blank line ends a paragraph (and thus a sentence), even without terminal punctuation.
      if (c === '\n' && i + 1 < n && fullText[i + 1] === '\n') {
        let k = i;
        while (k < n && fullText[k] === '\n') k++;
        if (k < n) {
          this.accept(fullText.slice(segStart, k), k, out);
          segStart = k;
          i = k;
          continue;
        }
      }
      i++;
    }

    this.frontier = segStart;
    return out;
  }

  /** End of stream: return the trailing remainder (plus anything held) as a final sentence. */
  flush(): SentencePiece | null {
    const remainder = (this.held + this.lastText.slice(this.frontier)).trim();
    this.held = '';
    this.frontier = this.lastText.length;
    if (!remainder) return null;
    return { text: remainder, endOffset: this.lastText.length };
  }

  private accept(piece: string, endOffset: number, out: SentencePiece[]) {
    const candidate = this.held + piece;
    if (candidate.trim().length < MERGE_MIN_CHARS) {
      this.held = candidate; // too short to speak on its own — merge into the next sentence
      return;
    }
    out.push({ text: candidate.trim(), endOffset });
    this.held = '';
  }
}

function isSpace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r';
}
