// Pure-JS SentencePiece **Unigram** tokenizer for the pocket-tts text tokenizer.model.
//
// Why not the upstream sentencepiece.js? That is an emscripten *embind* build, and embind's
// createNamedFunction uses `new Function(...)`, i.e. it needs 'unsafe-eval'. MV3 extensions
// cannot grant 'unsafe-eval' (only 'wasm-unsafe-eval'), so it throws a CSP violation at
// runtime. This reimplements exactly the pieces of SentencePiece the model needs — no eval,
// no wasm — and lets us drop the 4 MB module.
//
// The model was verified to be: model_type=UNIGRAM, byte_fallback=true, normalizer='identity'
// (no NFKC / no precompiled charsmap), add_dummy_prefix=true, remove_extra_whitespaces=false.
// So "normalization" is just: prepend ▁ and turn spaces into ▁ (U+2581). Encoding is the
// standard Unigram Viterbi (max sum of piece log-probs) with a single-character unknown
// fallback that byte-expands to <0xNN> pieces. Cross-checked id-for-id against Python
// `sentencepiece` (see lib/__tests__/spTokenizer.test.ts).

const SPACE = '▁'; // ▁ meta symbol
const UNK_PENALTY = 10.0; // SentencePiece kUnkPenalty

interface Vocab {
  pieces: string[]; // id -> surface string
  pieceToId: Map<string, { id: number; score: number }>; // matchable pieces
  byteToId: (b: number) => number; // 0..255 -> id of <0xNN>
  isByteId: (id: number) => boolean;
  byteOfId: (id: number) => number; // id -> byte value 0..255
  unkId: number;
  unkScore: number;
  maxLen: number; // longest piece in codepoints
}

/** Parse a SentencePiece ModelProto (only the fields we need: the repeated pieces). */
function parseModel(buf: ArrayBuffer): Vocab {
  const b = new Uint8Array(buf);
  const dv = new DataView(buf);
  let i = 0;
  const readVarint = (): number => {
    let shift = 0;
    let result = 0;
    for (;;) {
      const x = b[i++];
      result += (x & 0x7f) * Math.pow(2, shift);
      if (!(x & 0x80)) break;
      shift += 7;
    }
    return result;
  };

  const pieces: string[] = [];
  const scores: number[] = [];
  const types: number[] = [];

  while (i < b.length) {
    const tag = readVarint();
    const field = tag >> 3;
    const wire = tag & 7;
    if (field === 1 && wire === 2) {
      // SentencePiece message { string piece=1; float score=2; Type type=3 (default NORMAL) }
      const len = readVarint();
      const end = i + len;
      let piece = '';
      let score = 0;
      let type = 1; // default NORMAL
      while (i < end) {
        const t2 = readVarint();
        const f2 = t2 >> 3;
        const w2 = t2 & 7;
        if (f2 === 1 && w2 === 2) {
          const l2 = readVarint();
          piece = new TextDecoder().decode(b.subarray(i, i + l2));
          i += l2;
        } else if (f2 === 2 && w2 === 5) {
          score = dv.getFloat32(i, true);
          i += 4;
        } else if (f2 === 3 && w2 === 0) {
          type = readVarint();
        } else {
          // skip unknown
          if (w2 === 2) { const l = readVarint(); i += l; }
          else if (w2 === 0) readVarint();
          else if (w2 === 5) i += 4;
          else if (w2 === 1) i += 8;
          else break;
        }
      }
      pieces.push(piece);
      scores.push(score);
      types.push(type);
    } else if (wire === 2) {
      const len = readVarint();
      i += len;
    } else if (wire === 0) {
      readVarint();
    } else if (wire === 5) {
      i += 4;
    } else if (wire === 1) {
      i += 8;
    } else {
      break;
    }
  }

  // Piece types: NORMAL(1 or absent→default), UNKNOWN(2), CONTROL(3), USER_DEFINED(4), BYTE(6).
  const pieceToId = new Map<string, { id: number; score: number }>();
  const byteId = new Array<number>(256).fill(-1);
  let unkId = 0;
  let minScore = Infinity;
  let maxLen = 1;

  for (let id = 0; id < pieces.length; id++) {
    const type = types[id];
    const piece = pieces[id];
    if (type === 2) unkId = id; // UNKNOWN
    if (type === 6) {
      // BYTE piece "<0xNN>"
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
      if (m) byteId[parseInt(m[1], 16)] = id;
      continue;
    }
    if (type === 3 || type === 2) continue; // CONTROL / UNKNOWN aren't matchable surfaces
    // NORMAL / USER_DEFINED participate in matching.
    pieceToId.set(piece, { id, score: scores[id] });
    if (scores[id] < minScore) minScore = scores[id];
    const cpLen = Array.from(piece).length;
    if (cpLen > maxLen) maxLen = cpLen;
  }
  if (!Number.isFinite(minScore)) minScore = 0;

  // Byte pieces <0x00>..<0xFF> are contiguous in id order, so byteId[0] is the id of <0x00>.
  const firstByteId = byteId[0]; // id of <0x00> (-1 if the model has no byte pieces)
  const lastByteId = byteId[255];
  return {
    pieces,
    pieceToId,
    byteToId: (bt) => byteId[bt],
    isByteId: (id) => firstByteId >= 0 && id >= firstByteId && id <= lastByteId,
    byteOfId: (id) => id - firstByteId,
    unkId,
    unkScore: minScore - UNK_PENALTY,
    maxLen,
  };
}

export class SpUnigramTokenizer {
  private vocab: Vocab | null = null;

  load(buf: ArrayBuffer): void {
    this.vocab = parseModel(buf);
  }

  /** Encode text to piece ids (Unigram Viterbi + byte fallback). No BOS/EOS added. */
  encodeIds(text: string): number[] {
    const v = this.vocab;
    if (!v) throw new Error('Tokenizer not loaded');
    // add_dummy_prefix + escape whitespace (identity normalization otherwise).
    const norm = SPACE + text.replace(/ /g, SPACE);
    const chars = Array.from(norm); // codepoints
    const n = chars.length;

    const NEG = -Infinity;
    const bestScore = new Float64Array(n + 1).fill(NEG);
    bestScore[0] = 0;
    const backStart = new Int32Array(n + 1).fill(-1);
    const backId = new Int32Array(n + 1).fill(-1);
    const backUnk = new Uint8Array(n + 1); // 1 => this edge is an unknown (byte-fallback) char

    for (let i = 0; i < n; i++) {
      if (bestScore[i] === NEG) continue;
      let hasSingle = false;
      const maxL = Math.min(v.maxLen, n - i);
      let sub = '';
      for (let len = 1; len <= maxL; len++) {
        sub += chars[i + len - 1];
        const entry = v.pieceToId.get(sub);
        if (entry !== undefined) {
          const s = bestScore[i] + entry.score;
          if (s > bestScore[i + len]) {
            bestScore[i + len] = s;
            backStart[i + len] = i;
            backId[i + len] = entry.id;
            backUnk[i + len] = 0;
          }
          if (len === 1) hasSingle = true;
        }
      }
      // Guarantee progress: if no single-character piece matched here, add an unknown edge
      // covering exactly one character (SentencePiece semantics).
      if (!hasSingle) {
        const s = bestScore[i] + v.unkScore;
        if (s > bestScore[i + 1]) {
          bestScore[i + 1] = s;
          backStart[i + 1] = i;
          backId[i + 1] = v.unkId;
          backUnk[i + 1] = 1;
        }
      }
    }

    // Backtrack.
    const nodes: { start: number; end: number; id: number; unk: boolean }[] = [];
    let pos = n;
    while (pos > 0) {
      const start = backStart[pos];
      nodes.push({ start, end: pos, id: backId[pos], unk: backUnk[pos] === 1 });
      pos = start;
    }
    nodes.reverse();

    // Expand unknown nodes to their UTF-8 bytes via <0xNN> pieces.
    const ids: number[] = [];
    const enc = new TextEncoder();
    for (const node of nodes) {
      if (node.unk) {
        const surface = chars.slice(node.start, node.end).join('');
        for (const byte of enc.encode(surface)) {
          const bid = v.byteToId(byte);
          ids.push(bid >= 0 ? bid : v.unkId);
        }
      } else {
        ids.push(node.id);
      }
    }
    return ids;
  }

  /** Decode piece ids back to text (byte pieces → UTF-8, ▁ → space, drop dummy prefix). */
  decodeIds(ids: number[]): string {
    const v = this.vocab;
    if (!v) throw new Error('Tokenizer not loaded');
    const out: string[] = [];
    let byteBuf: number[] = [];
    const flush = () => {
      if (byteBuf.length) {
        out.push(new TextDecoder().decode(new Uint8Array(byteBuf)));
        byteBuf = [];
      }
    };
    for (const id of ids) {
      if (v.isByteId(id)) {
        byteBuf.push(v.byteOfId(id));
      } else {
        flush();
        out.push(v.pieces[id] ?? '');
      }
    }
    flush();
    let s = out.join('').replace(/▁/g, ' ');
    if (s.startsWith(' ')) s = s.slice(1); // add_dummy_prefix space
    return s;
  }
}
