import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { SpUnigramTokenizer } from '../spTokenizer';

// Fixtures (pocket-tts English tokenizer.model + reference encodings produced by Python
// `sentencepiece`). The test verifies our pure-JS Unigram tokenizer matches the reference
// id-for-id. Skipped automatically if the fixtures aren't present.
const here = dirname(fileURLToPath(import.meta.url));
const modelPath = join(here, 'fixtures', 'tokenizer.model');
const refPath = join(here, 'fixtures', 'sp_reference.json');
const hasFixtures = existsSync(modelPath) && existsSync(refPath);

const d = hasFixtures ? describe : describe.skip;

d('SpUnigramTokenizer matches Python sentencepiece', () => {
  const buf = readFileSync(modelPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const ref = JSON.parse(readFileSync(refPath, 'utf8')) as { text: string; ids: number[]; decoded: string }[];
  const tok = new SpUnigramTokenizer();
  tok.load(ab);

  for (const { text, ids } of ref) {
    it(`encodes: ${JSON.stringify(text)}`, () => {
      expect(tok.encodeIds(text)).toEqual(ids);
    });
  }

  it('round-trips decode(encode(text)) back to the input text', () => {
    for (const { text } of ref) {
      expect(tok.decodeIds(tok.encodeIds(text))).toBe(text);
    }
  });
});
