// Normalize the LaTeX math delimiters small local models emit into the single form
// remark-math understands ($$...$$). Models trained on math (Qwen especially) freely mix
// \[...\] / \(...\) and bare \boxed{...}; remark-math only recognizes $ / $$, so without this
// those render as literal backslash-bracket noise. We deliberately run with
// singleDollarTextMath:false in Markdown.tsx so prose like "it costs $5 and $10" is never
// misparsed as math — every real math span is therefore converted to $$...$$ here.
//
// Code regions (fenced blocks and inline spans) are left untouched: a code sample may contain
// \[ or $ that must not be reinterpreted as math.

/** Split text into alternating non-code / code segments, preserving order. Only non-code
 *  segments are transformed by the caller. Handles ``` fenced blocks and `inline` spans. */
function splitCodeRegions(md: string): { text: string; code: boolean }[] {
  const segments: { text: string; code: boolean }[] = [];
  // Fenced blocks (``` or ~~~) first, then inline backtick spans within the prose gaps.
  const fenceRe = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  const pushProse = (text: string) => {
    if (!text) return;
    const inlineRe = /(`+)([\s\S]*?)\1/g;
    let li = 0;
    let im: RegExpExecArray | null;
    while ((im = inlineRe.exec(text)) !== null) {
      if (im.index > li) segments.push({ text: text.slice(li, im.index), code: false });
      segments.push({ text: im[0], code: true });
      li = im.index + im[0].length;
    }
    if (li < text.length) segments.push({ text: text.slice(li), code: false });
  };
  while ((m = fenceRe.exec(md)) !== null) {
    if (m.index > lastIndex) pushProse(md.slice(lastIndex, m.index));
    segments.push({ text: m[0], code: true });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < md.length) pushProse(md.slice(lastIndex));
  return segments;
}

function convertProse(text: string): string {
  let out = text;
  // \[ ... \] → $$ ... $$ (display) and \( ... \) → $$ ... $$ (inline). The [\s\S] class spans
  // newlines so multi-line display math converts intact.
  out = out.replace(/\\\[([\s\S]*?)\\\]/g, (_all, inner) => `$$${inner}$$`);
  out = out.replace(/\\\(([\s\S]*?)\\\)/g, (_all, inner) => `$$${inner}$$`);
  // Bare \boxed{...} not already inside a $$ span: wrap it so KaTeX renders the box. Balance the
  // braces so nested {} inside the argument are captured.
  out = wrapBareBoxed(out);
  return out;
}

/** Wrap any \boxed{...} that is not already within a $$...$$ math span. */
function wrapBareBoxed(text: string): string {
  let result = '';
  let i = 0;
  let inMath = false;
  while (i < text.length) {
    if (text.startsWith('$$', i)) {
      inMath = !inMath;
      result += '$$';
      i += 2;
      continue;
    }
    if (!inMath && text.startsWith('\\boxed{', i)) {
      const start = i;
      i += '\\boxed{'.length;
      let depth = 1;
      while (i < text.length && depth > 0) {
        if (text[i] === '{') depth++;
        else if (text[i] === '}') depth--;
        i++;
      }
      result += `$$${text.slice(start, i)}$$`;
      continue;
    }
    result += text[i];
    i++;
  }
  return result;
}

/** Convert \[...\], \(...\) and bare \boxed{} into $$...$$, leaving code regions untouched. */
export function normalizeMathDelimiters(md: string): string {
  if (!md.includes('\\[') && !md.includes('\\(') && !md.includes('\\boxed')) return md;
  return splitCodeRegions(md)
    .map((seg) => (seg.code ? seg.text : convertProse(seg.text)))
    .join('');
}
