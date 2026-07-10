// Barge-in echo guard. With the mic kept live during TTS playback (so the user can interrupt),
// the mic can also pick up the assistant's own spoken voice. If a transcript that arrives during
// playback closely matches what was just spoken, it is almost certainly that echo — not the user
// — so the panel discards it instead of treating it as an interruption.
//
// This is the last of three layers (browser echo-cancellation + the VAD ducking profile are the
// first two); it only has to catch echo the first two let through.

export const ECHO_OVERLAP_THRESHOLD = 0.75;
export const ECHO_MIN_TOKENS = 2;

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

/** Fraction (0..1) of the transcript's tokens that also appear in the recently spoken text. */
export function echoOverlap(transcript: string, spoken: string[]): number {
  const tTokens = tokenize(transcript);
  if (!tTokens.length) return 0;
  const spokenSet = new Set<string>();
  for (const s of spoken) for (const w of tokenize(s)) spokenSet.add(w);
  if (!spokenSet.size) return 0;
  let matched = 0;
  for (const w of tTokens) if (spokenSet.has(w)) matched++;
  return matched / tTokens.length;
}

/** True when a transcript is probably the assistant's own TTS echoed back through the mic. */
export function isLikelyEcho(transcript: string, spoken: string[]): boolean {
  if (tokenize(transcript).length < ECHO_MIN_TOKENS) return false;
  return echoOverlap(transcript, spoken) >= ECHO_OVERLAP_THRESHOLD;
}
