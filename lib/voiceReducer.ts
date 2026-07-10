// Pure state machine for hands-free voice mode, extracted from the panel so its transitions are
// unit-testable (see lib/__tests__/voiceReducer.test.ts). The panel owns the side effects
// (mic mute, submit, TTS); this only decides the next UI/loop state.
//
// The loop: listening → speech (user talking) → transcribing → thinking (LLM) → speaking (TTS)
// → listening. `off` is the resting state. The cardinal rule the tests pin down: an event must
// never strand the loop in `thinking` — every terminal path (reply spoken, error, empty, toggle
// off) resolves back to `listening` or `off`.

export type VoiceState = 'off' | 'listening' | 'speech' | 'transcribing' | 'thinking' | 'speaking';

export type VoiceAction =
  /** Turn voice mode off from any state. */
  | { type: 'toggleOff' }
  /** A status update from the STT worker (only meaningful during the listening phase). */
  | { type: 'sttState'; state: 'listening' | 'speech' | 'transcribing' | 'muted' }
  /** A finished transcript arrived → hand off to the LLM. */
  | { type: 'transcript' }
  /** The reply is ready and about to be spoken. */
  | { type: 'speak' }
  /** Speech finished, or generation produced nothing / errored → resume listening. */
  | { type: 'resume' };

/** Phases the panel drives directly; STT worker status is ignored while in them. */
function isPanelOwned(state: VoiceState): boolean {
  return state === 'thinking' || state === 'speaking';
}

export function voiceReducer(state: VoiceState, action: VoiceAction): VoiceState {
  switch (action.type) {
    case 'toggleOff':
      return 'off';
    case 'sttState':
      // Ignored unless we're actively listening — a stray worker status must never pull the loop
      // out of `off`, `thinking`, or `speaking`.
      if (state === 'off' || isPanelOwned(state)) return state;
      if (action.state === 'listening') return 'listening';
      if (action.state === 'speech') return 'speech';
      if (action.state === 'transcribing') return 'transcribing';
      return state; // 'muted' — keep the current phase
    case 'transcript':
      // A transcript that arrives while off (or already thinking/speaking) is ignored.
      if (state === 'off' || isPanelOwned(state)) return state;
      return 'thinking';
    case 'speak':
      return state === 'off' ? 'off' : 'speaking';
    case 'resume':
      return state === 'off' ? 'off' : 'listening';
    default:
      return state;
  }
}
