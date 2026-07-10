import { describe, it, expect } from 'vitest';
import { voiceReducer, type VoiceState } from '../voiceReducer';

const ALL_STATES: VoiceState[] = ['off', 'listening', 'speech', 'transcribing', 'thinking', 'speaking'];

describe('voiceReducer', () => {
  it('toggleOff returns to off from every state', () => {
    for (const s of ALL_STATES) {
      expect(voiceReducer(s, { type: 'toggleOff' })).toBe('off');
    }
  });

  it('ignores a transcript that arrives while off', () => {
    expect(voiceReducer('off', { type: 'transcript' })).toBe('off');
  });

  it('a transcript while listening hands off to thinking', () => {
    expect(voiceReducer('listening', { type: 'transcript' })).toBe('thinking');
    expect(voiceReducer('speech', { type: 'transcript' })).toBe('thinking');
  });

  it('never re-enters thinking from a transcript while already thinking or speaking', () => {
    expect(voiceReducer('thinking', { type: 'transcript' })).toBe('thinking');
    expect(voiceReducer('speaking', { type: 'transcript' })).toBe('speaking');
  });

  it('resume (reply spoken / empty / error) always returns to listening unless off', () => {
    expect(voiceReducer('thinking', { type: 'resume' })).toBe('listening');
    expect(voiceReducer('speaking', { type: 'resume' })).toBe('listening');
    expect(voiceReducer('off', { type: 'resume' })).toBe('off');
  });

  it('speak transitions to speaking unless off', () => {
    expect(voiceReducer('thinking', { type: 'speak' })).toBe('speaking');
    expect(voiceReducer('off', { type: 'speak' })).toBe('off');
  });

  it('reflects STT worker status only during the listening phase', () => {
    expect(voiceReducer('listening', { type: 'sttState', state: 'speech' })).toBe('speech');
    expect(voiceReducer('speech', { type: 'sttState', state: 'transcribing' })).toBe('transcribing');
    expect(voiceReducer('transcribing', { type: 'sttState', state: 'listening' })).toBe('listening');
  });

  it('ignores STT worker status while off, thinking, or speaking (panel owns those phases)', () => {
    expect(voiceReducer('off', { type: 'sttState', state: 'speech' })).toBe('off');
    expect(voiceReducer('thinking', { type: 'sttState', state: 'listening' })).toBe('thinking');
    expect(voiceReducer('speaking', { type: 'sttState', state: 'speech' })).toBe('speaking');
  });

  it("'muted' status keeps the current phase", () => {
    expect(voiceReducer('listening', { type: 'sttState', state: 'muted' })).toBe('listening');
  });
});
