import { describe, it, expect, vi } from 'vitest';
import { sttLoad, sttStartListening, sttMute, sttRelease } from '../sttClient';
import type { WebllmPort } from '../webllmClient';

/** Minimal stand-in for a runtime Port: records outgoing messages and lets a test drive
 *  incoming ones (and a disconnect). */
function makeFakePort() {
  const msgListeners = new Set<(m: unknown) => void>();
  const discListeners = new Set<() => void>();
  const sent: any[] = [];
  const port = {
    onMessage: {
      addListener: (f: (m: unknown) => void) => msgListeners.add(f),
      removeListener: (f: (m: unknown) => void) => msgListeners.delete(f),
    },
    onDisconnect: {
      addListener: (f: () => void) => discListeners.add(f),
      removeListener: (f: () => void) => discListeners.delete(f),
    },
    postMessage: (m: unknown) => sent.push(m),
  } as unknown as WebllmPort;
  return {
    port,
    sent,
    emit: (m: unknown) => [...msgListeners].forEach((f) => f(m)),
    disconnect: () => [...discListeners].forEach((f) => f()),
    listenerCount: () => msgListeners.size,
  };
}

describe('sttLoad', () => {
  it('reports progress, resolves on sttReady, and cleans up its listener', async () => {
    const fake = makeFakePort();
    const onProgress = vi.fn();
    const p = sttLoad(fake.port, onProgress);
    const id = fake.sent[0].id as number;
    expect(fake.sent[0]).toMatchObject({ type: 'sttLoad' });

    fake.emit({ type: 'sttProgress', id: id + 999, progress: 0.1 }); // wrong id → ignored
    fake.emit({ type: 'sttProgress', id, progress: 0.5 });
    fake.emit({ type: 'sttReady', id });
    await expect(p).resolves.toBeUndefined();

    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith(0.5);
    expect(fake.listenerCount()).toBe(0);
  });

  it('rejects on sttError', async () => {
    const fake = makeFakePort();
    const p = sttLoad(fake.port, () => {});
    const id = fake.sent[0].id as number;
    fake.emit({ type: 'sttError', id, message: 'boom' });
    await expect(p).rejects.toThrow('boom');
    expect(fake.listenerCount()).toBe(0);
  });

  it('rejects when the port disconnects mid-load', async () => {
    const fake = makeFakePort();
    const p = sttLoad(fake.port, () => {});
    fake.disconnect();
    await expect(p).rejects.toThrow(/disconnected/i);
    expect(fake.listenerCount()).toBe(0);
  });
});

describe('sttStartListening', () => {
  it('posts sttStart and routes id-matched status/transcript messages to callbacks', () => {
    const fake = makeFakePort();
    const onState = vi.fn();
    const onTranscript = vi.fn();
    const onStopped = vi.fn();
    const session = sttStartListening(fake.port, { mode: 'auto', onState, onTranscript, onStopped });

    expect(fake.sent[0]).toMatchObject({ type: 'sttStart', mode: 'auto', id: session.id });

    fake.emit({ type: 'sttState', id: session.id, state: 'speech' });
    fake.emit({ type: 'sttTranscript', id: session.id, text: 'hello there' });
    fake.emit({ type: 'sttTranscript', id: session.id + 999, text: 'other session' }); // ignored

    expect(onState).toHaveBeenCalledWith('speech');
    expect(onTranscript).toHaveBeenCalledTimes(1);
    expect(onTranscript).toHaveBeenCalledWith('hello there');
  });

  it('stop(flush) posts sttStop, and sttStopped fires onStopped and cleans up', () => {
    const fake = makeFakePort();
    const onStopped = vi.fn();
    const session = sttStartListening(fake.port, { mode: 'ptt', onTranscript: () => {}, onStopped });

    session.stop(true);
    expect(fake.sent.find((m) => m.type === 'sttStop')).toMatchObject({ flush: true });

    fake.emit({ type: 'sttStopped', id: session.id });
    expect(onStopped).toHaveBeenCalledTimes(1);
    expect(fake.listenerCount()).toBe(0);
  });

  it('surfaces sttError without ending the session (it keeps listening)', () => {
    const fake = makeFakePort();
    const onError = vi.fn();
    const onStopped = vi.fn();
    const session = sttStartListening(fake.port, { mode: 'auto', onTranscript: () => {}, onError, onStopped });

    fake.emit({ type: 'sttError', id: session.id, message: 'transcribe failed' });
    expect(onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'transcribe failed' }));
    expect(onStopped).not.toHaveBeenCalled();
    expect(fake.listenerCount()).toBe(1); // still attached
  });

  it('fires onStopped when the port disconnects', () => {
    const fake = makeFakePort();
    const onStopped = vi.fn();
    sttStartListening(fake.port, { mode: 'auto', onTranscript: () => {}, onStopped });
    fake.disconnect();
    expect(onStopped).toHaveBeenCalledTimes(1);
    expect(fake.listenerCount()).toBe(0);
  });
});

describe('sttMute / sttRelease', () => {
  it('post the expected control messages', () => {
    const fake = makeFakePort();
    sttMute(fake.port, true);
    sttRelease(fake.port);
    expect(fake.sent).toEqual([
      { type: 'sttMute', muted: true },
      { type: 'sttRelease' },
    ]);
  });

  it('swallow errors when the port is already gone', () => {
    const port = {
      postMessage: () => { throw new Error('port closed'); },
    } as unknown as WebllmPort;
    expect(() => sttMute(port, false)).not.toThrow();
    expect(() => sttRelease(port)).not.toThrow();
  });
});
