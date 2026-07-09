import { describe, expect, it } from 'vitest';
import {
  createModelLoader,
  isLoadInterruption,
  LOAD_CANCELED,
  LOAD_SUPERSEDED,
  type EngineHost,
  type LoadProgressFn,
} from '../modelLoader';

interface FakeEngine {
  id: number;
  unloaded: boolean;
}

interface FakeWorker {
  id: number;
  terminated: boolean;
}

interface Creation {
  worker: FakeWorker;
  model: string;
  ctx: number;
  onProgress: LoadProgressFn;
  resolve: (e: FakeEngine) => void;
  reject: (e: unknown) => void;
}

/** Host whose engine creations stay pending until the test resolves them. */
function makeHost() {
  let nextId = 0;
  const workers: FakeWorker[] = [];
  const creations: Creation[] = [];
  const host: EngineHost<FakeEngine, FakeWorker> = {
    spawnWorker() {
      const worker = { id: nextId++, terminated: false };
      workers.push(worker);
      return worker;
    },
    terminateWorker(worker) {
      worker.terminated = true;
    },
    createEngine(worker, model, ctx, onProgress) {
      return new Promise<FakeEngine>((resolve, reject) => {
        creations.push({ worker, model, ctx, onProgress, resolve, reject });
      });
    },
    async unloadEngine(engine) {
      engine.unloaded = true;
    },
  };
  const makeEngine = (): FakeEngine => ({ id: nextId++, unloaded: false });
  return { host, workers, creations, makeEngine };
}

const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe('createModelLoader', () => {
  it('loads a model and exposes the engine', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const pending = loader.ensure('A', 8192, () => {});
    await tick();
    expect(loader.loading).toEqual({ model: 'A', ctx: 8192 });
    const engine = makeEngine();
    creations[0].resolve(engine);
    await pending;
    expect(loader.engine).toBe(engine);
    expect(loader.loading).toBeNull();
  });

  it('joins an identical in-flight load instead of starting another', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const seenA: number[] = [];
    const seenB: number[] = [];
    const first = loader.ensure('A', 8192, (_t, p) => seenA.push(p));
    const second = loader.ensure('A', 8192, (_t, p) => seenB.push(p));
    await tick();
    expect(creations).toHaveLength(1);
    creations[0].onProgress('fetching', 0.5);
    creations[0].resolve(makeEngine());
    await Promise.all([first, second]);
    expect(seenA).toEqual([0.5]);
    expect(seenB).toEqual([0.5]);
  });

  it('a request for a different model supersedes the in-flight load', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const first = loader.ensure('A', 8192, () => {});
    await tick();
    const second = loader.ensure('B', 8192, () => {});
    await expect(first).rejects.toThrow(LOAD_SUPERSEDED);
    expect(creations[0].worker.terminated).toBe(true);
    await tick();
    expect(creations).toHaveLength(2);
    expect(creations[1].model).toBe('B');
    const engine = makeEngine();
    creations[1].resolve(engine);
    await second;
    expect(loader.engine).toBe(engine);
  });

  it('a context change alone also supersedes (context is part of the load identity)', async () => {
    const { host, creations } = makeHost();
    const loader = createModelLoader(host);
    const first = loader.ensure('A', 4096, () => {});
    await tick();
    void loader.ensure('A', 8192, () => {}).catch(() => {});
    await expect(first).rejects.toThrow(LOAD_SUPERSEDED);
    await tick();
    expect(creations.map((c) => c.ctx)).toEqual([4096, 8192]);
  });

  it('cancel() rejects waiters with the canceled sentinel and terminates the worker', async () => {
    const { host, creations } = makeHost();
    const loader = createModelLoader(host);
    const pending = loader.ensure('A', 8192, () => {});
    await tick();
    expect(loader.cancel('B', 8192)).toBe(false);
    expect(loader.cancel('A', 4096)).toBe(false);
    expect(loader.cancel('A', 8192)).toBe(true);
    await expect(pending).rejects.toSatisfy(isLoadInterruption);
    await expect(pending).rejects.toThrow(LOAD_CANCELED);
    expect(creations[0].worker.terminated).toBe(true);
    expect(loader.loading).toBeNull();
    expect(loader.engine).toBeNull();
  });

  it('evicts the resident engine before installing a different one', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const loadA = loader.ensure('A', 8192, () => {});
    await tick();
    const engineA = makeEngine();
    creations[0].resolve(engineA);
    await loadA;
    const workerA = creations[0].worker;

    const loadB = loader.ensure('B', 8192, () => {});
    await tick();
    expect(engineA.unloaded).toBe(true);
    expect(workerA.terminated).toBe(true);
    expect(loader.engine).toBeNull();
    const engineB = makeEngine();
    creations[1].resolve(engineB);
    await loadB;
    expect(loader.engine).toBe(engineB);
  });

  it('re-ensuring the resident model is a no-op', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const loadA = loader.ensure('A', 8192, () => {});
    await tick();
    creations[0].resolve(makeEngine());
    await loadA;
    await loader.ensure('A', 8192, () => {});
    expect(creations).toHaveLength(1);
  });

  it('a failed load rejects, terminates its worker, and allows a retry', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const first = loader.ensure('A', 8192, () => {});
    await tick();
    creations[0].reject(new Error('device lost'));
    await expect(first).rejects.toThrow('device lost');
    expect(creations[0].worker.terminated).toBe(true);
    expect(loader.loading).toBeNull();

    const retry = loader.ensure('A', 8192, () => {});
    await tick();
    expect(creations).toHaveLength(2);
    const engine = makeEngine();
    creations[1].resolve(engine);
    await retry;
    expect(loader.engine).toBe(engine);
  });

  it('canceling a superseding load before eviction keeps the resident engine intact', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const loadA = loader.ensure('A', 8192, () => {});
    await tick();
    const engineA = makeEngine();
    creations[0].resolve(engineA);
    await loadA;

    // Cancel B synchronously, before its load reaches the eviction step.
    const loadB = loader.ensure('B', 8192, () => {});
    expect(loader.cancel('B', 8192)).toBe(true);
    await expect(loadB).rejects.toThrow(LOAD_CANCELED);
    await tick();
    expect(loader.engine).toBe(engineA);
    expect(engineA.unloaded).toBe(false);

    // The resident model is still instantly available.
    await loader.ensure('A', 8192, () => {});
    expect(creations).toHaveLength(1);
  });

  it('an engine that finishes loading after cancellation is dropped, not installed', async () => {
    const { host, creations, makeEngine } = makeHost();
    const loader = createModelLoader(host);
    const pending = loader.ensure('A', 8192, () => {});
    await tick();
    loader.cancel('A', 8192);
    await expect(pending).rejects.toThrow(LOAD_CANCELED);
    creations[0].resolve(makeEngine());
    await tick();
    expect(loader.engine).toBeNull();
    expect(creations[0].worker.terminated).toBe(true);
  });
});
