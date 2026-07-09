// Single-slot model load manager for the offscreen document. Pure orchestration with
// injected effects, so the concurrency rules are unit-testable without WebGPU or workers.
//
// Rules:
// - At most one engine is resident (its weights + KV cache are preallocated GPU memory)
//   and at most one load is in flight.
// - Loads never queue. A request for the model+context already loading JOINS it; a request
//   for a different one CANCELS it — the newest request wins. Nothing would be gained by
//   queueing: a multi-GB download would block the model the user actually wants now, and
//   canceling a download loses no work — fetched weight shards stay in the browser cache,
//   so the download resumes where it stopped.
// - Cancellation terminates the loading worker outright; waiters are rejected with a
//   sentinel message the UI treats as silence, not failure.

export type LoadProgressFn = (text: string, progress: number) => void;

/** Sentinel rejection messages for loads that ended on purpose rather than failing. */
export const LOAD_CANCELED = 'model-load-canceled';
export const LOAD_SUPERSEDED = 'model-load-superseded';

/** True for rejections produced by canceling/superseding a load (not real failures). */
export function isLoadInterruption(e: unknown): boolean {
  const message = e instanceof Error ? e.message : String(e);
  return message === LOAD_CANCELED || message === LOAD_SUPERSEDED;
}

/** Effects the loader orchestrates, injected so tests can fake them. */
export interface EngineHost<E, W> {
  spawnWorker(): W;
  terminateWorker(worker: W): void;
  createEngine(worker: W, model: string, ctx: number, onProgress: LoadProgressFn): Promise<E>;
  unloadEngine(engine: E): Promise<void>;
}

export interface ModelLoader<E> {
  /** The resident engine, if any. */
  readonly engine: E | null;
  /** The load currently in flight, if any. */
  readonly loading: { model: string; ctx: number } | null;
  /** Ensure the model+context is resident, joining or superseding any in-flight load. */
  ensure(model: string, ctx: number, onProgress: LoadProgressFn): Promise<void>;
  /** Cancel the in-flight load if it matches; returns whether one was canceled. */
  cancel(model: string, ctx: number): boolean;
}

export function createModelLoader<E, W>(host: EngineHost<E, W>): ModelLoader<E> {
  let engine: E | null = null;
  let engineWorker: W | null = null;
  let loadedKey: string | null = null;
  /** Serializes old-engine teardown so a new load never overlaps the GPU release. */
  let teardown: Promise<void> = Promise.resolve();

  interface Load {
    key: string;
    model: string;
    ctx: number;
    listeners: Set<LoadProgressFn>;
    canceled: boolean;
    promise: Promise<void>;
    cancel: (message: string) => void;
  }
  let active: Load | null = null;

  const keyOf = (model: string, ctx: number) => `${model}|${ctx}`;

  async function ensure(model: string, ctx: number, onProgress: LoadProgressFn): Promise<void> {
    const key = keyOf(model, ctx);

    // Join an identical load already in flight (progress fans out to every requester).
    if (active && active.key === key) {
      const current = active;
      current.listeners.add(onProgress);
      try {
        await current.promise;
        return;
      } finally {
        current.listeners.delete(onProgress);
      }
    }

    // A different model is mid-load: the newest request wins.
    if (active) active.cancel(LOAD_SUPERSEDED);

    if (engine && loadedKey === key) return;

    const worker = host.spawnWorker();
    let rejectCancel!: (e: Error) => void;
    const cancelPromise = new Promise<never>((_, reject) => {
      rejectCancel = reject;
    });

    const load: Load = {
      key,
      model,
      ctx,
      listeners: new Set([onProgress]),
      canceled: false,
      promise: Promise.resolve(),
      cancel(message: string) {
        load.canceled = true;
        // Terminating the worker abandons the download/load; the engine promise never
        // settles after this, so waiters are released via the rejection instead.
        host.terminateWorker(worker);
        if (active === load) active = null;
        rejectCancel(new Error(message));
      },
    };

    const run = async () => {
      // Wait out any previous teardown, then evict the resident engine. The fields are
      // cleared before the async release starts, so no concurrent caller can observe
      // (and hand out) an engine that is being torn down.
      await teardown;
      if (load.canceled) return;
      if (engine) {
        const oldEngine = engine;
        const oldWorker = engineWorker;
        engine = null;
        engineWorker = null;
        loadedKey = null;
        teardown = (async () => {
          try {
            await host.unloadEngine(oldEngine);
          } catch {
            /* ignore */
          }
          if (oldWorker != null) host.terminateWorker(oldWorker);
        })();
        await teardown;
        if (load.canceled) return;
      }
      let created: E;
      try {
        created = await host.createEngine(worker, model, ctx, (text, progress) => {
          for (const listener of load.listeners) listener(text, progress);
        });
      } catch (e) {
        host.terminateWorker(worker);
        throw e;
      }
      if (load.canceled) {
        // Canceled in the same beat the engine finished — drop it with its worker.
        host.terminateWorker(worker);
        return;
      }
      engine = created;
      engineWorker = worker;
      loadedKey = key;
    };

    load.promise = Promise.race([run(), cancelPromise]).finally(() => {
      if (active === load) active = null;
    });
    active = load;
    await load.promise;
  }

  return {
    get engine() {
      return engine;
    },
    get loading() {
      return active ? { model: active.model, ctx: active.ctx } : null;
    },
    ensure,
    cancel(model: string, ctx: number): boolean {
      if (active && active.key === keyOf(model, ctx)) {
        active.cancel(LOAD_CANCELED);
        return true;
      }
      return false;
    },
  };
}
