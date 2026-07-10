// Engine-side WebLLM helpers. Imports the heavy @mlc-ai/web-llm runtime, so this
// module must ONLY be imported by the offscreen document — never the side panel.
import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
  type InitProgressReport,
  type MLCEngineInterface,
  type WebWorkerMLCEngine,
} from '@mlc-ai/web-llm';
import type { ChatMessage } from './types';
import type { GenerateOptions } from './webllmClient';

/** Map our portable options onto the WebLLM request fields. */
function genFields(options: GenerateOptions) {
  return {
    temperature: options.temperature,
    ...(options.maxTokens != null ? { max_tokens: options.maxTokens } : {}),
    ...(options.responseFormat ? { response_format: options.responseFormat } : {}),
    ...(options.enableThinking === false ? { extra_body: { enable_thinking: false } } : {}),
    ...(options.frequencyPenalty ? { frequency_penalty: options.frequencyPenalty } : {}),
  };
}

/** Create (download + initialise) an in-browser engine for the given model, hosted in the
 *  given dedicated worker so weight deserialization and generation never block the renderer
 *  main thread (which the side panel usually shares — a busy main thread freezes its UI).
 *  `contextWindowSize` overrides the build's default (4096); bounded by the model build's
 *  compiled maximum and available GPU memory. */
export function createWebllmEngine(
  worker: Worker,
  modelId: string,
  contextWindowSize: number,
  onProgress: (report: InitProgressReport) => void,
): Promise<WebWorkerMLCEngine> {
  return CreateWebWorkerMLCEngine(
    worker,
    modelId,
    { appConfig: prebuiltAppConfig, initProgressCallback: onProgress },
    { context_window_size: contextWindowSize },
  );
}

/** True when the model's weights are already in the browser cache (no download needed).
 *  Used to gate pre-warming: we never auto-download gigabytes without an explicit ask. */
export async function isModelCached(modelId: string): Promise<boolean> {
  try {
    return await hasModelInCache(modelId, prebuiltAppConfig);
  } catch {
    return false;
  }
}

export interface WebllmStreamParams {
  engine: MLCEngineInterface;
  messages: ChatMessage[];
  options: GenerateOptions;
  signal: AbortSignal;
}

/** Stream a completion from an in-browser engine, yielding content deltas. */
export async function* chatStreamWebllm({
  engine,
  messages,
  options,
  signal,
}: WebllmStreamParams): AsyncGenerator<string, void, unknown> {
  if (signal.aborted) return;
  const onAbort = () => void engine.interruptGenerate();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    const chunks = await engine.chat.completions.create({ messages, stream: true, ...genFields(options) });
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) yield delta;
    }
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}

/** Run a single non-streaming completion (JSON extraction, map-reduce calls), returning the
 *  full content. Throws 'extract:truncated' when the model hit the token limit mid-output. */
export async function chatCompleteWebllm({
  engine,
  messages,
  options,
  signal,
}: WebllmStreamParams): Promise<string> {
  if (signal.aborted) return '';
  const onAbort = () => void engine.interruptGenerate();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    const res = await engine.chat.completions.create({ messages, stream: false, ...genFields(options) });
    const choice = res.choices[0];
    // Hitting max_tokens is fatal only for JSON output (truncated JSON is unparseable).
    // Bounded prose (map/reduce summary calls) is clipped on purpose — return it as-is.
    if (choice?.finish_reason === 'length' && options.responseFormat?.type === 'json_object') {
      throw new Error('extract:truncated');
    }
    return choice?.message?.content ?? '';
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}
