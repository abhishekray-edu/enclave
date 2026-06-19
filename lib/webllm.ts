// Engine-side WebLLM helpers. Imports the heavy @mlc-ai/web-llm runtime, so this
// module must ONLY be imported by the offscreen document — never the side panel.
import {
  CreateMLCEngine,
  prebuiltAppConfig,
  type InitProgressReport,
  type MLCEngine,
} from '@mlc-ai/web-llm';
import type { ChatMessage } from './types';

/** Create (download + initialise) an in-browser engine for the given model.
 *  `contextWindowSize` overrides the build's default (4096); bounded by the model
 *  build's compiled maximum (e.g. 40960 for Qwen3 4B) and available GPU memory. */
export function createWebllmEngine(
  modelId: string,
  contextWindowSize: number,
  onProgress: (report: InitProgressReport) => void,
): Promise<MLCEngine> {
  return CreateMLCEngine(
    modelId,
    { appConfig: prebuiltAppConfig, initProgressCallback: onProgress },
    { context_window_size: contextWindowSize },
  );
}

export interface WebllmStreamParams {
  engine: MLCEngine;
  messages: ChatMessage[];
  temperature: number;
  signal: AbortSignal;
}

/** Stream a completion from an in-browser engine, yielding content deltas. */
export async function* chatStreamWebllm({
  engine,
  messages,
  temperature,
  signal,
}: WebllmStreamParams): AsyncGenerator<string, void, unknown> {
  if (signal.aborted) return;
  const onAbort = () => void engine.interruptGenerate();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    const chunks = await engine.chat.completions.create({
      messages,
      stream: true,
      temperature,
    });
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) yield delta;
    }
  } finally {
    signal.removeEventListener('abort', onAbort);
  }
}
