// Engine-side WebLLM helpers. Imports the heavy @mlc-ai/web-llm runtime, so this
// module must ONLY be imported by the offscreen document — never the side panel.
import {
  CreateWebWorkerMLCEngine,
  hasModelInCache,
  prebuiltAppConfig,
  type ChatCompletionMessageParam,
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

/** A message's text plus any attached files, folded into one prompt block. */
function textWithFiles(m: ChatMessage): string {
  if (!m.files?.length) return m.content;
  const blocks = m.files.map((f) => `--- ATTACHED FILE: ${f.name} ---\n${f.text}\n--- END OF FILE ---`);
  return [m.content, ...blocks].join('\n\n');
}

/** Map portable ChatMessages onto WebLLM's OpenAI-style params: fold file attachments into
 *  the text, and turn images into image_url content parts. Only the NEWEST image-bearing
 *  message keeps its images — each image costs ~2k tokens of the vision build's 4k context,
 *  so re-sending every historical image would overflow it almost immediately. On a
 *  non-vision model images are replaced by a note, so the model doesn't answer as if it saw
 *  something it never did. */
export function toEngineMessages(messages: ChatMessage[], vision: boolean): ChatCompletionMessageParam[] {
  const lastWithImages = messages.reduce((acc, m, i) => (m.images?.length ? i : acc), -1);
  const firstWithImages = messages.findIndex((m) => m.images?.length);
  return messages.map((m, i): ChatCompletionMessageParam => {
    const text = textWithFiles(m);
    if (m.role === 'user' && m.images?.length) {
      if (vision && i === lastWithImages) {
        // When an earlier image was dropped, the conversation still contains the model's own
        // description of it — which a small VLM will happily keep answering from. Anchor it
        // to the pixels explicitly so the newest image wins over the old words.
        const label =
          firstWithImages !== lastWithImages
            ? '(A NEW image is attached above. It replaces every earlier image — answer about THIS image, not from earlier descriptions.)\n'
            : '';
        return {
          role: 'user',
          content: [
            ...m.images.map((url) => ({ type: 'image_url' as const, image_url: { url } })),
            { type: 'text' as const, text: label + text },
          ],
        };
      }
      const note = vision
        ? '[The image that was attached here has been replaced by a newer one and is gone — you cannot see it anymore. Never answer about it from memory; only the image on the newest message is visible.]'
        : '[The user attached an image, but the current model cannot see images. Tell the user to switch to the vision model in Settings if the question needs the image.]';
      return { role: 'user', content: `${text}\n\n${note}` };
    }
    // Rebuild plain messages too: ChatMessage carries UI-only fields (error/structured)
    // that must not leak into the engine request.
    return { role: m.role, content: text } as ChatCompletionMessageParam;
  });
}

export interface WebllmStreamParams {
  engine: MLCEngineInterface;
  messages: ChatMessage[];
  options: GenerateOptions;
  signal: AbortSignal;
  /** The loaded model understands images (webllmModel(id).vision). Default false. */
  vision?: boolean;
}

/** Stream a completion from an in-browser engine, yielding content deltas. */
export async function* chatStreamWebllm({
  engine,
  messages,
  options,
  signal,
  vision,
}: WebllmStreamParams): AsyncGenerator<string, void, unknown> {
  if (signal.aborted) return;
  const onAbort = () => void engine.interruptGenerate();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    const chunks = await engine.chat.completions.create({
      messages: toEngineMessages(messages, vision === true),
      stream: true,
      ...genFields(options),
    });
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
  vision,
}: WebllmStreamParams): Promise<string> {
  if (signal.aborted) return '';
  const onAbort = () => void engine.interruptGenerate();
  signal.addEventListener('abort', onAbort, { once: true });
  try {
    const res = await engine.chat.completions.create({
      messages: toEngineMessages(messages, vision === true),
      stream: false,
      ...genFields(options),
    });
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
