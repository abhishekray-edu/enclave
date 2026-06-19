import type { ChatMessage, Settings } from './types';

/** Thrown when Ollama is unreachable or blocks the request (likely CORS). */
export class OllamaConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OllamaConnectionError';
  }
}

/** List installed model tags via GET /api/tags. */
export async function listModels(endpoint: string): Promise<string[]> {
  let res: Response;
  try {
    res = await fetch(`${endpoint}/api/tags`);
  } catch (e) {
    // A fetch that rejects (vs. returns a non-2xx) is almost always CORS or "server down".
    throw new OllamaConnectionError(
      `Could not reach Ollama at ${endpoint}. Is it running, and is OLLAMA_ORIGINS set to allow this extension?`,
    );
  }
  if (!res.ok) {
    throw new OllamaConnectionError(`Ollama returned ${res.status} for /api/tags`);
  }
  const data = (await res.json()) as { models?: Array<{ name: string }> };
  return (data.models ?? []).map((m) => m.name).sort();
}

export interface ChatStreamParams {
  settings: Settings;
  messages: ChatMessage[];
  signal: AbortSignal;
}

/**
 * Stream a chat completion from POST /api/chat (stream: true).
 * Yields content deltas as they arrive. Parses Ollama's NDJSON (one JSON object per line).
 */
export async function* chatStream({
  settings,
  messages,
  signal,
}: ChatStreamParams): AsyncGenerator<string, void, unknown> {
  let res: Response;
  try {
    res = await fetch(`${settings.endpoint}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal,
      body: JSON.stringify({
        model: settings.model,
        messages,
        stream: true,
        keep_alive: '30m', // keep the model resident for snappy follow-ups
        options: {
          num_ctx: settings.numCtx, // must be set; Ollama defaults to ~4K otherwise
          temperature: settings.temperature,
        },
      }),
    });
  } catch (e) {
    if ((e as Error).name === 'AbortError') return;
    throw new OllamaConnectionError(
      `Could not reach Ollama at ${settings.endpoint}. Is it running, and is OLLAMA_ORIGINS set to allow this extension?`,
    );
  }

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '');
    throw new OllamaConnectionError(
      `Ollama /api/chat failed (${res.status})${detail ? `: ${detail}` : ''}`,
    );
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // NDJSON: process complete lines, keep the trailing partial in the buffer.
      let nl: number;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const obj = JSON.parse(line) as {
          message?: { content?: string };
          done?: boolean;
          error?: string;
        };
        if (obj.error) throw new OllamaConnectionError(obj.error);
        const delta = obj.message?.content;
        if (delta) yield delta;
        if (obj.done) return;
      }
    }
  } catch (e) {
    if ((e as Error).name === 'AbortError') return;
    throw e;
  } finally {
    reader.cancel().catch(() => {});
  }
}
