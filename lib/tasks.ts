// Task definitions that drive the generation pipeline (system prompt, sampling, output
// mode, retrieval/summarization eligibility). Panel-side; no @mlc-ai/web-llm import.
import type { Settings } from './types';

export type TaskKind = 'ask' | 'summarize' | 'explain' | 'extract';
export type OutputMode = 'markdown' | 'json';

/** A JSON-schema-constrained extraction preset (rendered with XGrammar/json_object). */
export interface ExtractionSchema {
  id: string;
  label: string;
  /** JSON Schema object; stringified for the model's response_format.schema. */
  schema: Record<string, unknown>;
  /** Natural-language instruction telling the model what to pull out. */
  instruction: string;
}

export interface TaskSpec {
  kind: TaskKind;
  /** Human-readable chat-bubble text for this action (empty for free-form 'ask'). */
  label: string;
  /** Task system prompt; '' falls back to settings.systemPrompt. */
  systemPrompt: string;
  /** Per-task temperature; 'user' falls back to settings.temperature. */
  temperature: number | 'user';
  outputMode: OutputMode;
  /** Cap on generated tokens (bounds JSON + summaries; avoids json whitespace stalls). */
  maxTokens?: number;
  /** May use retrieval (RAG) on large pages. */
  allowRag: boolean;
  /** Large pages route through map-reduce instead of truncation. */
  supportsMapReduce: boolean;
}

export const TASKS: Record<'ask' | 'summarize' | 'explain', TaskSpec> = {
  ask: {
    kind: 'ask',
    label: '',
    systemPrompt: '',
    temperature: 'user',
    outputMode: 'markdown',
    allowRag: true,
    supportsMapReduce: false,
  },
  summarize: {
    kind: 'summarize',
    label: 'Summarize this page',
    systemPrompt:
      'You summarize web pages. Output concise, faithful bullet points grounded ONLY in the ' +
      'provided text. Do not add information that is not present. No preamble.',
    temperature: 0.2,
    outputMode: 'markdown',
    maxTokens: 700,
    allowRag: false,
    supportsMapReduce: true,
  },
  explain: {
    kind: 'explain',
    label: 'Explain the selection',
    systemPrompt:
      'You explain the passage the user selected, in simple, clear terms. Define jargon. Be ' +
      'accurate and brief. Use the surrounding page only for context.',
    temperature: 0.3,
    outputMode: 'markdown',
    allowRag: false,
    supportsMapReduce: false,
  },
};

export const EXTRACTION_SCHEMAS: ExtractionSchema[] = [
  {
    id: 'article-meta',
    label: 'Article metadata',
    instruction: 'Extract the title, author(s), publication date, and 3-6 key topics.',
    schema: {
      type: 'object',
      properties: {
        title: { type: 'string' },
        authors: { type: 'array', items: { type: 'string' } },
        published: { type: ['string', 'null'] },
        topics: { type: 'array', items: { type: 'string' } },
      },
      required: ['title', 'topics'],
    },
  },
  {
    id: 'key-facts',
    label: 'Key facts',
    instruction: 'Extract the most important factual claims as label/value pairs.',
    schema: {
      type: 'object',
      properties: {
        facts: {
          type: 'array',
          items: {
            type: 'object',
            properties: { label: { type: 'string' }, value: { type: 'string' } },
            required: ['label', 'value'],
          },
        },
      },
      required: ['facts'],
    },
  },
  {
    id: 'contacts',
    label: 'Contacts',
    instruction: 'Extract any names, emails, phone numbers, and organizations mentioned.',
    schema: {
      type: 'object',
      properties: {
        emails: { type: 'array', items: { type: 'string' } },
        phones: { type: 'array', items: { type: 'string' } },
        people: { type: 'array', items: { type: 'string' } },
        orgs: { type: 'array', items: { type: 'string' } },
      },
    },
  },
];

/** Build a TaskSpec for an extraction run from a chosen schema preset. */
export function extractTask(s: ExtractionSchema): TaskSpec {
  return {
    kind: 'extract',
    label: `Extract: ${s.label}`,
    systemPrompt:
      'You extract structured data from the page. Return ONLY a JSON object matching the schema. ' +
      'Use null or empty arrays for fields not present in the text. Do not invent values. ' +
      s.instruction,
    temperature: 0,
    outputMode: 'json',
    maxTokens: 1024,
    allowRag: true,
    supportsMapReduce: false,
  };
}

export function resolveTemperature(spec: TaskSpec, s: Settings): number {
  return spec.temperature === 'user' ? s.temperature : spec.temperature;
}
