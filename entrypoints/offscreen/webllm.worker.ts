// Dedicated worker hosting the WebLLM engine. The offscreen document (and the side panel,
// which usually shares its renderer main thread) stays responsive while multi-GB weights
// deserialize and while generation runs. Driven by WebWorkerMLCEngine on the main side.
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm';

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg: MessageEvent) => handler.onmessage(msg);
