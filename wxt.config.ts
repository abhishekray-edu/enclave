import { defineConfig } from 'wxt';
import tailwindcss from '@tailwindcss/vite';
import { cp, mkdir } from 'node:fs/promises';
import path from 'node:path';

// onnxruntime-web (the Transformers.js backend used for RAG embeddings and compression)
// dynamic-imports its wasm runtime from a CDN by default, which the extension CSP
// (script-src 'self') blocks. Ship the runtime inside the extension instead; lib/ortEnv.ts
// points the loader here at runtime.
const ORT_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];

// See https://wxt.dev/api/config.html
export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  vite: () => ({
    plugins: [tailwindcss()],
  }),
  hooks: {
    'build:done': async (wxt) => {
      const src = path.resolve('node_modules/onnxruntime-web/dist');
      const dest = path.join(wxt.config.outDir, 'ort');
      await mkdir(dest, { recursive: true });
      for (const f of ORT_FILES) await cp(path.join(src, f), path.join(dest, f));
    },
  },
  manifest: {
    name: 'Enclave',
    description:
      'Run LLMs locally in your browser, on your own GPU. Ask about any page — nothing ever leaves your machine.',
    permissions: ['sidePanel', 'contextMenus', 'storage', 'tabs', 'offscreen', 'scripting', 'activeTab'],
    // WebLLM runs the model via WebAssembly + WebGPU, which requires 'wasm-unsafe-eval'.
    content_security_policy: {
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
    action: {
      default_title: 'Open Enclave',
    },
    commands: {
      'open-panel': {
        suggested_key: { default: 'Ctrl+Shift+L', mac: 'Command+Shift+L' },
        description: 'Open the Enclave side panel',
      },
    },
  },
});
