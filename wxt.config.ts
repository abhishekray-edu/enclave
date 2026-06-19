import { defineConfig } from 'wxt';
import tailwindcss from '@tailwindcss/vite';

// See https://wxt.dev/api/config.html
export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  vite: () => ({
    plugins: [tailwindcss()],
  }),
  manifest: {
    name: 'Enclave',
    description:
      'A private, on-device AI that reads the page and answers your questions. Runs a local model in your browser — nothing leaves your machine.',
    permissions: ['sidePanel', 'contextMenus', 'storage', 'tabs', 'offscreen'],
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
