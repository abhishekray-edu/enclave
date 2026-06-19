// Renders public/logo.svg into the PNG icon sizes Chrome needs.
// Run: node scripts/gen-icons.mjs
import sharp from 'sharp';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const svg = readFileSync(resolve(root, 'public/logo.svg'));
const sizes = [16, 32, 48, 96, 128];

for (const size of sizes) {
  const out = resolve(root, `public/icon/${size}.png`);
  await sharp(svg, { density: 384 }).resize(size, size).png().toFile(out);
  console.log(`wrote public/icon/${size}.png`);
}
