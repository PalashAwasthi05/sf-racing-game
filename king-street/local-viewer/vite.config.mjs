import {fileURLToPath} from 'node:url';
import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwind from '@tailwindcss/postcss';

const here = fileURLToPath(new URL('.', import.meta.url));
const project = fileURLToPath(new URL('../', import.meta.url));
export default defineConfig({
  root: here,
  publicDir: false,
  plugins: [react()],
  css: {postcss: {plugins: [tailwind({base: project})]}},
  resolve: {alias: {'@': project}},
  build: {outDir: process.env.WORLD_CLIENT_OUT || project+'.local-viewer', emptyOutDir: false,
    reportCompressedSize: false, sourcemap: false},
});
