// Serves index.html + bench.ts and the viewer module copies under variants/, against the live viewer's public/
// folder and node_modules, on its own port and dependency cache: the running preview on :3001 never sees these files.
import { fileURLToPath } from 'node:url';
import { defineConfig } from '../king-street/node_modules/vite/dist/node/index.js';
const here = fileURLToPath(new URL('.', import.meta.url));
const viewer = fileURLToPath(new URL('../king-street/', import.meta.url));
const addons = ['loaders/GLTFLoader.js', 'utils/BufferGeometryUtils.js', 'controls/OrbitControls.js', 'postprocessing/EffectComposer.js', 'postprocessing/RenderPass.js', 'postprocessing/GTAOPass.js', 'postprocessing/OutputPass.js', 'postprocessing/Pass.js', 'postprocessing/ShaderPass.js', 'postprocessing/UnrealBloomPass.js', 'objects/Sky.js', 'csm/CSM.js'];
export default defineConfig({
  root: here,
  cacheDir: here + '.vite-cache',
  publicDir: viewer + 'public',
  server: { port: Number(process.env.BENCH_PORT || 3107), strictPort: true, host: '127.0.0.1', fs: { allow: [here, viewer] }, hmr: false, watch: { ignored: ['**/node_modules/**', '**/.vite-cache/**'] } },
  // all dependencies optimised up front: a dependency discovered mid-run reloads the optimiser and can leave two copies of three
  optimizeDeps: { include: ['three', ...addons.map(a => 'three/addons/' + a)] },
});
