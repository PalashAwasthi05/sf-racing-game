import {openDisplayCache, displayChunk} from './scene-cache.ts';

export type SceneManifest = {
  contentHash?: string;
  startup?: SceneManifest & {sourceHash: string; version: number};
  startupGpu?: SceneManifest & {sourceHash: string; version: number; profile: string; portableHash: string};
  format?: 'gzip-chunks';
  chunks: string[];
  compressedBytes: number;
  uncompressedBytes?: number;
  world?: string;
  cars?: string;
  overhead?: string;
  spatialIndex?: string;
};

export async function loadSceneManifest() {
  const requested = new URL(window.location.href).searchParams.get('manifest');
  const path = requested && /^\/compare\/[\w.-]+\.json$/.test(requested) ? requested : '/model.json';
  const response = await fetch(path, {cache: 'no-cache'});
  if (!response.ok) throw new Error('Scene manifest unavailable');
  return withStartupPackage(await response.json() as SceneManifest);
}

const startupVersion = 3;
type Startup = NonNullable<SceneManifest['startup']>;
function validStartup(value: unknown, sourceHash: string | undefined): value is Startup {
  if (!value || typeof value !== 'object' || !sourceHash || !/^[0-9a-f]{12}$/.test(sourceHash)) return false;
  const d = value as Partial<Startup>;
  if ((d.version !== 1 && d.version !== 2 && d.version !== startupVersion) || d.format !== 'gzip-chunks' ||
      d.sourceHash !== sourceHash || !d.contentHash || !/^[0-9a-f]{12}$/.test(d.contentHash) ||
      !Number.isSafeInteger(d.compressedBytes) || d.compressedBytes! <= 0 ||
      !Number.isSafeInteger(d.uncompressedBytes) || d.uncompressedBytes! < 28 || !Array.isArray(d.chunks)) return false;
  return d.chunks.length === Math.ceil(d.uncompressedBytes! / (8 * 1024 * 1024)) &&
    d.chunks.every((url, index) => url === '/world-' + d.contentHash + '-' + String(index).padStart(2, '0') + '.bin');
}

function validGpuStartup(manifest: SceneManifest) {
  const gpu = manifest.startupGpu;
  return validStartup(gpu, manifest.contentHash) && gpu?.version === 3 && gpu.profile === 's3tc-v1' &&
    gpu.portableHash === manifest.startup?.contentHash;
}

export function sceneForDisplay(manifest: SceneManifest, s3tc = false): SceneManifest {
  if (s3tc && validGpuStartup(manifest)) return manifest.startupGpu!;
  return validStartup(manifest.startup, manifest.contentHash) ? manifest.startup : manifest;
}

export async function withGpuStartupPackage(manifest: SceneManifest): Promise<SceneManifest> {
  if (validGpuStartup(manifest) || !validStartup(manifest.startup, manifest.contentHash)) return manifest;
  try {
    const response = await fetch('/startup-gpu-' + manifest.contentHash + '.json');
    if (response.ok) {
      const candidate = {...manifest, startupGpu: await response.json() as SceneManifest['startupGpu']};
      if (validGpuStartup(candidate)) return candidate;
    }
  } catch { /* The portable display package is complete on every device. */ }
  return manifest;
}

// A frozen manifest must continue to describe exactly the world originally
// reviewed. Look up only its own source hash and retain its archival download.
export async function withStartupPackage(manifest: SceneManifest): Promise<SceneManifest> {
  const hash = manifest.contentHash;
  if (manifest.format === 'gzip-chunks' || !hash || !/^[0-9a-f]{12}$/.test(hash) ||
      !manifest.chunks.every((url, i) => url === '/world-' + hash + '-' + String(i).padStart(2, '0') + '.bin') ||
      (validStartup(manifest.startup, hash) && manifest.startup.version === startupVersion)) return manifest;
  // Try newer packages first; an older inline package is already a complete
  // fallback. Frozen manifests without one can still discover their v2 copy.
  const fallback = validStartup(manifest.startup, hash) ? manifest.startup : undefined;
  for (const version of [startupVersion, 2]) {
    if (fallback && version <= fallback.version) break;
    try {
      const response = await fetch('/startup-v' + version + '-' + hash + '.json');
      if (response.ok) {
        const descriptor: unknown = await response.json();
        if (validStartup(descriptor, hash) && descriptor.version === version) return {...manifest, startup: descriptor};
      }
    } catch { /* Keep the older display package or the complete archive. */ }
  }
  return manifest;
}

export function sceneStream(manifest: SceneManifest, onProgress: (message: string) => void, signal?: AbortSignal, cacheGroup?:string) {
  const abort = new AbortController();
  const stop = () => abort.abort();
  if(signal?.aborted) stop();
  else signal?.addEventListener('abort',stop,{once:true});
  const detach = () => signal?.removeEventListener('abort',stop);
  const separate = manifest.format === 'gzip-chunks';
  const cache = separate && manifest.contentHash ? openDisplayCache(cacheGroup ?? manifest.contentHash) : null;
  const chunks = async function* () {
    let loaded = 0;
    const pending: Promise<{buffer: ArrayBuffer; compressedSize: number} | {error: unknown}>[] = [];
    let next = 0;
    async function load(url: string) {
      if (separate) return displayChunk(url, await cache, abort.signal);
      const response = await fetch(url, {signal: abort.signal});
      if (!response.ok) throw new Error('Scene download failed');
      const compressed = await response.arrayBuffer();
      return {buffer: compressed, compressedSize: compressed.byteLength};
    }
    // Display chunks are independent gzip files. Inflate them concurrently,
    // then deliver bytes in order. An archival GLB is one continuous stream.
    while (next < manifest.chunks.length || pending.length) {
      while (pending.length < (separate ? 4 : 3) && next < manifest.chunks.length) {
        pending.push(load(manifest.chunks[next++]).catch((error: unknown) => ({error})));
      }
      const item = await pending.shift()!;
      if ('error' in item) throw item.error;
      loaded += item.compressedSize;
      onProgress('Loading detailed scene… ' + Math.round(100 * loaded / manifest.compressedBytes) + '%');
      yield new Uint8Array(item.buffer);
    }
    if (loaded !== manifest.compressedBytes) throw new Error('Incomplete model download');
  };
  const iterator = chunks();
  const stream = new ReadableStream<Uint8Array<ArrayBuffer>>({
    async pull(controller) {
      try {
        const chunk = await iterator.next();
        if (chunk.done) {detach(); controller.close();}
        else controller.enqueue(chunk.value);
      } catch (error) {
        detach(); abort.abort();
        controller.error(error);
      }
    },
    cancel() { detach(); abort.abort(); },
  });
  return separate ? stream : stream.pipeThrough(new DecompressionStream('gzip'));
}

export async function downloadScene(manifest: SceneManifest) {
  const blob = await new Response(sceneStream(manifest, () => {}), {
    headers: {'Content-Type': 'model/gltf-binary'},
  }).blob();
  if (manifest.uncompressedBytes !== undefined && blob.size !== manifest.uncompressedBytes) {
    throw new Error('Incomplete model geometry');
  }
  return blob;
}
