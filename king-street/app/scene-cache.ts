// Large local worlds can exceed the browser's ordinary HTTP cache. Only
// content-addressed display packages use this cache. Archival downloads and
// mutable manifest aliases keep their normal HTTP behavior.
const prefix = 'world-pilot-display-';

export async function openDisplayCache(hash: string): Promise<Cache | null> {
  if (typeof caches === 'undefined' || !/^[0-9a-f]{12}$/.test(hash)) return null;
  try {
    const name = prefix + hash;
    const cache = await caches.open(name);
    const others = (await caches.keys()).filter(k => k.startsWith(prefix) && k !== name);
    // Keep the current world and one other for switching areas.
    await Promise.all(others.slice(0, -1).map(k => caches.delete(k)));
    return cache;
  } catch { return null; }
}

async function inflate(response: Response) {
  if (!response.ok || !response.body) throw new Error('Scene download failed');
  return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

export async function displayChunk(url: string, cache: Pick<Cache, 'match' | 'put'> | null, signal: AbortSignal) {
  if (cache) {
    let saved: Response | undefined;
    try { saved = await cache.match(url); } catch { /* Continue with a normal request. */ }
    if (saved) {
      const compressedSize = Number(saved.headers.get('X-World-Chunk-Bytes'));
      if (Number.isSafeInteger(compressedSize) && compressedSize > 0) {
        try { return {buffer: await inflate(saved), compressedSize}; }
        catch { /* A damaged entry is replaced by the fresh copy below. */ }
      }
    }
  }
  const response = await fetch(url, {signal});
  if (!response.ok) throw new Error('Scene download failed');
  if (!response.body) throw new Error('Empty model chunk');
  const declared = Number(response.headers.get('Content-Length'));
  // Stream the normal static-host response directly into both consumers. Wait
  // for this chunk's cache write so slow storage cannot queue a city's worth
  // of compressed bodies while the parser continues loading.
  if (!cache || (declared > 0 && Number.isSafeInteger(declared) && !response.headers.get('Content-Encoding'))) {
    const saved = cache ? response.clone() : null;
    const writing = saved && cache ? cache.put(url, new Response(saved.body, {
      headers: {'X-World-Chunk-Bytes': String(declared)},
    })).catch(() => {}) : Promise.resolve();
    let compressedSize = 0;
    const counted = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
      transform(bytes, controller) { compressedSize += bytes.byteLength; controller.enqueue(bytes); },
    }));
    const [buffer] = await Promise.all([inflate(new Response(counted)), writing]);
    return {buffer, compressedSize};
  }
  // Some hosts omit the byte count or transparently encode the HTTP response.
  const bytes = await response.arrayBuffer();
  const stored = new Response(bytes, {headers: {'X-World-Chunk-Bytes': String(bytes.byteLength)}});
  const [buffer] = await Promise.all([inflate(stored.clone()), cache.put(url, stored).catch(() => {})]);
  return {buffer, compressedSize: bytes.byteLength};
}
