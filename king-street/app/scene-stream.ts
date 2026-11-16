import {GLTFLoader, type GLTF, type GLTFParser} from 'three/addons/loaders/GLTFLoader.js';

import {flatNodes,omitStaticTrees} from './scene-flat.ts';
import {compressedTexturePlugin} from './compressed-textures.ts';
import {pooledImagePlugin,releaseParserImageSources,type SceneImagePool} from './scene-image-pool.ts';

type BufferView = {buffer: number; byteOffset?: number; byteLength: number; extensions?: unknown};

/** Once a scene is fully loaded, keep its associations for evidence/material
 * identity but release loader-only promises and the legacy GLB container. The
 * viewer uses the resolved scene from here, rather than requesting dependencies
 * from its parser again. Geometry attributes retain all bytes they still use. */
export function releaseSceneLoadingData(parser: GLTFParser) {
  const loading = parser as unknown as {
    cache: {removeAll(): void}; sourceCache?: Record<string, unknown>; textureCache?: Record<string, unknown>; extensions?: Record<string, {body?: ArrayBuffer | null}>;
  };
  loading.cache.removeAll();
  // GLTFLoader keeps image and sampler promises outside its dependency cache.
  // Drop references only: live texture Sources must remain usable after a
  // WebGL context restore, and retired images are closed at material cleanup.
  loading.sourceCache = {};
  loading.textureCache = {};
  const binary = loading.extensions?.KHR_binary_glTF;
  if (binary) binary.body = null;
}

/** Called only after the viewer has consumed node associations, corrected
 * winding and merged materials. The renderer owns the resolved objects now.
 * A parser plugin can otherwise retain its GLB directory and resolved image
 * promises for the entire time a driving cell is resident. Image leases for
 * context restoration are kept independently under the same parser identity. */
export function finishDrawableParser(parser: GLTFParser) {
  releaseSceneLoadingData(parser);
  parser.associations.clear();
  const loader = parser as unknown as Record<string, unknown>;
  for (const field of ['plugins','extensions','primitiveCache','nodeCache','nodeNamesUsed']) loader[field] = {};
  const document = parser.json;
  if (document) loader.json = {asset: document.asset};
}

// The exporter puts ordinary disjoint buffer views into one large BIN chunk. Inflate
// straight into the exact buffers Three will use for geometry and embedded images.
// GLTFLoader still handles accessors, offsets within a view, materials and scene nodes.
export async function parseSceneStream(stream: ReadableStream<Uint8Array>, expectedBytes?: number, onGeometry?: () => void, options: {imagePool?:SceneImagePool; omitStreamedTrees?:ReadonlySet<string>; yieldCpu?:()=>Promise<void>} = {}) {
  const parsers=new Set<GLTFParser>();let complete=false;
 const configure=(loader:GLTFLoader)=>{
  loader.register(parser=>{parsers.add(parser);return compressedTexturePlugin(parser,options.imagePool);});
  if(options.imagePool)loader.register(parser=>{parsers.add(parser);return pooledImagePlugin(parser,options.imagePool!);});
  return loader;
 };
 const reader = stream.getReader();
  let current = new Uint8Array(0) as Uint8Array, cursor = 0;
  async function take(length: number, keep = true) {
    const result = new Uint8Array(keep ? length : 0);
    let written = 0;
    while (written < length) {
      if (cursor === current.length) {
        const {value, done} = await reader.read();
        if (done) throw new Error('Incomplete model geometry');
        current = value;
        cursor = 0;
      }
      const n = Math.min(length - written, current.length - cursor);
      if (keep) result.set(current.subarray(cursor, cursor + n), written);
      written += n;
      cursor += n;
    }
    return result;
  }
  async function finish() {
    if (cursor !== current.length || !(await reader.read()).done) throw new Error('Unexpected model data after the GLB');
    onGeometry?.();
  }
  try {
    const prefix = await take(20), header = new DataView(prefix.buffer);
    // Keep Three's normal loader for other container layouts and extensions,
    // including compressed buffer views whose decoder needs the full BIN chunk.
    async function ordinary(parts: Uint8Array[]) {
      if (cursor < current.length) parts.push(current.subarray(cursor));
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        parts.push(value);
      }
      const size = parts.reduce((sum, p) => sum + p.length, 0);
      if (expectedBytes !== undefined && size !== expectedBytes) throw new Error('Incomplete model geometry');
      const bytes = new Uint8Array(size);
      let offset = 0;
      for (const p of parts) { bytes.set(p, offset); offset += p.length; }
      onGeometry?.();
      const result = await configure(new GLTFLoader()).parseAsync(bytes.buffer, '/');
      releaseSceneLoadingData(result.parser);
      complete=true;return result;
    }
    if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2 ||
        header.getUint32(16, true) !== 0x4e4f534a) return await ordinary([prefix]);
    const total = header.getUint32(8, true), jsonLength = header.getUint32(12, true);
    if ((expectedBytes !== undefined && expectedBytes !== total) || jsonLength > total - 20) throw new Error('Invalid model length');
    const jsonBytes = await take(jsonLength);
    if (total - 20 - jsonLength < 8) return await ordinary([prefix, jsonBytes]);
    const binHeader = await take(8), bin = new DataView(binHeader.buffer), bodyLength = bin.getUint32(0, true);
    if (bin.getUint32(4, true) !== 0x004e4942 || bodyLength !== total - jsonLength - 28) {
      return await ordinary([prefix, jsonBytes, binHeader]);
    }
    const jsonText = new TextDecoder().decode(jsonBytes);
    const json = JSON.parse(jsonText) as {
      buffers?: {uri?: string; byteLength: number}[]; bufferViews?: BufferView[]; extensionsUsed?: string[]; extensionsRequired?: string[];
    };
    const directory = (json.bufferViews ?? []).map((v, index) => ({
      index, buffer: v.buffer, start: v.byteOffset ?? 0, length: v.byteLength,
    })).sort((a,b) => a.start - b.start);
    const binSize = json.buffers?.[0]?.byteLength;
    let end = 0;
    const disjoint = directory.every(v => {
      if (v.buffer !== 0 || !Number.isSafeInteger(v.start) || !Number.isSafeInteger(v.length) ||
          v.start < end || v.length < 0 || v.start + v.length > (binSize ?? 0)) return false;
      end = v.start + v.length;
      return true;
    });
    if (json.buffers?.length !== 1 || json.buffers[0].uri !== undefined || !Number.isSafeInteger(binSize) ||
        binSize! > bodyLength || json.extensionsUsed?.length || json.extensionsRequired?.length ||
        json.bufferViews?.some(v => v.extensions) || !disjoint) {
      return await ordinary([prefix, jsonBytes, binHeader]);
    }
    const skipped = options.omitStreamedTrees ? omitStaticTrees(json, options.omitStreamedTrees) : new Set<number>();
    const buffers: ArrayBuffer[] = [];
    const waiting = new Map<number, {resolve: (value: ArrayBuffer) => void; reject: (reason: unknown) => void}>();
    let failure: unknown;
    const loader = configure(new GLTFLoader());
    let ready!: () => void, rejectGeometry!: (error: unknown) => void;
    const binaryReady = new Promise<void>((resolve, reject) => { ready = resolve; rejectGeometry = reject; });
    // General glTF files may not have any static node consumers on this promise.
    void binaryReady.catch(() => {});
    loader.register(parser => ({
      ...flatNodes(parser, buffers, binaryReady, options.yieldCpu),
      name: 'viewer_streamed_buffer_views',
      loadBufferView(index: number) {
        if (failure) return Promise.reject(failure);
        if (skipped.has(index)) return Promise.reject(new Error('Omitted static tree buffer view requested'));
        if (buffers[index]) return Promise.resolve(buffers[index]);
        if (!json.bufferViews?.[index]) return Promise.reject(new Error('Unknown model buffer view'));
        return new Promise<ArrayBuffer>((resolve, reject) => waiting.set(index, {resolve, reject}));
      },
    }));
    // GLTFLoader accepts an already parsed glTF document as well as binary
    // or JSON text. Its declarations currently only list the latter inputs.
    const parseDocument = loader.parseAsync.bind(loader) as unknown as (document: typeof json, base: string) => Promise<GLTF>;
    // Start texture decoding when each embedded image arrives. Keep the rest
    // of the byte stream moving while Three resolves its material dependencies.
    const model = parseDocument(json, '/');
    void model.catch(() => {}); // A stream failure and a parser failure can coincide.
    try {
      let position = 0;
      for (const v of directory) {
        if (v.start > position) await take(v.start - position, false);
        if (skipped.has(v.index)) {await take(v.length, false);position = v.start + v.length;continue;}
        const buffer = (await take(v.length)).buffer;
        buffers[v.index] = buffer;
        waiting.get(v.index)?.resolve(buffer);
        waiting.delete(v.index);
        position = v.start + v.length;
      }
      if (position < bodyLength) await take(bodyLength - position, false);
      await finish();
      ready();
      const result = await model;
      // Dependency promises otherwise retain the original buffer views after
      // static batching has moved vertex storage into shared merged arrays.
      buffers.length = 0;
      releaseSceneLoadingData(result.parser);
      complete=true;return result;
    } catch (error) {
      failure = error;
      rejectGeometry(error);
      for (const w of waiting.values()) w.reject(error);
      throw error;
    }
  } finally {
    if(!complete)for(const parser of parsers)releaseParserImageSources(parser);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
