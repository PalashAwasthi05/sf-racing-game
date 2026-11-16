import * as THREE from 'three';
import {readTextureCpuImage} from './compressed-textures.ts';

// One material per surface class instead of one per building. The generator gives every building its own copy of its
// wall material (render_gray#3a3e45, render_gray#767477, ...): the same texture pattern, the same normal and roughness
// maps, a different tint baked into the colour map. Each copy is its own draw call (they cannot be batched across
// buildings), its own three maps to upload and its own shader program state; at street level the scene pass issued
// several hundred draws for them and the WebGL calls behind each draw were most of the frame.
//
// Here the copies of a class whose colour map is nearly flat (render, plaster: the pattern within 6 % of its mean) and,
// texel for texel, the reference map times a colour (98 % of texels within 5 % on a 256 x 128 copy), and whose other
// maps and settings match, are replaced by the class's
// most used copy, with each mesh carrying its own copy's tint as a vertex colour (applied before any surface shading:
// weathering.ts reads the colour after it). Patterned maps (brick and mortar, panels and their joints, tinted separately:
// thin lines a downsampled check cannot vouch for) keep their own material. The replaced maps are never uploaded.
// Nothing here refers to a place.

const S = 256, H = 128;
type Sample = Float32Array;
// A sampled channel has only 256 possible inputs. Keep the rounded linear
// values identical to the original Float32 samples across every material.
const linearChannels = Float32Array.from({length: 256}, (_, i) => i / 255);
const srgbChannels = Float32Array.from({length: 256}, (_, i) => {
  const v = i / 255;
  return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
});
const contexts = new Map<number, CanvasRenderingContext2D>();
async function sample(tex: THREE.Texture | null, w = S, h = H): Promise<Sample | null> {
  if (!tex || typeof document === 'undefined') return null;
  if ((tex as THREE.CompressedTexture).isCompressedTexture)
    return readTextureCpuImage(tex, image => sampleImage(tex, image, w, h));
  return sampleImage(tex, tex.image, w, h);
}
function sampleImage(tex: THREE.Texture, img: CanvasImageSource & {width?: number; height?: number}, w: number, h: number): Sample | null {
  if (!img?.width || typeof document === 'undefined') return null;
  let ctx = contexts.get(w);
  if (!ctx) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    ctx = canvas.getContext('2d', {willReadFrequently: true})!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    contexts.set(w, ctx);
  }
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  const pixels = ctx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h * 3);
  const table = tex!.colorSpace === THREE.SRGBColorSpace ? srgbChannels : linearChannels;
  for (let source = 0, dest = 0; source < pixels.length; source += 4) {
    out[dest++] = table[pixels[source]];
    out[dest++] = table[pixels[source + 1]];
    out[dest++] = table[pixels[source + 2]];
  }
  return out;
}
const mean = (a: Sample) => { const m = [0, 0, 0]; for (let i = 0; i < a.length; i += 3) { m[0] += a[i]; m[1] += a[i + 1]; m[2] += a[i + 2]; } const n = a.length / 3; return m.map(x => x / n); };
function thumbnails() {
  const cache = new WeakMap<THREE.Texture, Promise<Sample | null>>();
  return (texture: THREE.Texture) => {
    if (!cache.has(texture)) cache.set(texture, sample(texture, 16, 8));
    return cache.get(texture);
  };
}
const sameMap = async (a: THREE.Texture | null, b: THREE.Texture | null, thumbnail: ReturnType<typeof thumbnails>) => {
  if (!a || !b) return a === b;
  if (a === b) return true;
  const ia = a.image as {width?: number; height?: number}, ib = b.image as {width?: number; height?: number};
  if (ia?.width !== ib?.width || ia?.height !== ib?.height || a.colorSpace !== b.colorSpace) return false;
  const sa = await thumbnail(a), sb = await thumbnail(b); if (!sa || !sb) return false;
  let d = 0; for (let i = 0; i < sa.length; i++) d = Math.max(d, Math.abs(sa[i] - sb[i])); return d < 2 / 255;
};
const sameSettings = (a: THREE.MeshStandardMaterial, b: THREE.MeshStandardMaterial) =>
  a.roughness === b.roughness && a.metalness === b.metalness && a.side === b.side && a.transparent === b.transparent && a.opacity === b.opacity &&
  a.alphaTest === b.alphaTest && a.color.equals(b.color) && a.emissive.equals(b.emissive) && !a.emissiveMap && !b.emissiveMap &&
  a.normalScale.equals(b.normalScale) && a.flatShading === b.flatShading && a.vertexColors === b.vertexColors && a.polygonOffset === b.polygonOffset &&
  a.map?.repeat.equals(b.map!.repeat) && a.map?.offset.equals(b.map!.offset) && a.map?.wrapS === b.map?.wrapS && a.map?.wrapT === b.map?.wrapT;

type TintMemoOptions = {
  sceneHash: string;
  textureIndex: (texture: THREE.Texture) => number | undefined;
};
type TintEntry = boolean | number[];

// Only pixel comparisons are reusable. Material settings, layer rank and the
// class reference are still chosen from the current scene every time.
function tintMemo(options?: TintMemoOptions) {
  const slot = 'world-pilot-tint-samples-v1';
  let entries: Record<string, TintEntry> = Object.create(null);
  let dirty = false;
  if (options) {
    try {
      const saved = JSON.parse(localStorage.getItem(slot) ?? 'null');
      if (saved?.sceneHash === options.sceneHash && saved.entries && typeof saved.entries === 'object')
        entries = saved.entries;
    } catch { /* Storage can be unavailable in a private or embedded viewer. */ }
  }
  return {
    key(texture: THREE.Texture | null) {
      if (!options || !texture) return undefined;
      const index = options.textureIndex(texture);
      return index === undefined ? undefined : index + '/' + texture.colorSpace;
    },
    get(key?: string) { return key ? entries[key] : undefined; },
    set(key: string | undefined, value: TintEntry) {
      if (key) { entries[key] = value; dirty = true; }
    },
    save() {
      if (!options || !dirty) return;
      try { localStorage.setItem(slot, JSON.stringify({sceneHash: options.sceneHash, entries})); }
      catch { /* The live result is already in use. */ }
    },
  };
}
/** Replaces tint-only copies of each class's material (see above). `classOf` names a mesh's surface class. Returns the
 *  materials replaced (their maps can be disposed) and per class how many copies were merged into one. */
export async function mergeTintVariants(meshes: THREE.Mesh[], classOf: (o: THREE.Mesh) => string, rankOf: (o: THREE.Mesh) => number = () => 0, options?: TintMemoOptions) {
  // every mesh of a material (a building's window reveals take its wall material under another class name), and the
  // material's class: the one most of its meshes carry
  const users = new Map<THREE.MeshStandardMaterial, THREE.Mesh[]>(), blocked = new Set<THREE.Material>();
  for (const o of meshes) {
    const m = o.material; if (Array.isArray(m)) { m.forEach(x => blocked.add(x)); continue; }
    if (!(m instanceof THREE.MeshStandardMaterial)) continue;
    if (m.constructor !== THREE.MeshStandardMaterial || !m.map || m.vertexColors || o.geometry.attributes.color) { blocked.add(m); continue; }
    let list = users.get(m); if (!list) { list = []; users.set(m, list); } list.push(o);
  }
  const byClass = new Map<string, Map<THREE.MeshStandardMaterial, THREE.Mesh[]>>();
  for (const [m, list] of users) {
    if (blocked.has(m)) continue;
    const count = new Map<string, number>(); for (const o of list) { const c = classOf(o); count.set(c, (count.get(c) ?? 0) + o.geometry.attributes.position.count); }
    // and its depth priority (the highest of its meshes'): copies are merged only within one priority, so the fixed order
    // between coincident surfaces (world-view.ts) is kept
    const c = [...count.entries()].sort((a, b) => b[1] - a[1])[0][0] + '|' + Math.max(...list.map(rankOf));
    let mats = byClass.get(c); if (!mats) { mats = new Map(); byClass.set(c, mats); } mats.set(m, list);
  }
  const replaced: THREE.MeshStandardMaterial[] = [], merged: Record<string, number> = {};
  const thumbnail = thumbnails();
  const memo = tintMemo(options);
  for (const [cls, mats] of byClass) {
    if (mats.size < 2) continue;
    // the class's most used copy carries the others
    const ref = [...mats.entries()].sort((a, b) => b[1].length - a[1].length)[0][0];
    let rs: Sample | null = null, rm: number[] | null = null;
    const refKey = memo.key(ref.map);
    const cachedFlat = memo.get(refKey);
    if (cachedFlat === false) continue;
    if (cachedFlat !== true) {
      rs = await sample(ref.map); if (!rs) continue; rm = mean(rs);
      let v = 0; const n = rs.length / 3, lm = .2126 * rm[0] + .7152 * rm[1] + .0722 * rm[2];
      for (let i = 0; i < n; i++) { const l = .2126 * rs[3 * i] + .7152 * rs[3 * i + 1] + .0722 * rs[3 * i + 2]; v += (l - lm) ** 2; }
      const flat = Math.sqrt(v / n) / Math.max(lm, 1e-4) <= .06;
      memo.set(refKey, flat);
      if (!flat) continue;
    }
    const into: [THREE.MeshStandardMaterial, number[]][] = [];
    for (const m of mats.keys()) {
      if (m === ref || !sameSettings(m, ref) || !(await sameMap(m.normalMap, ref.normalMap, thumbnail)) || !(await sameMap(m.roughnessMap, ref.roughnessMap, thumbnail)) || !(await sameMap(m.metalnessMap, ref.metalnessMap, thumbnail)) || !(await sameMap(m.aoMap, ref.aoMap, thumbnail))) continue;
      const materialKey = memo.key(m.map);
      const pairKey = refKey && materialKey ? refKey + '|' + materialKey : undefined;
      const cached = memo.get(pairKey);
      if (cached === false) continue;
      if (Array.isArray(cached) && cached.length === 3 && cached.every(Number.isFinite)) {
        into.push([m, cached]); continue;
      }
      rs ??= await sample(ref.map);
      if (!rs) continue;
      const referenceMean = rm ??= mean(rs);
      const s = await sample(m.map); if (!s) continue;
      const sm = mean(s), tint = sm.map((x, k) => x / Math.max(referenceMean[k], 1e-6));
      // The 98th-percentile test only needs the number above the threshold.
      const pixels = s.length / 3;
      const limit = pixels - Math.floor(pixels * .98);
      let failures = 0;
      for (let i = 0; i < s.length && failures < limit; i += 3) {
        for (let k = 0; k < 3; k++) {
          const relative = Math.abs(s[i + k] - rs[i + k] * tint[k]) / Math.max(s[i + k], 2e-3);
          if (Math.fround(relative) >= .05) { failures++; break; }
        }
      }
      memo.set(pairKey, failures < limit ? tint : false);
      if (failures < limit) into.push([m, tint]);
    }
    if (!into.length) continue;
    const paint = (o: THREE.Mesh, t: number[]) => { const n = o.geometry.attributes.position.count, a = new Float32Array(n * 3); for (let i = 0; i < n; i++) { a[3 * i] = t[0]; a[3 * i + 1] = t[1]; a[3 * i + 2] = t[2]; } o.geometry.setAttribute('color', new THREE.BufferAttribute(a, 3)); };
    for (const o of mats.get(ref)!) paint(o, [1, 1, 1]);
    for (const [m, t] of into) { for (const o of mats.get(m)!) { paint(o, t); o.material = ref; } replaced.push(m); }
    ref.vertexColors = true; ref.needsUpdate = true;
    merged[cls] = (merged[cls] ?? 0) + into.length + 1;
  }
  memo.save();
  return {replaced, merged};
}
