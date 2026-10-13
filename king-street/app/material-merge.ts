import * as THREE from 'three';

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
const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
function sample(tex: THREE.Texture | null, w = S, h = H): Sample | null {
  const img = tex?.image as CanvasImageSource & {width?: number; height?: number} | undefined;
  if (!img || !canvas || !img.width) return null;
  canvas.width = w; canvas.height = h; const ctx = canvas.getContext('2d', {willReadFrequently: true})!;
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; ctx.clearRect(0, 0, w, h); ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data, out = new Float32Array(w * h * 3);
  const srgb = tex!.colorSpace === THREE.SRGBColorSpace, lin = (v: number) => { v /= 255; return srgb ? (v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4) : v; };
  for (let i = 0; i < w * h; i++) { out[3 * i] = lin(d[4 * i]); out[3 * i + 1] = lin(d[4 * i + 1]); out[3 * i + 2] = lin(d[4 * i + 2]); }
  return out;
}
const mean = (a: Sample) => { const m = [0, 0, 0]; for (let i = 0; i < a.length; i += 3) { m[0] += a[i]; m[1] += a[i + 1]; m[2] += a[i + 2]; } const n = a.length / 3; return m.map(x => x / n); };
const sameMap = (a: THREE.Texture | null, b: THREE.Texture | null) => {
  if (!a || !b) return a === b;
  if (a === b) return true;
  const ia = a.image as {width?: number; height?: number}, ib = b.image as {width?: number; height?: number};
  if (ia?.width !== ib?.width || ia?.height !== ib?.height || a.colorSpace !== b.colorSpace) return false;
  const sa = sample(a, 16, 8), sb = sample(b, 16, 8); if (!sa || !sb) return false;
  let d = 0; for (let i = 0; i < sa.length; i++) d = Math.max(d, Math.abs(sa[i] - sb[i])); return d < 2 / 255;
};
const sameSettings = (a: THREE.MeshStandardMaterial, b: THREE.MeshStandardMaterial) =>
  a.roughness === b.roughness && a.metalness === b.metalness && a.side === b.side && a.transparent === b.transparent && a.opacity === b.opacity &&
  a.alphaTest === b.alphaTest && a.color.equals(b.color) && a.emissive.equals(b.emissive) && !a.emissiveMap && !b.emissiveMap &&
  a.normalScale.equals(b.normalScale) && a.flatShading === b.flatShading && a.vertexColors === b.vertexColors && a.polygonOffset === b.polygonOffset &&
  a.map?.repeat.equals(b.map!.repeat) && a.map?.offset.equals(b.map!.offset) && a.map?.wrapS === b.map?.wrapS && a.map?.wrapT === b.map?.wrapT;

/** Replaces tint-only copies of each class's material (see above). `classOf` names a mesh's surface class. Returns the
 *  materials replaced (their maps can be disposed) and per class how many copies were merged into one. */
export function mergeTintVariants(meshes: THREE.Mesh[], classOf: (o: THREE.Mesh) => string, rankOf: (o: THREE.Mesh) => number = () => 0) {
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
  for (const [cls, mats] of byClass) {
    if (mats.size < 2) continue;
    // the class's most used copy carries the others
    const ref = [...mats.entries()].sort((a, b) => b[1].length - a[1].length)[0][0];
    const rs = sample(ref.map); if (!rs) continue; const rm = mean(rs);
    { let v = 0; const n = rs.length / 3, lm = .2126 * rm[0] + .7152 * rm[1] + .0722 * rm[2];
      for (let i = 0; i < n; i++) { const l = .2126 * rs[3 * i] + .7152 * rs[3 * i + 1] + .0722 * rs[3 * i + 2]; v += (l - lm) ** 2; }
      if (Math.sqrt(v / n) / Math.max(lm, 1e-4) > .06) continue; }
    const into: [THREE.MeshStandardMaterial, number[]][] = [];
    for (const m of mats.keys()) {
      if (m === ref || !sameSettings(m, ref) || !sameMap(m.normalMap, ref.normalMap) || !sameMap(m.roughnessMap, ref.roughnessMap) || !sameMap(m.metalnessMap, ref.metalnessMap) || !sameMap(m.aoMap, ref.aoMap)) continue;
      const s = sample(m.map); if (!s) continue; const sm = mean(s), tint = sm.map((x, k) => x / Math.max(rm[k], 1e-6));
      const rel = new Float32Array(s.length / 3);
      for (let i = 0; i < rel.length; i++) { let e = 0; for (let k = 0; k < 3; k++) e = Math.max(e, Math.abs(s[3 * i + k] - rs[3 * i + k] * tint[k]) / Math.max(s[3 * i + k], 2e-3)); rel[i] = e; }
      rel.sort(); if (rel[Math.floor(rel.length * .98)] < .05) into.push([m, tint]);
    }
    if (!into.length) continue;
    const paint = (o: THREE.Mesh, t: number[]) => { const n = o.geometry.attributes.position.count, a = new Float32Array(n * 3); for (let i = 0; i < n; i++) { a[3 * i] = t[0]; a[3 * i + 1] = t[1]; a[3 * i + 2] = t[2]; } o.geometry.setAttribute('color', new THREE.BufferAttribute(a, 3)); };
    for (const o of mats.get(ref)!) paint(o, [1, 1, 1]);
    for (const [m, t] of into) { for (const o of mats.get(m)!) { paint(o, t); o.material = ref; } replaced.push(m); }
    ref.vertexColors = true; ref.needsUpdate = true;
    merged[cls] = (merged[cls] ?? 0) + into.length + 1;
  }
  return {replaced, merged};
}
