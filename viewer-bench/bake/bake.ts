import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {MeshoptDecoder} from 'three/examples/jsm/libs/meshopt_decoder.module.js';

async function load() {
  const l = new GLTFLoader(); l.setMeshoptDecoder(MeshoptDecoder);
  const g = await l.loadAsync('/bake/car-pack.glb'); g.scene.updateMatrixWorld(true); return g.scene;
}
(window as any).__inspect = async () => {
  const scene = await load(); const out: any[] = [];
  scene.traverse(o => {
    if (!/body$/i.test(o.name)) return;
    const q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(); o.matrixWorld.decompose(p, q, s);
    const axes = ['x', 'y', 'z'].map((a, i) => new THREE.Vector3(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0).applyQuaternion(q).toArray().map(v => +v.toFixed(3)));
    const box = new THREE.Box3().setFromObject(o);
    const parts = o.children.map(c => { const m = c as THREE.Mesh; const mat = m.material as THREE.MeshStandardMaterial; return [c.name, mat?.name, mat?.map?.image?.width, (m.geometry?.index?.count ?? 0) / 3]; });
    out.push({name: o.name, pos: p.toArray().map(v => +v.toFixed(2)), scale: s.toArray().map(v => +v.toFixed(5)), axes, size: box.getSize(new THREE.Vector3()).toArray().map(v => +v.toFixed(2)), parts});
  });
  return out;
};

type Part = {geom: THREE.BufferGeometry; mat: THREE.MeshStandardMaterial; role: 'body' | 'glass' | 'optics' | 'wheel'};
export type PackCar = {name: string; parts: Part[]; L: number; W: number; H: number};

function texData(t: THREE.Texture | null | undefined) {
  if (!t) return null; const im = t.image as CanvasImageSource & {width: number; height: number};
  const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; const x = c.getContext('2d')!; x.drawImage(im, 0, 0);
  return {w: c.width, h: c.height, d: x.getImageData(0, 0, c.width, c.height).data};
}
function sample(t: ReturnType<typeof texData>, u: number, v: number, flipY: boolean) {
  if (!t) return [255, 255, 255];
  u = ((u % 1) + 1) % 1; v = ((v % 1) + 1) % 1; const x = Math.min(t.w - 1, Math.floor(u * t.w)), y = Math.min(t.h - 1, Math.floor((flipY ? 1 - v : v) * t.h));
  const k = (y * t.w + x) * 4; return [t.d[k], t.d[k + 1], t.d[k + 2]];
}

/** Every car of the pack in its own frame: front along +x, up +y, centred over the ground at the origin. */
export function extract(scene: THREE.Object3D): PackCar[] {
  const bodies: THREE.Object3D[] = [], wheels: THREE.Object3D[] = [];
  scene.traverse(o => { if (/body$/i.test(o.name)) bodies.push(o); else if (/^Wheel_[A-Z]\d*(_\d+)?$/.test(o.name)) wheels.push(o); });
  const centre = (o: THREE.Object3D) => new THREE.Box3().setFromObject(o).getCenter(new THREE.Vector3());
  const bc = bodies.map(centre), owner = new Map<THREE.Object3D, THREE.Object3D[]>(bodies.map(b => [b, []]));
  for (const w of wheels) { const p = centre(w); let best = 0; bc.forEach((c, i) => { if (Math.hypot(c.x - p.x, c.z - p.z) < Math.hypot(bc[best].x - p.x, bc[best].z - p.z)) best = i; }); owner.get(bodies[best])!.push(w); }
  const cars: PackCar[] = [];
  for (const b of bodies) {
    const parts: Part[] = [];
    const take = (o: THREE.Object3D, wheel: boolean) => o.traverse(m => {
      if (!(m as THREE.Mesh).isMesh) return; const mesh = m as THREE.Mesh, mat = mesh.material as THREE.MeshStandardMaterial;
      const g = mesh.geometry.clone(); for (const k of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(k)) g.deleteAttribute(k);
      // dequantise (KHR_mesh_quantization) before baking the node transform in
      for (const k of Object.keys(g.attributes)) { const a = g.attributes[k] as THREE.BufferAttribute; if (a.normalized || !(a.array instanceof Float32Array)) { const f = new Float32Array(a.count * a.itemSize); for (let i = 0; i < a.count; i++) for (let j = 0; j < a.itemSize; j++) f[i * a.itemSize + j] = a.getComponent(i, j); g.setAttribute(k, new THREE.BufferAttribute(f, a.itemSize)); } }
      g.applyMatrix4(mesh.matrixWorld);
      const role = wheel ? 'wheel' : /glass/i.test(mat.name) ? 'glass' : /optics/i.test(mat.name) ? 'optics' : 'body';
      parts.push({geom: g, mat, role});
    });
    take(b, false); for (const w of owner.get(b)!) take(w, true);
    // the long axis in plan (principal axis of the body's vertices)
    const body = parts.filter(p => p.role === 'body').map(p => p.geom.attributes.position as THREE.BufferAttribute);
    let n = 0, mx = 0, mz = 0; for (const a of body) for (let i = 0; i < a.count; i++) { mx += a.getX(i); mz += a.getZ(i); n++; } mx /= n; mz /= n;
    let sxx = 0, sxz = 0, szz = 0; for (const a of body) for (let i = 0; i < a.count; i++) { const dx = a.getX(i) - mx, dz = a.getZ(i) - mz; sxx += dx * dx; sxz += dx * dz; szz += dz * dz; }
    const ang = .5 * Math.atan2(2 * sxz, sxx - szz); let ux = Math.cos(ang), uz = Math.sin(ang);
    // front is where the lamps are not red: rear lamps are red in the optics texture
    let red = 0, redN = 0, white = 0, whiteN = 0;
    for (const p of parts.filter(p => p.role === 'optics')) {
      const t = texData(p.mat.map), pos = p.geom.attributes.position, uv = p.geom.attributes.uv, idx = p.geom.index!;
      for (let i = 0; i < idx.count; i += 3) {
        let x = 0, z = 0, u = 0, v = 0; for (let k = 0; k < 3; k++) { const j = idx.getX(i + k); x += pos.getX(j) / 3; z += pos.getZ(j) / 3; u += uv.getX(j) / 3; v += uv.getY(j) / 3; }
        const [r, g, bb] = sample(t, u, v, p.mat.map?.flipY ?? false), s = (x - mx) * ux + (z - mz) * uz;
        if (r > 90 && r > 1.6 * g && r > 1.6 * bb) { red += s; redN++; } else if (r + g + bb > 450) { white += s; whiteN++; }
      }
    }
    if (redN && whiteN && red / redN > white / whiteN) { ux = -ux; uz = -uz; }
    // car frame: x forward, z to the car's right... three's convention here: x forward, y up, z = forward x up
    const R = new THREE.Matrix4().makeBasis(new THREE.Vector3(ux, 0, uz), new THREE.Vector3(0, 1, 0), new THREE.Vector3(-uz, 0, ux)).transpose();
    for (const p of parts) p.geom.applyMatrix4(R);
    const box = new THREE.Box3(); for (const p of parts) { p.geom.computeBoundingBox(); box.union(p.geom.boundingBox!); }
    const c = box.getCenter(new THREE.Vector3()); for (const p of parts) p.geom.translate(-c.x, -box.min.y, -c.z);
    const size = box.getSize(new THREE.Vector3());
    cars.push({name: b.name.replace(/_?body$/i, '').toLowerCase(), parts, L: size.x, W: size.z, H: size.y});
    console.log(b.name, 'redN', redN, 'whiteN', whiteN, 'L W H', size.x.toFixed(2), size.z.toFixed(2), size.y.toFixed(2), 'wheels', owner.get(b)!.length);
  }
  return cars;
}

/** A contact sheet: every car from the side and from front-quarter, to check orientation and textures. */
(window as any).__sheet = async () => {
  const cars = extract(await load());
  const r = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true}); r.setSize(1500, 900); r.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#9aa4ab');
  scene.add(new THREE.HemisphereLight('#ffffff', '#555555', 2)); const sun = new THREE.DirectionalLight('#ffffff', 2.5); sun.position.set(5, 10, 7); scene.add(sun);
  cars.forEach((car, i) => {
    const g = new THREE.Group(); for (const p of car.parts) g.add(new THREE.Mesh(p.geom, p.mat));
    g.position.set((i % 5) * 6 - 12, 0, Math.floor(i / 5) * 7 - 3.5); scene.add(g);
  });
  const cam = new THREE.PerspectiveCamera(35, 1500 / 900, .1, 200); const images: Record<string, string> = {};
  for (const [n, p] of [['sheet-front3q.png', [22, 14, 26]], ['sheet-side.png', [0, 4, 40]]] as const) { cam.position.set(...p); cam.lookAt(0, 0, 0); r.render(scene, cam); images[n] = r.domElement.toDataURL('image/png'); }
  return {images, cars: cars.map(c => [c.name, +c.L.toFixed(2), +c.W.toFixed(2), +c.H.toFixed(2), c.parts.map(p => p.role + ':' + ((p.geom.index?.count ?? 0) / 3))])};
};

(window as any).__textures = async () => {
  const cars = extract(await load()); const images: Record<string, string> = {};
  for (const car of cars) for (const p of car.parts) if (p.mat.map && (p.role === 'body' || p.role === 'wheel' || p.role === 'optics')) {
    const im = p.mat.map.image as CanvasImageSource & {width: number; height: number}; const c = document.createElement('canvas'); c.width = im.width; c.height = im.height; c.getContext('2d')!.drawImage(im, 0, 0);
    images[`tex-${car.name}-${p.role}.png`] = c.toDataURL('image/png');
  }
  return {images, n: Object.keys(images).length, flip: cars[0].parts[0].mat.map?.flipY};
};

(window as any).__uvRanges = async () => extract(await load()).map(c => [c.name, c.parts.filter(p => p.geom.attributes.uv).map(p => { const a = p.geom.attributes.uv; let u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9; for (let i = 0; i < a.count; i++) { u0 = Math.min(u0, a.getX(i)); u1 = Math.max(u1, a.getX(i)); v0 = Math.min(v0, a.getY(i)); v1 = Math.max(v1, a.getY(i)); } return p.role + ' ' + [u0, u1, v0, v1].map(x => x.toFixed(3)).join(','); })]);

// ---------------------------------------------------------------------------------------------------------------
// The bake: one colour atlas (paint neutralised to grey where the paint mask is set), one surface atlas
// (R paint mask, G roughness, B metalness), and per car three levels of detail of its opaque parts and its glass.
import {mergeGeometries, mergeVertices} from 'three/addons/utils/BufferGeometryUtils.js';
import {GLTFExporter} from 'three/addons/exporters/GLTFExporter.js';
// @ts-ignore the simplifier ships with the viewer's dependencies (meshoptimizer 0.22, MIT)
import {MeshoptSimplifier} from '/@fs/Users/palash/Documents/Codex/2026-09-08/i-want-to-create-a-heuristic/work/king-street/node_modules/.pnpm/meshoptimizer@0.22.0/node_modules/meshoptimizer/meshopt_simplifier.module.js';

const ATLAS = 2048, PAINT_GAIN = 1.25;
const lum = (r: number, g: number, b: number) => .2126 * r + .7152 * g + .0722 * b;
/** smoothstep from edge e0 to edge e1 (e0 > e1 gives the falling step). */
const sstep = (e0: number, e1: number, x: number) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };

/** The paint colour of a body texture: the heaviest colour cluster on the car's upward and sideways faces above the
 *  wheel centres (roof, bonnet, doors), weighted by area. */
function paintOf(t: NonNullable<ReturnType<typeof texData>>, geom: THREE.BufferGeometry, H: number) {
  const pos = geom.attributes.position, uv = geom.attributes.uv, idx = geom.index!, pts: number[][] = [], w: number[] = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3();
  for (let i = 0; i < idx.count; i += 3) {
    const i0 = idx.getX(i), i1 = idx.getX(i + 1), i2 = idx.getX(i + 2);
    a.fromBufferAttribute(pos, i0); b.fromBufferAttribute(pos, i1); c.fromBufferAttribute(pos, i2);
    n.copy(b).sub(a).cross(c.clone().sub(a)); const area = n.length() / 2; n.normalize();
    if ((a.y + b.y + c.y) / 3 < .4 * H || n.y < -.3) continue;
    const u = (uv.getX(i0) + uv.getX(i1) + uv.getX(i2)) / 3, v = (uv.getY(i0) + uv.getY(i1) + uv.getY(i2)) / 3;
    pts.push(sample(t, u, v, false)); w.push(area);
  }
  // k-means, k = 4, seeded from the heaviest samples
  const order = w.map((x, i) => i).sort((i, j) => w[j] - w[i]); let cent = [0, 3, 7, 15].map(k => [...pts[order[Math.min(k, order.length - 1)]]]);
  let mass = [0, 0, 0, 0];
  for (let it = 0; it < 12; it++) {
    const sum = cent.map(() => [0, 0, 0]); mass = [0, 0, 0, 0];
    pts.forEach((p, i) => { let best = 0, bd = 1e9; cent.forEach((q, k) => { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2; if (d < bd) { bd = d; best = k; } }); mass[best] += w[i]; for (let j = 0; j < 3; j++) sum[best][j] += p[j] * w[i]; });
    cent = sum.map((s, k) => mass[k] ? s.map(x => x / mass[k]) : cent[k]);
  }
  // the black panels behind the windows weigh a lot: the heaviest cluster that is not near black, unless the car is black
  const total = mass.reduce((a, b) => a + b), lit = [0, 1, 2, 3].filter(k => Math.max(...cent[k]) > 14 && mass[k] > .15 * total);
  const chroma = (k: number) => Math.max(...cent[k]) - Math.min(...cent[k]), coloured = lit.filter(k => chroma(k) > 25), pool = coloured.length ? coloured : lit;
  const k = pool.length ? pool.reduce((a, b) => mass[a] >= mass[b] ? a : b) : mass.indexOf(Math.max(...mass)); return cent[k];
}

/** Paint weight of one pixel: the same hue direction as the paint at a plausible brightness, or, for a near-black or
 *  near-grey paint, the same brightness band. */
function paintWeight(r: number, g: number, b: number, p: number[]) {
  const pl = lum(p[0], p[1], p[2]), pc = Math.max(p[0], p[1], p[2]) - Math.min(p[0], p[1], p[2]);
  const s = (r * p[0] + g * p[1] + b * p[2]) / Math.max(1, p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  if (pc < 30) {   // achromatic paint (black, grey, white): brightness band, low chroma
    const c = Math.max(r, g, b) - Math.min(r, g, b), l = lum(r, g, b);
    return sstep(40, 18, c) * sstep(pl * .45 + 6, pl * .7 + 8, l) * sstep(Math.max(250, pl * 1.9), Math.max(200, pl * 1.5), l);
  }
  const res = Math.hypot(r - s * p[0], g - s * p[1], b - s * p[2]) / Math.max(8, Math.hypot(r, g, b));
  return sstep(.3, .12, res) * sstep(.1, .25, s);
}

type Tile = {key: unknown; w: number; h: number; x: number; y: number; draw: (c: CanvasRenderingContext2D, m: CanvasRenderingContext2D) => void};

(window as any).__bake = async () => {
  await MeshoptSimplifier.ready;
  // the supercar stands in for no parking type; it would only cost download
  const cars = extract(await load()).filter(c => c.name !== 'sport'), SCALE = 1.08;
  for (const car of cars) { for (const p of car.parts) p.geom.scale(SCALE, SCALE, SCALE); car.L *= SCALE; car.W *= SCALE; car.H *= SCALE; }
  // tiles: one per distinct texture (bodies and lamps at 512, wheels at 256), one small solid tile per untextured material
  const tiles = new Map<unknown, Tile>(); const paintLog: Record<string, number[]> = {};
  for (const car of cars) {
    const bodyGeom = mergeGeometries(car.parts.filter(p => p.role === 'body').map(p => p.geom.index ? p.geom.toNonIndexed() : p.geom).map(g => { const x = new THREE.BufferGeometry(); x.setAttribute('position', g.attributes.position); if (g.attributes.uv) x.setAttribute('uv', g.attributes.uv); return x; }));
    for (const p of car.parts) {
      if (p.role === 'glass') continue; const m = p.mat, key = m.map ?? m;
      if (tiles.has(key)) continue;
      const size = m.map ? (p.role === 'wheel' && !/body/i.test(m.name) ? 256 : 512) : 32;
      const isBody = /body/i.test(m.name);
      let paint: number[] | null = null;
      if (isBody && m.map) { const td = texData(m.map)!; const bg = mergeVertices(bodyGeom); paint = paintOf(td, bg, car.H); paintLog[car.name] = paint.map(Math.round); }
      if (isBody && !m.map) paint = m.color.toArray().map(v => v * 255);
      const mr = (m.roughnessMap ?? null) as THREE.Texture | null, rough = m.roughness, metal = m.metalness, base = m.color.clone();
      tiles.set(key, {key, w: size, h: size, x: 0, y: 0, draw: (c, s) => {
        const t = tiles.get(key)!;
        if (m.map) c.drawImage(m.map.image as CanvasImageSource, t.x, t.y, t.w, t.h); else { c.fillStyle = '#' + base.getHexString(); c.fillRect(t.x, t.y, t.w, t.h); }
        if (mr) s.drawImage(mr.image as CanvasImageSource, t.x, t.y, t.w, t.h); else { s.fillStyle = '#ffffff'; s.fillRect(t.x, t.y, t.w, t.h); }
        const ci = c.getImageData(t.x, t.y, t.w, t.h), si = s.getImageData(t.x, t.y, t.w, t.h), cd = ci.data, sd = si.data;
        const pl = paint ? Math.max(8, lum(paint[0], paint[1], paint[2])) : 1;
        for (let k = 0; k < cd.length; k += 4) {
          // the surface atlas: glTF keeps roughness in G and metalness in B; the factors are baked in
          const ro = Math.round(sd[k + 1] * rough), me = Math.round(sd[k + 2] * metal);
          let mask = 0;
          if (!m.map && isBody) { cd[k] = cd[k + 1] = cd[k + 2] = Math.round(255 / PAINT_GAIN); mask = 255; }
          else if (paint) {
            const r = cd[k], g = cd[k + 1], b = cd[k + 2], wgt = paintWeight(r, g, b, paint);
            if (wgt > 0) { const grey = Math.min(255, lum(r, g, b) / pl * 255 / PAINT_GAIN); cd[k] = r + (grey - r) * wgt; cd[k + 1] = g + (grey - g) * wgt; cd[k + 2] = b + (grey - b) * wgt; mask = Math.round(wgt * 255); }
          }
          sd[k] = mask; sd[k + 1] = ro; sd[k + 2] = me; sd[k + 3] = 255;
        }
        c.putImageData(ci, t.x, t.y); s.putImageData(si, t.x, t.y);
      }});
    }
  }
  // shelf packing, largest first
  let x = 0, y = 0, row = 0; for (const t of [...tiles.values()].sort((a, b) => b.h - a.h)) { if (x + t.w > ATLAS) { x = 0; y += row; row = 0; } t.x = x; t.y = y; x += t.w; row = Math.max(row, t.h); }
  const usedH = Math.pow(2, Math.ceil(Math.log2(y + row)));
  const colour = document.createElement('canvas'), surf = document.createElement('canvas'); colour.width = surf.width = ATLAS; colour.height = surf.height = usedH;
  const cc = colour.getContext('2d', {willReadFrequently: true})!, sc = surf.getContext('2d', {willReadFrequently: true})!;
  for (const t of tiles.values()) t.draw(cc, sc);
  // geometry: atlas uvs, then the levels of detail
  const exportScene = new THREE.Scene(), stats: Record<string, number[]> = {}, plain = new THREE.MeshBasicMaterial();
  for (const car of cars) {
    const opaque: THREE.BufferGeometry[] = [], glass: THREE.BufferGeometry[] = [];
    for (const p of car.parts) {
      let g = p.geom.index ? p.geom.toNonIndexed() : p.geom.clone();
      if (p.role === 'glass') { g.deleteAttribute('uv'); glass.push(g); continue; }
      const t = tiles.get(p.mat.map ?? p.mat)!, uv = g.attributes.uv as THREE.BufferAttribute | undefined, out = new Float32Array(g.attributes.position.count * 2);
      for (let i = 0; i < g.attributes.position.count; i++) {
        // half-texel inset so bilinear filtering never reads the neighbouring tile
        const u = uv ? THREE.MathUtils.clamp(uv.getX(i), .5 / t.w, 1 - .5 / t.w) : .5, v = uv ? THREE.MathUtils.clamp(uv.getY(i), .5 / t.h, 1 - .5 / t.h) : .5;
        out[i * 2] = (t.x + u * t.w) / ATLAS; out[i * 2 + 1] = (t.y + v * t.h) / usedH;
      }
      g.setAttribute('uv', new THREE.BufferAttribute(out, 2)); if (!g.attributes.normal) g.computeVertexNormals(); opaque.push(g);
    }
    const o0 = mergeVertices(mergeGeometries(opaque), 1e-4), g0 = mergeVertices(mergeGeometries(glass), 1e-4);
    const lods = [o0, simplify(o0, .26, .01), coarse(o0, 500)], glods = [g0, simplify(g0, .5, .01), coarse(g0, 24)];
    lods.forEach((g, i) => { const m = new THREE.Mesh(g, plain); m.name = `${car.name}__opaque__${i}`; exportScene.add(m); });
    glods.forEach((g, i) => { const m = new THREE.Mesh(g, plain); m.name = `${car.name}__glass__${i}`; exportScene.add(m); });
    stats[car.name] = [...lods, ...glods].map(g => g.index!.count / 3).concat([+car.L.toFixed(2), +car.W.toFixed(2), +car.H.toFixed(2)]);
  }
  // compact binary: per mesh positions int16 over the car's box, normals int8, uvs uint16, indices uint16; gzipped
  const header: any[] = [], chunks: ArrayBuffer[] = []; let offset = 0;
  const push = (a: ArrayBufferView) => { const b = a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength) as ArrayBuffer; const pad = (4 - b.byteLength % 4) % 4; chunks.push(b); if (pad) chunks.push(new ArrayBuffer(pad)); const o = offset; offset += b.byteLength + pad; return o; };
  exportScene.children.forEach(o => {
    const g = (o as THREE.Mesh).geometry, p = g.attributes.position, n = g.attributes.normal, uv = g.attributes.uv; g.computeBoundingBox(); const bb = g.boundingBox!;
    const lo = bb.min.toArray(), span = bb.getSize(new THREE.Vector3()).toArray().map(v => Math.max(v, 1e-4));
    const P = new Int16Array(p.count * 3), N = new Int8Array(p.count * 4), U = new Uint16Array(uv ? p.count * 2 : 0);
    for (let i = 0; i < p.count; i++) {
      for (let j = 0; j < 3; j++) P[i * 3 + j] = Math.round(((p.getComponent(i, j) - lo[j]) / span[j]) * 65535 - 32768);
      for (let j = 0; j < 3; j++) N[i * 4 + j] = Math.round(THREE.MathUtils.clamp(n.getComponent(i, j), -1, 1) * 127);
      if (uv) for (let j = 0; j < 2; j++) U[i * 2 + j] = Math.round(THREE.MathUtils.clamp(uv.getComponent(i, j), 0, 1) * 65535);
    }
    const I = new Uint16Array(g.index!.array);
    header.push({name: o.name, count: p.count, triangles: I.length / 3, lo, span, position: push(P), normal: push(N), uv: uv ? push(U) : -1, index: push(I), indexCount: I.length});
  });
  const json = new TextEncoder().encode(JSON.stringify({version: 1, atlas: [ATLAS, usedH], paintGain: PAINT_GAIN, cars: cars.map(c => ({name: c.name, length: +c.L.toFixed(3), width: +c.W.toFixed(3), height: +c.H.toFixed(3)})), meshes: header}));
  const lenBuf = new Uint32Array([json.byteLength]), jpad = (4 - json.byteLength % 4) % 4;
  const blob = new Blob([lenBuf, json, new Uint8Array(jpad), ...chunks]);
  const gz = await new Response(blob.stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
  let bin = ''; const bytes = new Uint8Array(gz); for (let i = 0; i < bytes.length; i += 32768) bin += String.fromCharCode(...bytes.subarray(i, i + 32768));
  const raw = await blob.arrayBuffer();
  return {b64: btoa(bin), rawBytes: raw.byteLength, images: {'atlas-colour.webp': colour.toDataURL('image/webp', .9), 'atlas-surface.webp': surf.toDataURL('image/webp', .92)}, stats, paintLog, atlas: [ATLAS, usedH], tiles: tiles.size};
};

function simplify(g: THREE.BufferGeometry, ratio: number, error: number) {
  const idx = new Uint32Array(g.index!.array), pos = g.attributes.position.array as Float32Array;
  const target = Math.floor(idx.length * ratio / 3) * 3;
  const [out] = MeshoptSimplifier.simplify(idx, pos, 3, target, error, []);
  const s = g.clone(); s.setIndex(new THREE.BufferAttribute(out, 1)); return s;
}

(window as any).__ready = true;
(window as any).__dbg = async () => {
  const cars = extract(await load()); const out: any = {};
  for (const car of cars.filter(c => ['sedan', 'hatchback', 'compact'].includes(c.name))) {
    const p = car.parts.find(p => p.role === 'body')!, t = texData(p.mat.map)!, g = p.geom, uv = g.attributes.uv, idx = g.index!;
    const hist = (flip: boolean) => { let blk = 0, col = 0; for (let i = 0; i < idx.count; i += 3) { const j = idx.getX(i); const [r, gg, b] = sample(t, uv.getX(j), uv.getY(j), flip); if (r + gg + b < 60) blk++; else col++; } return [blk, col]; };
    out[car.name] = {noflip: hist(false), flip: hist(true), img: [t.w, t.h], centre: sample(t, .5, .5, false), mapFlip: p.mat.map!.flipY, chan: p.mat.map!.channel, uvType: (g.attributes.uv.array as any).constructor.name};
  }
  return out;
};

/** The far level: welded by position alone (texture seams no longer hold the simplifier back; a seam's texture smears,
 *  unseen at that distance), simplified to about `tris` triangles, normals recomputed. */
function coarse(g: THREE.BufferGeometry, tris: number) {
  const w = g.clone(); w.deleteAttribute('normal'); const uv = w.attributes.uv; w.deleteAttribute('uv');
  const m = mergeVertices(w, 1e-3);
  // carry a uv over to each welded vertex from the nearest original vertex at that position
  const map = new Map<string, number>(); const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) map.set(`${p.getX(i).toFixed(3)},${p.getY(i).toFixed(3)},${p.getZ(i).toFixed(3)}`, i);
  const mp = m.attributes.position, muv = new Float32Array(mp.count * 2);
  for (let i = 0; i < mp.count; i++) { const j = map.get(`${mp.getX(i).toFixed(3)},${mp.getY(i).toFixed(3)},${mp.getZ(i).toFixed(3)}`) ?? 0; if (uv) { muv[i * 2] = uv.getX(j); muv[i * 2 + 1] = uv.getY(j); } }
  const idx = new Uint32Array(m.index!.array), pos = mp.array as Float32Array;
  const [out] = MeshoptSimplifier.simplify(idx, pos, 3, tris * 3, .2, []);
  const s = new THREE.BufferGeometry(); s.setAttribute('position', mp); if (uv) s.setAttribute('uv', new THREE.BufferAttribute(muv, 2)); s.setIndex(new THREE.BufferAttribute(out, 1)); s.computeVertexNormals();
  return s;
}
