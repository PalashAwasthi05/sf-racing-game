import * as THREE from 'three';
import {mergeGeometries,toCreasedNormals} from 'three/addons/utils/BufferGeometryUtils.js';

// Parked cars along the kerbs. Where they stand comes with the world (areas/parking.py: the kerbs of the built road
// surface on the streets OpenStreetMap marks for parking, clear of corners, crossings and the garage doors the
// photographs show). The cars are modelled (public/vehicle/parked/, see modelledCars below); when those files are
// missing, seven simple body types are drawn instead, each a side profile extruded across its width with rounded edges,
// the cabin drawn in toward the roof, wheel arches cut over the wheels, glass standing proud of the body, a dark
// B-pillar, grille, mirrors, lights and plates. Either way every car of one material is one BatchedMesh, each car culled
// on its own and painted its own colour. Nothing here is location specific.

export type ParkedCar = {x: number; y: number; z: number; heading: number; pitch: number; type: string; paint: string};
type Kind = {name: string; length: number; width: number; height: number};
type Shape = {top: [number, number][]; glass: [number, number][]; axles: [number, number]; wheel: number; bottom: number; pillar: boolean};

// Side profiles as fractions of length (from the rear bumper) and height, rear to front along the top; the glass
// outline; the axles as fractions of length; the wheel radius in metres; the sill as a fraction of height.
const SHAPES: Record<string, Shape> = {
  sedan: {top: [[0, .42], [.01, .62], [.05, .69], [.22, .71], [.3, .74], [.42, 1], [.66, 1], [.79, .72], [.95, .65], [1, .55], [1, .3]], glass: [[.31, .745], [.425, .975], [.655, .975], [.775, .725]], axles: [.19, .81], wheel: .33, bottom: .13, pillar: true},
  hatch: {top: [[0, .45], [.01, .64], [.04, .78], [.09, .99], [.58, 1], [.74, .72], [.95, .64], [1, .54], [1, .3]], glass: [[.045, .78], [.095, .965], [.57, .975], [.72, .735]], axles: [.17, .83], wheel: .32, bottom: .13, pillar: true},
  suv: {top: [[0, .4], [.01, .64], [.03, .72], [.06, 1], [.62, 1], [.76, .71], [.97, .64], [1, .55], [1, .3]], glass: [[.065, .725], [.075, .965], [.61, .97], [.745, .72]], axles: [.18, .82], wheel: .37, bottom: .16, pillar: true},
  wagon: {top: [[0, .44], [.01, .62], [.02, .75], [.05, .98], [.64, 1], [.77, .72], [.96, .64], [1, .54], [1, .3]], glass: [[.055, .755], [.06, .96], [.63, .97], [.755, .73]], axles: [.18, .82], wheel: .33, bottom: .13, pillar: true},
  pickup: {top: [[0, .36], [.005, .6], [.39, .6], [.4, .62], [.4, 1], [.62, 1], [.72, .69], [.97, .63], [1, .55], [1, .32]], glass: [[.408, .705], [.408, .965], [.615, .965], [.705, .7]], axles: [.17, .8], wheel: .4, bottom: .18, pillar: false},
  van: {top: [[0, .3], [.005, .95], [.03, 1], [.78, 1], [.88, .64], [.98, .56], [1, .46], [1, .3]], glass: [[.71, .64], [.775, .95], [.865, .655]], axles: [.17, .84], wheel: .34, bottom: .13, pillar: false},
  coupe: {top: [[0, .44], [.01, .6], [.06, .68], [.25, .72], [.45, 1], [.64, 1], [.79, .7], [.97, .6], [1, .5], [1, .3]], glass: [[.27, .73], [.455, .975], [.635, .975], [.775, .705]], axles: [.18, .81], wheel: .32, bottom: .14, pillar: false},
};
const BEVEL = .04, BEVEL_DEPTH = .05, TUMBLEHOME = .13, PLAN_TAPER = .09;

/** The triangles of a non-indexed geometry that have area. A zero-area triangle (the extrusion leaves a few where a
 *  rounded corner meets a straight edge) takes a zero normal, which the shader's normalize turns into NaN; the bloom
 *  spread one NaN pixel into a black block over a third of the screen. */
function dropDegenerate(g: THREE.BufferGeometry) {
  const p = g.attributes.position.array as ArrayLike<number>, keep: number[] = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let t = 0; t < p.length; t += 9) {
    a.set(p[t], p[t + 1], p[t + 2]); b.set(p[t + 3], p[t + 4], p[t + 5]); c.set(p[t + 6], p[t + 7], p[t + 8]);
    if (b.sub(a).cross(c.sub(a)).lengthSq() > 1e-12) for (let k = 0; k < 9; k++) keep.push(p[t + k]);
  }
  const out = new THREE.BufferGeometry(); out.setAttribute('position', new THREE.Float32BufferAttribute(keep, 3)); return out;
}

/** Non-indexed, no uv, normals smoothed across rounded edges and kept sharp across creases (over 35 degrees). */
function finish(g: THREE.BufferGeometry, colour?: THREE.Color) {
  let n = dropDegenerate(g.index ? g.toNonIndexed() : g); n = toCreasedNormals(n, .61);
  if (n.index) n = n.toNonIndexed();
  if (colour) { const c = new Float32Array(n.attributes.position.count * 3); for (let i = 0; i < c.length; i += 3) { c[i] = colour.r; c[i + 1] = colour.g; c[i + 2] = colour.b; } n.setAttribute('color', new THREE.BufferAttribute(c, 3)); }
  return n;
}

/** The cabin narrows toward the roof: points above the belt line are drawn in across the car. */
function tumblehome(g: THREE.BufferGeometry, belt: number, top: number) {
  const p = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < p.count; i++) { const y = p.getY(i); if (y > belt) p.setZ(i, p.getZ(i) * (1 - TUMBLEHOME * Math.min(1, (y - belt) / Math.max(.01, top - belt)))); }
  p.needsUpdate = true;
}

function box(w: number, h: number, d: number, x: number, y: number, z: number) { const b = new THREE.BoxGeometry(w, h, d); b.translate(x, y, z); return b; }

/** The body narrows toward its bumpers in plan, as a car's corners round off (z drawn in near the ends). */
function planTaper(g: THREE.BufferGeometry, L: number) {
  const p = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < p.count; i++) { const t = THREE.MathUtils.smoothstep(Math.abs(p.getX(i)), .3 * L, .5 * L + BEVEL); p.setZ(i, p.getZ(i) * (1 - PLAN_TAPER * t * t)); }
  p.needsUpdate = true;
}

/** Continues a shape from its current point pts[0] along pts, every inner corner rounded by a fillet of at most `r` (no
 *  overshoot, unlike a spline through it). pts[0] is not drawn again: a repeated point is a zero-length edge, and the
 *  extrusion's bevel divides by edge lengths (NaN vertices, which the bloom spread into black blocks on screen). */
function filleted(shape: THREE.Shape, pts: [number, number][], r: number) {
  for (let i = 1; i < pts.length; i++) {
    const P = new THREE.Vector2(...pts[i]);
    if (i === pts.length - 1) { shape.lineTo(P.x, P.y); continue; }
    const A = new THREE.Vector2(...pts[i - 1]), B = new THREE.Vector2(...pts[i + 1]);
    const ra = Math.min(r, A.distanceTo(P) * .45, B.distanceTo(P) * .45);
    const p0 = P.clone().add(A.clone().sub(P).setLength(ra)), p1 = P.clone().add(B.clone().sub(P).setLength(ra));
    shape.lineTo(p0.x, p0.y); shape.quadraticCurveTo(P.x, P.y, p1.x, p1.y);
  }
}

/** A wheel's outer face: five silver spokes on a dark ground, coloured per triangle so the pattern stays crisp. */
function spokedRim(radius: number) {
  const disc = new THREE.CircleGeometry(radius, 30).toNonIndexed(), pos = disc.attributes.position, col = new Float32Array(pos.count * 3);
  const silver = new THREE.Color('#c8ccd0'), dark = new THREE.Color('#2b2d30'), hub = new THREE.Color('#9ea3a7');
  for (let t = 0; t < pos.count; t += 3) {
    const cx = (pos.getX(t) + pos.getX(t + 1) + pos.getX(t + 2)) / 3, cy = (pos.getY(t) + pos.getY(t + 1) + pos.getY(t + 2)) / 3;
    const a = (Math.atan2(cy, cx) + Math.PI * 2) % (Math.PI * 2), c = Math.floor(a / (Math.PI / 5)) % 2 === 0 ? silver : dark;
    for (let k = 0; k < 3; k++) { const v = t + k, rr = Math.hypot(pos.getX(v), pos.getY(v)); const cc = rr < radius * .25 ? hub : c; col[v * 3] = cc.r; col[v * 3 + 1] = cc.g; col[v * 3 + 2] = cc.b; }
  }
  disc.setAttribute('color', new THREE.BufferAttribute(col, 3)); return disc;
}

/** One body type's parts by material: paint, glass, dark (tyres, pillar, grille), bright (rims, lights, plates). */
function buildKind(k: Kind) {
  const s = SHAPES[k.name] ?? SHAPES.sedan, L = k.length, H = k.height, W = k.width, yb = s.bottom * H, r = s.wheel, ra = r + .05;
  const outline = new THREE.Shape(); outline.moveTo(0, yb);
  for (const ax of s.axles) {
    const cx = ax * L, b = Math.asin(Math.min(1, (r - yb) / ra)), t0 = Math.PI + b, t1 = -b;
    for (let i = 0; i <= 10; i++) { const t = t0 + (t1 - t0) * i / 10; outline.lineTo(cx + ra * Math.cos(t), r + ra * Math.sin(t)); }
  }
  outline.lineTo(L, yb);
  // up the front and over the top to the rear, every corner rounded (a box with corners cut read as a toy)
  filleted(outline, [[L, yb], ...[...s.top].reverse().map(([x, y]) => [x * L, y * H] as [number, number])], .22);
  const body = new THREE.ExtrudeGeometry(outline, {depth: W - 2 * BEVEL_DEPTH, bevelEnabled: true, bevelThickness: BEVEL_DEPTH, bevelSize: BEVEL, bevelSegments: 2, curveSegments: 4, steps: 1});
  body.translate(-L / 2, 0, -(W - 2 * BEVEL_DEPTH) / 2);
  const belt = Math.min(...s.glass.map(p => p[1])) * H - .02; tumblehome(body, belt, H); planTaper(body, L);
  // glass: the window outline pushed out past the rounded body edge, a little wider than the body
  const gx = s.glass.map(p => p[0] * L), gy = s.glass.map(p => p[1] * H), mx = gx.reduce((a, b) => a + b) / gx.length, my = gy.reduce((a, b) => a + b) / gy.length;
  const glassShape = new THREE.Shape(s.glass.map((_, i) => { const dx = gx[i] - mx, dy = gy[i] - my, d = Math.hypot(dx, dy) || 1; return new THREE.Vector2(gx[i] + dx / d * (BEVEL + .02), gy[i] + dy / d * (BEVEL + .02)); }));
  const glass = new THREE.ExtrudeGeometry(glassShape, {depth: W + .02, bevelEnabled: false, curveSegments: 1}); glass.translate(-L / 2, 0, -(W + .02) / 2); tumblehome(glass, belt, H); planTaper(glass, L);
  const front = L / 2 + BEVEL, rear = -L / 2 - BEVEL, xs = gx.map(x => x - L / 2), aBase = Math.max(...xs);
  const paint = [body, box(.14, .1, .2, aBase - .1, belt + .08, W / 2 + .08), box(.14, .1, .2, aBase - .1, belt + .08, -W / 2 - .08)];
  const dark: THREE.BufferGeometry[] = [box(.05, .16 * H, .5 * W, front + .005, .4 * H, 0)], bright: THREE.BufferGeometry[] = [];
  if (s.pillar) {
    const xm = (Math.min(...xs) + aBase) / 2 + .1; dark.push(box(.08, (Math.max(...gy) - Math.min(...gy)) * .95, W + .05, xm, my, 0)); tumblehome(dark[dark.length - 1], belt, H);
    // door seams: the front door's leading edge, the B-pillar line and the rear door's trailing edge, sill to belt
    for (const x of [aBase - .02, xm, Math.min(...xs) + .05]) for (const side of [-1, 1]) dark.push(box(.012, belt - yb - .12, .008, x, (belt + yb) / 2 + .02, side * (W / 2 + .002)));
  }
  const tyre = new THREE.CylinderGeometry(r, r, .22, 16); tyre.rotateX(Math.PI / 2);
  const rim = spokedRim(r * .64);
  for (const ax of s.axles) for (const side of [-1, 1]) {
    dark.push(tyre.clone().translate(ax * L - L / 2, r, side * (W / 2 - .13)));
    const face = rim.clone(); if (side < 0) face.rotateY(Math.PI); face.translate(ax * L - L / 2, r, side * (W / 2 - .13 + .113));
    face.deleteAttribute('uv'); face.computeVertexNormals(); bright.push(face);
  }
  const lightY = Math.max(.5, s.top[s.top.length - 2][1] - .06) * H, tailY = Math.max(.5, s.top[1][1] - .06) * H;
  for (const side of [-1, 1]) {
    bright.push(finish(box(.05, .08, .3, front, lightY, side * (W / 2 - .26)), new THREE.Color('#f1eee4')));
    bright.push(finish(box(.05, .1, .32, rear, tailY, side * (W / 2 - .22)), new THREE.Color('#8e1018')));
  }
  bright.push(finish(box(.012, .11, .31, front + .03, .33 * H, 0), new THREE.Color('#ecebe6')), finish(box(.012, .11, .31, rear - .012, .44 * H, 0), new THREE.Color('#ecebe6')));
  return {paint: mergeGeometries(paint.map(g => finish(g))), glass: finish(glass), dark: mergeGeometries(dark.map(g => finish(g))), bright: mergeGeometries(bright)};
}

export type ParkedCars = {group: THREE.Group; materials: THREE.Material[]; reflective: THREE.MeshStandardMaterial[]; circles: number[]; count: number};

// ---------------------------------------------------------------------------------------------------------------
// Modelled cars (public/vehicle/parked/, baked by viewer-bench/bake from "Generic passenger car pack" by Comrade1280,
// CC BY 4.0; see CREDITS.txt there). Ten unbranded cars, each in three levels of detail; one colour atlas whose paint is
// neutral grey under a paint mask (the surface atlas's red channel; green roughness, blue metalness), so every parked
// car takes its own paint and its trim, lamps and tyres keep theirs. All cars are two draws: opaque parts and glass.

/** Which modelled cars stand in for each parking body type (parking.py's types). */
const MODELS: Record<string, string[]> = {sedan: ['sedan'], hatch: ['hatchback', 'compact'], suv: ['suv', 'offroad'], wagon: ['wagon'], pickup: ['pickup'], van: ['minivan'], coupe: ['coupe']};
/** Camera distances (m) out to which the full and the middle level of detail are drawn. */
export const PARKED_LOD = {full: 40, middle: 140};

type Kit = {geometry: Map<string, THREE.BufferGeometry>; colour: THREE.Texture; surface: THREE.Texture; paintGain: number};

async function loadKit(base: string): Promise<Kit | null> {
  try {
    const [bin, colour, surface] = await Promise.all([
      fetch(base + 'cars.bin.gz').then(async r => {
        if (!r.ok) throw new Error('cars.bin.gz ' + r.status);
        // a server may already have undone the gzip (Content-Encoding); decompress only what still starts with its magic
        const raw = await r.arrayBuffer(), b = new Uint8Array(raw, 0, 2);
        return b[0] === 0x1f && b[1] === 0x8b ? new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer() : raw;
      }),
      new THREE.TextureLoader().loadAsync(base + 'atlas-colour.webp'), new THREE.TextureLoader().loadAsync(base + 'atlas-surface.webp')]);
    const n = new Uint32Array(bin, 0, 1)[0], head = JSON.parse(new TextDecoder().decode(new Uint8Array(bin, 4, n))), start = 4 + n + (4 - n % 4) % 4;
    const geometry = new Map<string, THREE.BufferGeometry>();
    for (const m of head.meshes as {name: string; count: number; lo: number[]; span: number[]; position: number; normal: number; uv: number; index: number; indexCount: number}[]) {
      const P = new Int16Array(bin, start + m.position, m.count * 3), N = new Int8Array(bin, start + m.normal, m.count * 4), pos = new Float32Array(m.count * 3), nrm = new Float32Array(m.count * 3);
      for (let i = 0; i < m.count; i++) for (let j = 0; j < 3; j++) { pos[i * 3 + j] = m.lo[j] + (P[i * 3 + j] + 32768) / 65535 * m.span[j]; nrm[i * 3 + j] = N[i * 4 + j] / 127; }
      const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
      if (m.uv >= 0) g.setAttribute('uv', new THREE.BufferAttribute(new Uint16Array(bin, start + m.uv, m.count * 2), 2, true));
      else g.setAttribute('uv', new THREE.BufferAttribute(new Uint16Array(m.count * 2), 2, true));   // glass: one material, no texture
      g.setIndex(new THREE.BufferAttribute(new Uint16Array(bin, start + m.index, m.indexCount), 1)); geometry.set(m.name, g);
    }
    for (const t of [colour, surface]) { t.flipY = false; t.anisotropy = 4; t.needsUpdate = true; }
    colour.colorSpace = THREE.SRGBColorSpace;
    return {geometry, colour, surface, paintGain: head.paintGain};
  } catch (e) { console.warn('parked car models unavailable, drawing the simple bodies', e); return null; }
}

/** The paint mask, as a branch in three's colour and roughness chunks that only materials defining PARKED_PAINT take
 *  (so it holds whatever onBeforeCompile patches other modules chain onto the material: cascades, occlusion). Under the
 *  mask the grey atlas takes the car's own colour (its batch colour); trim, lamps and tyres keep theirs. The pack's maps
 *  go to zero roughness on some trim, and a mirror there blooms into a blob: roughness is held at 0.2 or more. */
function installPaintChunks() {
  const c = THREE.ShaderChunk as unknown as Record<string, string>;
  if (c.color_fragment.includes('PARKED_PAINT')) return;
  c.color_fragment = `#if defined( PARKED_PAINT ) && defined( USE_COLOR ) && defined( USE_ROUGHNESSMAP )
  { float paintMask = texture2D( roughnessMap, vRoughnessMapUv ).r; diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * vColor * PARKED_PAINT, paintMask ); }
#else
${c.color_fragment}
#endif`;
  c.roughnessmap_fragment = `${c.roughnessmap_fragment}
#ifdef PARKED_PAINT
  roughnessFactor = max( roughnessFactor, .2 );
#endif`;
}

function modelledCars(data: {types: Kind[]; cars: ParkedCar[]}, kit: Kit): ParkedCars {
  const paint = new THREE.MeshPhysicalMaterial({map: kit.colour, roughnessMap: kit.surface, metalnessMap: kit.surface, roughness: 1, metalness: 1, clearcoat: 1, clearcoatMap: kit.surface, clearcoatRoughness: .14});
  paint.userData.envIntensity = 1.25;
  installPaintChunks(); const withDefines = paint as unknown as {defines?: Record<string, string>}; withDefines.defines = {...withDefines.defines, PARKED_PAINT: kit.paintGain.toFixed(3)};
  const glassMat = new THREE.MeshStandardMaterial({color: '#0b1014', metalness: .1, roughness: .05}); glassMat.userData.envIntensity = 1.4;
  const group = new THREE.Group(); group.name = 'parked_cars'; group.userData.layer = 'street_objects';
  const models = [...new Set(Object.values(MODELS).flat())].filter(m => kit.geometry.has(`${m}__opaque__0`));
  const make = (part: 'opaque' | 'glass', material: THREE.Material) => {
    let verts = 0, index = 0; for (const m of models) for (let l = 0; l < 3; l++) { const g = kit.geometry.get(`${m}__${part}__${l}`)!; verts += g.attributes.position.count; index += g.index!.count; }
    const mesh = new THREE.BatchedMesh(data.cars.length, verts, index, material); mesh.name = 'parked_cars__' + part; mesh.castShadow = true; mesh.receiveShadow = true; mesh.sortObjects = false;
    const ids = new Map(models.map(m => [m, [0, 1, 2].map(l => mesh.addGeometry(kit.geometry.get(`${m}__${part}__${l}`)!))]));
    group.add(mesh); return {mesh, ids};
  };
  const opaque = make('opaque', paint), glass = make('glass', glassMat);
  const matrix = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(0, 0, 0, 'YZX'), one = new THREE.Vector3(1, 1, 1), at = new THREE.Vector3(), colour = new THREE.Color();
  const circles: number[] = [], placed: {x: number; z: number; model: string; o: number; g: number; lod: number}[] = [];
  data.cars.forEach((c, n) => {
    const options = (MODELS[c.type] ?? MODELS.sedan).filter(m => models.includes(m)); if (!options.length) return;
    // a stable pick among the stand-ins, from where the car stands
    const model = options[Math.abs(Math.floor(c.x * 7.31 + c.y * 3.17 + n * .5)) % options.length];
    e.set(0, c.heading, c.pitch); q.setFromEuler(e); at.set(c.x, c.z, -c.y); matrix.compose(at, q, one);
    const o = opaque.mesh.addInstance(opaque.ids.get(model)![2]), g = glass.mesh.addInstance(glass.ids.get(model)![2]);
    opaque.mesh.setMatrixAt(o, matrix); glass.mesh.setMatrixAt(g, matrix); opaque.mesh.setColorAt(o, colour.set(c.paint));
    placed.push({x: c.x, z: -c.y, model, o, g, lod: 2});
    const k = data.types.find(t => t.name === c.type)!, r = k.width / 2, ux = Math.cos(c.heading), uz = -Math.sin(c.heading);
    for (const f of [-1, 0, 1]) { const d = f * (k.length / 2 - r); circles.push(c.x + ux * d, -c.y + uz * d, r); }
  });
  // levels of detail by distance from the viewing camera, refreshed when it has moved a metre (the shadow passes use
  // orthographic cameras and leave them alone)
  // (BatchedMesh draws its culled list from its own onBeforeRender, so that runs after this)
  const last = new THREE.Vector3(Infinity, 0, 0), batchBefore = opaque.mesh.onBeforeRender.bind(opaque.mesh);
  opaque.mesh.onBeforeRender = (renderer, scene, camera, geometry, material, group) => { updateLod(camera); batchBefore(renderer, scene, camera, geometry, material, group); };
  const updateLod = (camera: THREE.Camera) => {
    if (!(camera as THREE.PerspectiveCamera).isPerspectiveCamera || camera.position.distanceToSquared(last) < 1) return;
    last.copy(camera.position); const full = PARKED_LOD.full ** 2, middle = PARKED_LOD.middle ** 2;
    for (const p of placed) {
      const d = (p.x - last.x) ** 2 + (p.z - last.z) ** 2, lod = d < full ? 0 : d < middle ? 1 : 2;
      if (lod !== p.lod) { p.lod = lod; opaque.mesh.setGeometryIdAt(p.o, opaque.ids.get(p.model)![lod]); glass.mesh.setGeometryIdAt(p.g, glass.ids.get(p.model)![lod]); }
    }
  };
  group.updateMatrixWorld(true); group.traverse(o => { o.matrixAutoUpdate = false; });
  return {group, materials: [paint, glassMat], reflective: [paint, glassMat], circles, count: placed.length};
}

/** The parked cars of a world, or null when it has none. Coordinates are the world's (x east, y north, z up). */
export async function loadParkedCars(url: string, models = '/vehicle/parked/'): Promise<ParkedCars | null> {
  const [data, kit] = await Promise.all([fetch(url, {cache: 'no-cache'}).then(r => r.ok ? r.json() as Promise<{types: Kind[]; cars: ParkedCar[]}> : null).catch(() => null), models ? loadKit(models) : Promise.resolve(null)]);
  if (!data || !data.cars?.length) return null;
  if (kit) return modelledCars(data, kit);
  const materials = {
    paint: new THREE.MeshPhysicalMaterial({color: '#ffffff', metalness: .5, roughness: .3, clearcoat: 1, clearcoatRoughness: .05}),
    glass: new THREE.MeshStandardMaterial({color: '#0d1317', metalness: .1, roughness: .06}),
    dark: new THREE.MeshStandardMaterial({color: '#141516', metalness: 0, roughness: .72}),
    bright: new THREE.MeshStandardMaterial({vertexColors: true, metalness: .35, roughness: .32}),
  };
  materials.paint.userData.envIntensity = 1.25; materials.glass.userData.envIntensity = 1.4;
  const kinds = new Map(data.types.map(k => [k.name, buildKind(k)]));
  const group = new THREE.Group(); group.name = 'parked_cars'; group.userData.layer = 'street_objects';
  const parts = ['paint', 'glass', 'dark', 'bright'] as const, meshes: Record<string, THREE.BatchedMesh> = {}, ids: Record<string, Map<string, number>> = {};
  for (const part of parts) {
    let verts = 0; for (const k of kinds.values()) verts += k[part].attributes.position.count;
    const m = new THREE.BatchedMesh(data.cars.length, verts, 0, materials[part]); m.name = 'parked_cars__' + part; m.castShadow = true; m.receiveShadow = true; m.sortObjects = false;
    ids[part] = new Map(); for (const [name, k] of kinds) ids[part].set(name, m.addGeometry(k[part]));
    meshes[part] = m; group.add(m);
  }
  const matrix = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(0, 0, 0, 'YZX'), one = new THREE.Vector3(1, 1, 1), at = new THREE.Vector3(), colour = new THREE.Color();
  const circles: number[] = [];
  for (const c of data.cars) {
    if (!kinds.has(c.type)) continue;
    e.set(0, c.heading, c.pitch); q.setFromEuler(e); at.set(c.x, c.z, -c.y); matrix.compose(at, q, one);
    for (const part of parts) {
      const i = meshes[part].addInstance(ids[part].get(c.type)!); meshes[part].setMatrixAt(i, matrix);
      if (part === 'paint') meshes[part].setColorAt(i, colour.set(c.paint));
    }
    // three circles along the car for the drivable car to run into (car-ground.ts Obstacles: x, z, radius)
    const k = data.types.find(t => t.name === c.type)!, r = k.width / 2, ux = Math.cos(c.heading), uz = -Math.sin(c.heading);
    for (const f of [-1, 0, 1]) { const d = f * (k.length / 2 - r); circles.push(c.x + ux * d, -c.y + uz * d, r); }
  }
  group.updateMatrixWorld(true); group.traverse(o => { o.matrixAutoUpdate = false; });
  return {group, materials: Object.values(materials), reflective: [materials.paint, materials.glass], circles, count: data.cars.length};
}
