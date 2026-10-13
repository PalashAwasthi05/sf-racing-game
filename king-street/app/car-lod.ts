import * as THREE from 'three';

// Lighter stand-ins for the player's car, built at load from whatever roadster model is loaded (so a regenerated model
// never needs a separate LOD file). The full car was drawn in the scene pass, the shadow cascades and the occlusion
// normals: about 100 draws and half a million triangles in each. Two stand-ins take over where the detail is invisible:
// - depth: every rigid group of parts (the body; each wheel's spinning parts; each hub) merged into one simplified mesh
//   per shadow flag, drawn only in the depth-only passes (they read no material, and a shadow map texel or an occlusion
//   sample is coarser than the simplification);
// - far: each part simplified on its own with its own material, drawn in the scene pass while the car is far from the
//   camera (a car parked at the other end of a street, or seen from an orbit view).
// Simplification is vertex clustering on a metric grid: fast (linear) and good to half a cell. Nothing here refers to a
// particular model; parts are found by walking the car's group.

type Source = {geometry: THREE.BufferGeometry; matrix: THREE.Matrix4 | null};

/** Clusters the vertices of the given parts (placed by their matrices) on a grid of `cell` metres and keeps every
 *  triangle whose corners land in three different cells. Normals are the mean of the merged vertices' normals. */
function cluster(sources: Source[], cell: number): THREE.BufferGeometry | null {
  let total = 0;
  for (const s of sources) total += s.geometry.attributes.position.count;
  if (!total) return null;
  const P = new Float32Array(total * 3), N = new Float32Array(total * 3), starts: number[] = [];
  const v = new THREE.Vector3(), normalMatrix = new THREE.Matrix3();
  let o = 0;
  for (const s of sources) {
    starts.push(o);
    const pos = s.geometry.attributes.position, nor = s.geometry.attributes.normal;
    if (s.matrix) normalMatrix.getNormalMatrix(s.matrix);
    for (let i = 0; i < pos.count; i++, o++) {
      v.fromBufferAttribute(pos, i); if (s.matrix) v.applyMatrix4(s.matrix); v.toArray(P, o * 3);
      if (nor) { v.fromBufferAttribute(nor, i); if (s.matrix) v.applyNormalMatrix(normalMatrix); v.toArray(N, o * 3); }
    }
  }
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < total; i++) { x0 = Math.min(x0, P[i * 3]); y0 = Math.min(y0, P[i * 3 + 1]); z0 = Math.min(z0, P[i * 3 + 2]); x1 = Math.max(x1, P[i * 3]); y1 = Math.max(y1, P[i * 3 + 1]); }
  const nx = Math.floor((x1 - x0) / cell) + 2, ny = Math.floor((y1 - y0) / cell) + 2;
  const ids = new Int32Array(total), map = new Map<number, number>();
  const sumP = new Float64Array(total * 3), sumN = new Float64Array(total * 3), count = new Uint32Array(total);
  let clusters = 0;
  for (let i = 0; i < total; i++) {
    const key = Math.floor((P[i * 3] - x0) / cell) + nx * (Math.floor((P[i * 3 + 1] - y0) / cell) + ny * Math.floor((P[i * 3 + 2] - z0) / cell));
    let c = map.get(key);
    if (c === undefined) { c = clusters++; map.set(key, c); }
    ids[i] = c; count[c]++;
    for (let k = 0; k < 3; k++) { sumP[c * 3 + k] += P[i * 3 + k]; sumN[c * 3 + k] += N[i * 3 + k]; }
  }
  let tris = 0;
  for (const s of sources) tris += (s.geometry.index ? s.geometry.index.count : s.geometry.attributes.position.count) / 3;
  const index = new Uint32Array(tris * 3);
  let io = 0;
  sources.forEach((s, si) => {
    const base = starts[si], idx = s.geometry.index, n = idx ? idx.count : s.geometry.attributes.position.count;
    // a mirrored placement (negative determinant) turns the winding inside out; swap two corners to keep faces outward
    const flip = !!s.matrix && s.matrix.determinant() < 0;
    for (let t = 0; t + 2 < n; t += 3) {
      const a = ids[base + (idx ? idx.getX(t) : t)];
      let b = ids[base + (idx ? idx.getX(t + 1) : t + 1)], c = ids[base + (idx ? idx.getX(t + 2) : t + 2)];
      if (a === b || b === c || a === c) continue;
      if (flip) { const x = b; b = c; c = x; }
      index[io++] = a; index[io++] = b; index[io++] = c;
    }
  });
  if (!io) return null;
  const pos = new Float32Array(clusters * 3), nor = new Float32Array(clusters * 3);
  for (let c = 0; c < clusters; c++) {
    for (let k = 0; k < 3; k++) pos[c * 3 + k] = sumP[c * 3 + k] / count[c];
    const l = Math.hypot(sumN[c * 3], sumN[c * 3 + 1], sumN[c * 3 + 2]) || 1;
    nor[c * 3] = sumN[c * 3] / l; nor[c * 3 + 1] = sumN[c * 3 + 1] / l; nor[c * 3 + 2] = sumN[c * 3 + 2] / l;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setIndex(new THREE.BufferAttribute(clusters > 65535 ? index.slice(0, io) : Uint16Array.from(index.subarray(0, io)), 1));
  g.computeBoundingSphere(); g.computeBoundingBox();
  return g;
}

const hasTexture = (m: THREE.Material) => Object.values(m).some(v => v instanceof THREE.Texture);
const trianglesOf = (g: THREE.BufferGeometry) => (g.index ? g.index.count : g.attributes.position.count) / 3;

/** Builds both stand-ins under `group` and returns the per-frame switch and the depth-pass swap. */
export function attachCarLod(group: THREE.Object3D, o: {enabled: () => boolean; depthCellM?: number; farCellM?: number; farFromM?: number}) {
  const depthCell = o.depthCellM ?? .02, farCell = o.farCellM ?? .03, farFrom = o.farFromM ?? 30;
  const meshes: THREE.Mesh[] = [];
  group.traverse(x => { const m = x as THREE.Mesh; if (m.isMesh && !m.userData.carLod) meshes.push(m); });
  for (const m of meshes) m.updateMatrix();

  // depth stand-ins: one per rigid parent, shadow flag and side
  const groups = new Map<string, {parent: THREE.Object3D; cast: boolean; side: THREE.Side; parts: THREE.Mesh[]}>();
  const parentIds = new Map<THREE.Object3D, number>();
  for (const m of meshes) {
    if (Array.isArray(m.material) || !m.parent) continue;
    const pid = parentIds.get(m.parent) ?? parentIds.set(m.parent, parentIds.size).get(m.parent)!;
    const key = `${pid}|${m.castShadow ? 1 : 0}|${m.material.side}`;
    let g = groups.get(key);
    if (!g) { g = {parent: m.parent, cast: m.castShadow, side: m.material.side, parts: []}; groups.set(key, g); }
    g.parts.push(m);
  }
  const depthMaterials = new Map<THREE.Side, THREE.MeshBasicMaterial>();
  const depth: THREE.Mesh[] = [];
  for (const g of groups.values()) {
    const geometry = cluster(g.parts.map(m => ({geometry: m.geometry, matrix: m.matrix})), depthCell);
    if (!geometry) continue;
    let material = depthMaterials.get(g.side);
    // never drawn in colour; its side decides which faces the shadow pass draws, as the parts' own sides did
    if (!material) { material = new THREE.MeshBasicMaterial({side: g.side, colorWrite: false}); material.name = 'car-lod-depth'; depthMaterials.set(g.side, material); }
    const d = new THREE.Mesh(geometry, material);
    d.name = 'car-lod-depth'; d.castShadow = g.cast; d.receiveShadow = false; d.visible = false; d.userData.carLod = true;
    g.parent.add(d); depth.push(d);
  }

  // far stand-ins: each part simplified on its own (shared geometry simplified once), same material and placement
  const simplified = new Map<THREE.BufferGeometry, THREE.BufferGeometry | null>();
  const far: {mesh: THREE.Mesh; lod: THREE.Mesh}[] = [];
  for (const m of meshes) {
    if (Array.isArray(m.material) || hasTexture(m.material) || !m.parent) continue; // no UVs on a stand-in
    let geometry = simplified.get(m.geometry);
    if (geometry === undefined) { geometry = cluster([{geometry: m.geometry, matrix: null}], farCell); simplified.set(m.geometry, geometry); }
    if (!geometry) continue;
    const lod = new THREE.Mesh(geometry, m.material);
    lod.name = m.name + '__far'; lod.position.copy(m.position); lod.quaternion.copy(m.quaternion); lod.scale.copy(m.scale);
    lod.castShadow = m.castShadow; lod.receiveShadow = m.receiveShadow; lod.renderOrder = m.renderOrder; lod.visible = false; lod.userData.carLod = true;
    m.parent.add(lod); far.push({mesh: m, lod});
  }

  let farOn = false, inDepth = false;
  const shown = [...meshes, ...far.map(f => f.lod)], saved = new Uint8Array(shown.length), at = new THREE.Vector3();
  function setFar(on: boolean) {
    if (on === farOn) return; farOn = on;
    for (const {mesh, lod} of far) { if (on) { lod.visible = mesh.visible; mesh.visible = false; } else { mesh.visible = lod.visible; lod.visible = false; } }
  }
  const stats = {
    parts: meshes.length, trianglesFull: meshes.reduce((n, m) => n + trianglesOf(m.geometry), 0),
    depthMeshes: depth.length, trianglesDepth: depth.reduce((n, m) => n + trianglesOf(m.geometry), 0),
    farMeshes: far.length, trianglesFar: far.reduce((n, f) => n + trianglesOf(f.lod.geometry), 0), far: false,
  };
  return {
    stats,
    /** Once a frame before rendering: the far stand-ins while the camera is far from the car (with a little hysteresis). */
    update(camera: THREE.Camera) {
      if (!o.enabled() || !group.visible) { setFar(false); stats.far = false; return; }
      group.getWorldPosition(at);
      const d = camera.position.distanceTo(at);
      setFar(farOn ? d > farFrom - 4 : d > farFrom); stats.far = farOn;
    },
    /** For the depth-only passes: the depth stand-ins in place of everything else of the car. */
    depthSwap: {
      enter() {
        if (!o.enabled() || !group.visible || inDepth) return; inDepth = true;
        shown.forEach((m, i) => { saved[i] = m.visible ? 1 : 0; m.visible = false; }); for (const d of depth) d.visible = true;
      },
      leave() {
        if (!inDepth) return; inDepth = false;
        shown.forEach((m, i) => { m.visible = saved[i] === 1; }); for (const d of depth) d.visible = false;
      },
    },
  };
}
