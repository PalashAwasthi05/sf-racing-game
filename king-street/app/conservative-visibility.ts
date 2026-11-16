import * as THREE from 'three';

/** An inner-coverage depth grid. A cell is an occluder only when an actual
 * opaque triangle covers its complete rectangle. Its depth is the farthest
 * point of that triangle; candidate boxes use their nearest point. */
export class OcclusionGrid {
  readonly depth: Float32Array;
  readonly width: number;
  readonly height: number;
  constructor(width = 128, height = 72) { this.width = width; this.height = height; this.depth = new Float32Array(width * height); this.clear(); }
  clear() { this.depth.fill(Infinity); }
  rasterize(points: readonly number[] | Float64Array, guard = .08) {
    const [ax, ay, az, bx, by, bz, cx, cy, cz] = points;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-8 || Math.min(az, bz, cz) <= 0) return;
    const sign = Math.sign(area), far = Math.max(az, bz, cz) + .05;
    const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(this.width - 1, Math.ceil(Math.max(ax, bx, cx)));
    const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), y1 = Math.min(this.height - 1, Math.ceil(Math.max(ay, by, cy)));
    // Expanding each cell by guard contracts the triangle's usable coverage,
    // leaving a numerical guard at silhouettes and shared triangle diagonals.
    const edges = [[ax, ay, bx, by], [bx, by, cx, cy], [cx, cy, ax, ay]];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const cell = y * this.width + x;
      if (this.depth[cell] <= far) continue;
      let covered = true;
      for (const [u, v, p, q] of edges) {
        const dx = p - u, dy = q - v;
        // The linear edge function's minimum over an expanded rectangle.
        const ex = sign * -dy >= 0 ? x - guard : x + 1 + guard;
        const ey = sign * dx >= 0 ? y - guard : y + 1 + guard;
        if (sign * (dx * (ey - v) - dy * (ex - u)) <= 1e-7) { covered = false; break; }
      }
      if (covered) this.depth[cell] = far;
    }
  }
  occluded(x0: number, y0: number, x1: number, y1: number, nearest: number) {
    if (!(nearest > 0) || x1 <= 0 || y1 <= 0 || x0 >= this.width || y0 >= this.height) return false;
    const left = Math.max(0, Math.floor(x0)), right = Math.min(this.width - 1, Math.floor(x1));
    const bottom = Math.max(0, Math.floor(y0)), top = Math.min(this.height - 1, Math.floor(y1));
    for (let y = bottom; y <= top; y++) for (let x = left; x <= right; x++) if (!(this.depth[y * this.width + x] < nearest - .05)) return false;
    return true;
  }
}

type Triangle = {owner: THREE.Mesh; p: Float64Array; side: THREE.Side; flip: number; center: THREE.Vector3};
type Candidate = {mesh: THREE.Mesh; box: THREE.Box3; triangles: number; hidden: boolean};
const opaque = (material: THREE.Material) => {
  const m = material as THREE.MeshPhysicalMaterial;
  return m.visible && m.colorWrite && !m.transparent && m.opacity === 1 && m.depthWrite && m.depthTest && m.depthFunc === THREE.LessEqualDepth &&
    !m.alphaTest && !m.alphaHash && !m.alphaToCoverage && !m.alphaMap && !m.stencilWrite && !m.transmission && !m.displacementMap && !m.clippingPlanes?.length && !m.wireframe && !m.polygonOffset;
};
const shown = (object: THREE.Object3D) => { for (let p: THREE.Object3D | null = object; p; p = p.parent) if (!p.visible) return false; return true; };

/** CPU conservative culling for exactly one beauty camera. Other cameras and
 * shadow callbacks always retain the full geometry. No previous-frame depth
 * survives a camera change. Prepare after temporal jitter, before beauty. */
export function conservativeVisibility(o: {camera: THREE.PerspectiveCamera; meshes: THREE.Mesh[]; layerOf: (mesh: THREE.Object3D) => string; enabled: () => boolean}) {
  const buildStarted = performance.now();
  const {camera} = o, grid = new OcclusionGrid(), candidates: Candidate[] = [], buckets = new Map<string, Triangle[]>();
  const clip = new THREE.Matrix4(), eye = new THREE.Vector3(), p = new THREE.Vector3();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), ab = new THREE.Vector3(), ac = new THREE.Vector3();
  const projected = new Float64Array(9), selected: {triangle: Triangle; distance: number}[] = [];
  const stats = {occluders: 0, samples: 0, selected: 0, rasterized: 0, tested: 0, hidden: 0, triangles: 0, ms: 0, buildMs: 0, ready: false};
  let active = false;
  let misses = 0, cooldown = 0;
  const lastEye = new THREE.Vector3(), lastOrientation = new THREE.Quaternion();
  const orientation = new THREE.Quaternion();
  const hooks: {mesh: THREE.Mesh; before: THREE.Mesh['onBeforeRender']; after: THREE.Mesh['onAfterRender']}[] = [];
  for (const mesh of o.meshes) {
    const g = mesh.geometry;
    if (!g.attributes.position || (mesh as THREE.InstancedMesh).isInstancedMesh || (mesh as THREE.SkinnedMesh).isSkinnedMesh || g.morphAttributes.position) continue;
    mesh.updateWorldMatrix(true, false);
    if (!g.boundingBox) g.computeBoundingBox();
    const candidate: Candidate = {mesh, box: g.boundingBox!.clone().applyMatrix4(mesh.matrixWorld), triangles: (g.index?.count ?? g.attributes.position.count) / 3, hidden: false};
    candidates.push(candidate);
    // Keep the original callbacks and invoke them with their original receiver.
    // eslint-disable-next-line @typescript-eslint/unbound-method
    const before = mesh.onBeforeRender, after = mesh.onAfterRender; let savedStart = 0, savedCount = 0, swapped = false;
    mesh.onBeforeRender = function (...args) {
      before.apply(this, args);
      if (active && candidate.hidden && args[2] === camera && !args[1].overrideMaterial) {
        savedStart = g.drawRange.start; savedCount = g.drawRange.count; g.setDrawRange(0, 0); swapped = true;
      }
    };
    mesh.onAfterRender = function (...args) {
      if (swapped) { g.setDrawRange(savedStart, savedCount); swapped = false; }
      after.apply(this, args);
    };
    hooks.push({mesh, before, after});
    if (g.drawRange.start !== 0 || Number.isFinite(g.drawRange.count) || !['buildings', 'facades'].includes(o.layerOf(mesh)) || Array.isArray(mesh.material) || !opaque(mesh.material)) continue;
    const positions = g.attributes.position, index = g.index, count = index?.count ?? positions.count;
    const at = (i: number) => index ? index.getX(i) : i;
    // Sampling may miss an occluder, which merely keeps extra geometry visible.
    // Bound the startup scan independently of source tessellation density.
    const stride = Math.max(1, Math.ceil(count / 3 / 128));
    let accepted = 0;
    for (let t = 0; t + 2 < count && accepted < 32; t += stride * 3) {
      stats.samples++;
      a.fromBufferAttribute(positions, at(t)).applyMatrix4(mesh.matrixWorld);
      b.fromBufferAttribute(positions, at(t + 1)).applyMatrix4(mesh.matrixWorld);
      c.fromBufferAttribute(positions, at(t + 2)).applyMatrix4(mesh.matrixWorld);
      ab.subVectors(b, a); ac.subVectors(c, a);
      const twiceArea = p.crossVectors(ab, ac).length();
      const longest = Math.max(ab.length(), ac.length(), b.distanceTo(c));
      if (twiceArea < 4 || twiceArea / longest < .75) continue;
      const center = a.clone().add(b).add(c).multiplyScalar(1 / 3);
      const triangle: Triangle = {owner: mesh, p: new Float64Array([...a.toArray(), ...b.toArray(), ...c.toArray()]), side: mesh.material.side, flip: Math.sign(mesh.matrixWorld.determinant()), center};
      const key = Math.floor(center.x / 64) + ',' + Math.floor(center.z / 64), bucket = buckets.get(key);
      if (bucket) bucket.push(triangle); else buckets.set(key, [triangle]);
      stats.occluders++; accepted++;
    }
  }
  stats.buildMs = performance.now() - buildStarted;
  function prepare() {
    const started = performance.now(); active = false; stats.ready = false;
    if (!o.enabled() || buckets.size === 0) { stats.hidden = stats.triangles = stats.ms = 0; return; }
    camera.updateMatrixWorld(); clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // Rebuild for this exact pose and current layer visibility, including jitter.
    eye.setFromMatrixPosition(camera.matrixWorld);
    camera.getWorldQuaternion(orientation);
    // If this view repeatedly saved no draws, give the CPU a break. A skipped
    // frame draws everything: an earlier frame's depth is never reused.
    if (cooldown > 0 && eye.distanceToSquared(lastEye) < 36 && orientation.angleTo(lastOrientation) < .17) {
      cooldown--;
      stats.tested = stats.hidden = stats.triangles = stats.ms = 0;
      return;
    }
    lastEye.copy(eye); lastOrientation.copy(orientation);
    grid.clear(); selected.length = 0;
    const bx = Math.floor(eye.x / 64), bz = Math.floor(eye.z / 64);
    for (let dx = -3; dx <= 3; dx++) for (let dz = -3; dz <= 3; dz++) for (const triangle of buckets.get((bx + dx) + ',' + (bz + dz)) ?? []) {
      const distance = triangle.center.distanceToSquared(eye);
      if (distance < 180 * 180 && shown(triangle.owner) && !Array.isArray(triangle.owner.material) && opaque(triangle.owner.material)) selected.push({triangle, distance});
    }
    selected.sort((a, b) => a.distance - b.distance);
    stats.selected = Math.min(selected.length, 1600); stats.rasterized = stats.tested = stats.hidden = stats.triangles = 0;
    const e = clip.elements, v = camera.matrixWorldInverse.elements;
    for (let i = 0; i < stats.selected; i++) {
      const triangle = selected[i].triangle, points = triangle.p; let forward = true;
      for (let k = 0; k < 3; k++) {
        const x = points[k * 3], y = points[k * 3 + 1], z = points[k * 3 + 2];
        const w = e[3] * x + e[7] * y + e[11] * z + e[15];
        if (w <= camera.near + .05) { forward = false; break; }
        projected[k * 3] = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * .5 + .5) * grid.width;
        projected[k * 3 + 1] = ((e[1] * x + e[5] * y + e[9] * z + e[13]) / w * .5 + .5) * grid.height;
        projected[k * 3 + 2] = w;
      }
      if (!forward) continue;
      const winding = ((projected[3] - projected[0]) * (projected[7] - projected[1]) - (projected[4] - projected[1]) * (projected[6] - projected[0])) * triangle.flip;
      if (triangle.side === THREE.FrontSide && winding <= 0 || triangle.side === THREE.BackSide && winding >= 0) continue;
      grid.rasterize(projected); stats.rasterized++;
    }
    for (const candidate of candidates) {
      candidate.hidden = false;
      if (!shown(candidate.mesh) || candidate.box.containsPoint(eye)) continue;
      const box = candidate.box;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, nearest = Infinity, forward = true;
      for (let corner = 0; corner < 8; corner++) {
        const x = corner & 1 ? box.max.x : box.min.x, y = corner & 2 ? box.max.y : box.min.y, z = corner & 4 ? box.max.z : box.min.z;
        const w = -(v[2] * x + v[6] * y + v[10] * z + v[14]);
        if (w <= camera.near + .05) { forward = false; break; }
        const px = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * .5 + .5) * grid.width;
        const py = ((e[1] * x + e[5] * y + e[9] * z + e[13]) / w * .5 + .5) * grid.height;
        x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py); nearest = Math.min(nearest, w);
      }
      if (!forward) continue;
      stats.tested++;
      candidate.hidden = grid.occluded(x0 - .08, y0 - .08, x1 + .08, y1 + .08, nearest);
      if (candidate.hidden) { stats.hidden++; stats.triangles += candidate.triangles; }
    }
    misses = stats.hidden === 0 ? Math.min(5, misses + 1) : 0;
    cooldown = misses >= 2 ? 2 ** misses - 1 : 0;
    active = true; stats.ready = true; stats.ms = performance.now() - started;
  }
  return {prepare, stats, reset() { active = false; }, dispose() { for (const {mesh, before, after} of hooks) { mesh.onBeforeRender = before; mesh.onAfterRender = after; } }};
}
