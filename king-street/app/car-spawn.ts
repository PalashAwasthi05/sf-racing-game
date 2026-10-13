import * as THREE from 'three';

/** Where the car starts: in the right-hand lane of a straight stretch of street near a point, heading along it.
 *  On a street with a painted centre line (the yellow markings) the car stands a lane's half width to the right of that
 *  line, where it runs straight. A street without one is read from the road surface: its kerbs in strips across it, the
 *  car right of their middle. Both are bucketed once, as points every half metre. Nothing here refers to a place. */
const BUCKET_M = 8, LANE_HALF_M = 1.9;

type Points = {x: Float32Array; z: Float32Array; buckets: Map<number, number[]>};
export type RoadSpots = Points & {centre: Points};

const key = (i: number, j: number) => (i + 4096) * 8192 + j + 4096;

function points(objects: THREE.Object3D[], keep: (o: THREE.Mesh) => boolean): Points {
  const xs: number[] = [], zs: number[] = [], seen = new Set<number>();
  for (const o of objects) {
    if (!(o instanceof THREE.Mesh) || !keep(o)) continue;
    o.updateWorldMatrix(true, false);
    const pos = o.geometry.getAttribute('position'), v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
      const k = key(Math.round(v.x * 2), Math.round(v.z * 2)); if (seen.has(k)) continue;   // one point per half metre
      seen.add(k); xs.push(v.x); zs.push(v.z);
    }
  }
  const buckets = new Map<number, number[]>();
  xs.forEach((x, n) => { const k = key(Math.floor(x / BUCKET_M), Math.floor(zs[n] / BUCKET_M)); const b = buckets.get(k); if (b) b.push(n); else buckets.set(k, [n]); });
  return {x: new Float32Array(xs), z: new Float32Array(zs), buckets};
}

/** A yellow marking: the centre line of a two-way street. */
function isCentreLine(o: THREE.Mesh) {
  const m = (Array.isArray(o.material) ? o.material[0] : o.material) as THREE.MeshStandardMaterial;
  if (m?.name === 'yellow') return true;
  const c = m?.color; return !!c && c.r > .5 && c.g > .4 && c.b < .5 * c.r && c.r - c.b > .25;
}

export function roadSpots(objects: THREE.Object3D[], layerOf: (o: THREE.Object3D) => string): RoadSpots {
  return {...points(objects, o => layerOf(o) === 'roads'), centre: points(objects, o => layerOf(o) === 'markings' && isCentreLine(o))};
}

function around(s: Points, x: number, z: number, r: number, out: number[]) {
  out.length = 0;
  const i0 = Math.floor((x - r) / BUCKET_M), i1 = Math.floor((x + r) / BUCKET_M), j0 = Math.floor((z - r) / BUCKET_M), j1 = Math.floor((z + r) / BUCKET_M);
  for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) for (const n of s.buckets.get(key(i, j)) ?? []) if (Math.hypot(s.x[n] - x, s.z[n] - z) <= r) out.push(n);
  return out;
}

/** Whether a road vertex lies within r of (x, z). */
export function onRoad(s: RoadSpots, x: number, z: number, r = 4) { return around(s, x, z, r, []).length > 0; }

/** The principal direction of some points: mean, unit direction, and the spread along and across it. */
function axis(s: Points, near: number[]) {
  let mx = 0, mz = 0; for (const k of near) { mx += s.x[k]; mz += s.z[k]; } mx /= near.length; mz /= near.length;
  let sxx = 0, sxz = 0, szz = 0; for (const k of near) { const dx = s.x[k] - mx, dz = s.z[k] - mz; sxx += dx * dx; sxz += dx * dz; szz += dz * dz; }
  const tr = sxx + szz, det = sxx * szz - sxz * sxz, l1 = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det)), l2 = tr - l1;
  let ux = sxz, uz = l1 - sxx; if (Math.hypot(ux, uz) < 1e-6) { ux = 1; uz = 0; } const ul = Math.hypot(ux, uz);
  return {mx, mz, ux: ux / ul, uz: uz / ul, along: l1 / near.length, across: Math.max(0, l2) / near.length};
}

/** The centre line at a point of it: where the line runs straight for 12 m (no crossing line, no bend), its point
 *  nearest (cx, cz) and its direction. */
function centreLine(s: Points, cx: number, cz: number, near: number[]) {
  around(s, cx, cz, 6, near); if (near.length < 10) return null;
  const a = axis(s, near);
  // a straight line 12 m long spreads ~3.5 m along it; double lines and a dashed gap stay well under 0.25 m across
  if (a.along < 6 || a.across > .06) return null;
  const t = (cx - a.mx) * a.ux + (cz - a.mz) * a.uz;
  return {px: a.mx + a.ux * t, pz: a.mz + a.uz * t, ux: a.ux, uz: a.uz};
}

/** The carriageway around (x, z) read from the road surface, for streets without a centre line: its direction from the
 *  surface within 8 m, its kerbs in 3 m strips across it there and 8 m either way, narrowed to what all three share. */
function surfaceLine(s: Points, cx: number, cz: number, near: number[]) {
  around(s, cx, cz, 8, near); if (near.length < 12) return null;
  const a = axis(s, near); if (a.along < 3 * a.across) return null;   // a junction or a plaza: no single direction
  const nx = -a.uz, nz = a.ux, span = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
  const t = (cx - a.mx) * a.ux + (cz - a.mz) * a.uz, px = a.mx + a.ux * t, pz = a.mz + a.uz * t;
  around(s, px, pz, 12, near);
  for (const k of near) {
    const dx = s.x[k] - px, dz = s.z[k] - pz, al = dx * a.ux + dz * a.uz, l = dx * nx + dz * nz;
    [-8, 0, 8].forEach((a0, i) => { if (Math.abs(al - a0) < 1.5) { span[i][0] = Math.min(span[i][0], l); span[i][1] = Math.max(span[i][1], l); } });
  }
  const lo = Math.max(...span.map(x => x[0])), hi = Math.min(...span.map(x => x[1]));
  if (!(hi - lo >= 5.5)) return null;                                   // narrower than two cars: an alley or a sliver
  const mid = (lo + hi) / 2;
  return {px: px + nx * mid, pz: pz + nz * mid, ux: a.ux, uz: a.uz, width: hi - lo};
}

/** [x, z, yaw] of the nearest good spot to (x, z), or null when the world has no roads. `clear(x, z)` says whether the
 *  car fits there (no parked car or pole); `preferYaw` picks which way along the street it faces. yaw follows car.ts:
 *  forward is (-sin yaw, -cos yaw), and its right is (cos yaw, -sin yaw). */
export function spawnOnRoad(s: RoadSpots, x: number, z: number, clear: (x: number, z: number) => boolean = () => true, preferYaw?: number): [number, number, number] | null {
  const near: number[] = [];
  const face = (ux: number, uz: number) => { let yaw = Math.atan2(-ux, -uz); if (preferYaw !== undefined && Math.cos(yaw - preferYaw) < 0) yaw += Math.PI; return yaw; };
  const nearest = (p: Points, limit: number) => Array.from({length: p.x.length}, (_, i) => i).sort((a, b) => Math.hypot(p.x[a] - x, p.z[a] - z) - Math.hypot(p.x[b] - x, p.z[b] - z)).slice(0, limit);
  const c = s.centre;
  for (const i of nearest(c, 6000)) {
    const line = centreLine(c, c.x[i], c.z[i], near); if (!line) continue;
    const yaw = face(line.ux, line.uz), qx = line.px + Math.cos(yaw) * LANE_HALF_M, qz = line.pz - Math.sin(yaw) * LANE_HALF_M;
    if (!onRoad(s, qx, qz, 1.5) || !clear(qx, qz)) continue;
    return [qx, qz, yaw];
  }
  for (const i of nearest(s, 6000)) {
    const m = surfaceLine(s, s.x[i], s.z[i], near); if (!m) continue;
    const yaw = face(m.ux, m.uz), off = m.width >= 6.5 ? Math.min(LANE_HALF_M, m.width / 4) : 0;
    const qx = m.px + Math.cos(yaw) * off, qz = m.pz - Math.sin(yaw) * off;
    if (!clear(qx, qz)) continue;
    return [qx, qz, yaw];
  }
  return null;
}
