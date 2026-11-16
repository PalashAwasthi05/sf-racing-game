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
  // Typed chunks avoid two growing boxed-number arrays coexisting with the
  // final float buffers over a whole neighborhood. Coordinates are rounded to
  // Float32 exactly as before; bucket/dedup keys still use the original doubles.
  const chunkSize=16384,xs:Float32Array[]=[],zs:Float32Array[]=[],seen = new Set<number>();
  const buckets = new Map<number, number[]>();
  let count=0;
  for (const o of objects) {
    if (!(o instanceof THREE.Mesh) || !keep(o)) continue;
    o.updateWorldMatrix(true, false);
    const pos = o.geometry.getAttribute('position'), v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
      const k = key(Math.round(v.x * 2), Math.round(v.z * 2)); if (seen.has(k)) continue;   // one point per half metre
      seen.add(k);
      const chunk=Math.floor(count/chunkSize),offset=count%chunkSize;
      if(!offset){xs.push(new Float32Array(chunkSize));zs.push(new Float32Array(chunkSize));}
      xs[chunk][offset]=v.x;zs[chunk][offset]=v.z;
      const bucketKey=key(Math.floor(v.x/BUCKET_M),Math.floor(v.z/BUCKET_M)),bucket=buckets.get(bucketKey);
      if(bucket)bucket.push(count);else buckets.set(bucketKey,[count]);
      count++;
    }
  }
  seen.clear();
  const x=new Float32Array(count),z=new Float32Array(count);
  for(let c=0;c<xs.length;c++){
    const offset=c*chunkSize,length=Math.min(chunkSize,count-offset);
    x.set(xs[c].subarray(0,length),offset);z.set(zs[c].subarray(0,length),offset);
  }
  return {x,z,buckets};
}

/** Exactly the stable distance-sorted prefix, using O(limit) scratch instead of
 * a boxed index and sort workspace for every road point in the neighborhood. */
export function nearestRoadPointIndices(p:Pick<Points,'x'|'z'>,x:number,z:number,limit:number):number[] {
  const capacity=Math.min(p.x.length,Math.max(0,Math.floor(limit)));
  if(!capacity)return [];
  const heap=new Int32Array(capacity),distances=new Float64Array(capacity);
  let size=0;
  const worse=(distance:number,index:number,otherDistance:number,otherIndex:number)=>
    distance>otherDistance||(distance===otherDistance&&index>otherIndex);
  for(let index=0;index<p.x.length;index++){
    const distance=Math.hypot(p.x[index]-x,p.z[index]-z);
    if(size<capacity){
      let at=size++;
      while(at){const parent=(at-1)>>>1;if(!worse(distance,index,distances[parent],heap[parent]))break;
        heap[at]=heap[parent];distances[at]=distances[parent];at=parent;}
      heap[at]=index;distances[at]=distance;
    }else if(worse(distances[0],heap[0],distance,index)){
      let at=0;
      for(;;){let child=at*2+1;if(child>=size)break;
        if(child+1<size&&worse(distances[child+1],heap[child+1],distances[child],heap[child]))child++;
        if(!worse(distances[child],heap[child],distance,index))break;
        heap[at]=heap[child];distances[at]=distances[child];at=child;}
      heap[at]=index;distances[at]=distance;
    }
  }
  return Array.from(heap).sort((a,b)=>Math.hypot(p.x[a]-x,p.z[a]-z)-Math.hypot(p.x[b]-x,p.z[b]-z)||a-b);
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
  const nearest = (p: Points, limit: number) => nearestRoadPointIndices(p,x,z,limit);
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
