import * as THREE from 'three';
import {classOf} from './finish';

// What the car stands on and runs into besides buildings, read at load from whatever world is loaded. Nothing here
// refers to a place; layers are the pipeline's own join keys.
// - The drivable surface is the top of the ground layers on a 0.25 m raster. The 3 m terrain grid the rest of the
//   viewer reads lies under the road (asphalt stands about 9 cm above it, pavements about 28 cm), so the car sank into
//   the road and a kerb was a ramp three metres long. Road paint is left out (it is drawn a few centimetres proud and a
//   tyre does not feel it); where no ground layer lies, the terrain grid is the surface. The raster is made tile by
//   tile where the car first needs it, from the triangles sorted into tiles at load.
// - What the car cannot enter goes to the building distance field as walls: steps (a stair is not a kerb), water that
//   stands above the ground, and the steep faces of ground layers whose top stands more than 0.3 m over the surface
//   beside them (a planter, a retaining wall, a platform edge), judged every half metre. A kerb or a steep street is
//   driven over.
// - Street objects are whatever the object layers (street furniture, sign posts, tree trunks) hold between 5 cm and
//   1.2 m above the terrain grid, marked on a 0.1 m grid, grouped into blobs and covered by circles.

const SURFACE = new Set(['roads', 'sidewalks', 'paths', 'cycleways', 'transit', 'landscape']);
const STEPS = new Set(['steps']);
const WATER = new Set(['water']);
const OBJECTS = new Set(['street_objects', 'signs', 'vegetation']);
const RISE_M = .3;                          // a steep face rising more than this is a wall
const BAND_LOW = .05, BAND_HIGH = 1.2;      // the bumper band above the terrain grid under an object
const OBJECT_CELL = .1, COVER_M = .25;      // object raster; spacing of the circles that cover a long blob
const TILE = 64;                            // cells per tile side
const BUCKET_M = 2;

type Mesh = THREE.Mesh;

function worldPositions(o: Mesh) {
  o.updateWorldMatrix(true, false);
  const pos = o.geometry.getAttribute('position'), e = o.matrixWorld.elements, out = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    out[i * 3] = e[0] * x + e[4] * y + e[8] * z + e[12]; out[i * 3 + 1] = e[1] * x + e[5] * y + e[9] * z + e[13]; out[i * 3 + 2] = e[2] * x + e[6] * y + e[10] * z + e[14];
  }
  return out;
}
/** Calls `f` with the offsets (into `p`) of each triangle's corners in world space. */
function eachTriangle(o: Mesh, f: (p: Float32Array, a: number, b: number, c: number) => void) {
  const p = worldPositions(o), idx = o.geometry.getIndex(), n = idx ? idx.count : p.length / 3;
  for (let t = 0; t + 2 < n; t += 3) f(p, (idx ? idx.getX(t) : t) * 3, (idx ? idx.getX(t + 1) : t + 1) * 3, (idx ? idx.getX(t + 2) : t + 2) * 3);
}
const normalY = (p: Float32Array, a: number, b: number, c: number) => {
  const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2], vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx; return Math.abs(ny) / (Math.hypot(nx, ny, nz) || 1);
};

/** A square raster over [-extent, extent] in x and z; cell (i, j) is the square whose centre is origin + (i + .5) cell. */
export class Grid {
  readonly n: number; readonly origin: number;
  constructor(extentM: number, readonly cell: number) { this.n = Math.ceil(extentM * 2 / cell) + 1; this.origin = -extentM; }
  index(x: number, z: number) { const i = Math.floor((x - this.origin) / this.cell), j = Math.floor((z - this.origin) / this.cell); return i < 0 || j < 0 || i >= this.n || j >= this.n ? -1 : j * this.n + i; }
  centre(k: number): [number, number] { return [this.origin + (k % this.n + .5) * this.cell, this.origin + (Math.floor(k / this.n) + .5) * this.cell]; }
  /** Every cell in [i0, i1] x [j0, j1] whose centre lies inside the triangle's plan, and the cells along its edges (so a
   *  sliver still counts), with the triangle's height there. */
  triangle(ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number, put: (i: number, j: number, y: number) => void, i0 = 0, i1 = this.n - 1, j0 = 0, j1 = this.n - 1) {
    const det = (bx - ax) * (cz - az) - (cx - ax) * (bz - az), o = this.origin, s = this.cell;
    if (Math.abs(det) > 1e-9) {
      const a0 = Math.max(i0, Math.floor((Math.min(ax, bx, cx) - o) / s - .5)), a1 = Math.min(i1, Math.ceil((Math.max(ax, bx, cx) - o) / s - .5));
      const b0 = Math.max(j0, Math.floor((Math.min(az, bz, cz) - o) / s - .5)), b1 = Math.min(j1, Math.ceil((Math.max(az, bz, cz) - o) / s - .5));
      for (let j = b0; j <= b1; j++) for (let i = a0; i <= a1; i++) {
        const px = o + (i + .5) * s, pz = o + (j + .5) * s;
        const w0 = ((bx - px) * (cz - pz) - (cx - px) * (bz - pz)) / det, w1 = ((cx - px) * (az - pz) - (ax - px) * (cz - pz)) / det, w2 = 1 - w0 - w1;
        if (w0 >= -1e-6 && w1 >= -1e-6 && w2 >= -1e-6) put(i, j, w0 * ay + w1 * by + w2 * cy);
      }
    }
    this.segment(ax, ay, az, bx, by, bz, put, i0, i1, j0, j1); this.segment(bx, by, bz, cx, cy, cz, put, i0, i1, j0, j1); this.segment(cx, cy, cz, ax, ay, az, put, i0, i1, j0, j1);
  }
  segment(ax: number, ay: number, az: number, bx: number, by: number, bz: number, put: (i: number, j: number, y: number) => void, i0 = 0, i1 = this.n - 1, j0 = 0, j1 = this.n - 1) {
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az) / (this.cell * .5)));
    for (let k = 0; k <= steps; k++) {
      const t = k / steps, i = Math.floor((ax + (bx - ax) * t - this.origin) / this.cell), j = Math.floor((az + (bz - az) * t - this.origin) / this.cell);
      if (i >= i0 && i <= i1 && j >= j0 && j <= j1) put(i, j, ay + (by - ay) * t);
    }
  }
}

/** Height of the drivable surface, rasterised a tile at a time on first use. */
export class Surface {
  readonly grid: Grid; private readonly tilesPerRow: number;
  private readonly tiles = new Map<number, Float32Array>();
  // triangles (9 floats each) and, per tile, the triangles that reach it
  private readonly tris: Float32Array; private readonly start: Int32Array; private readonly list: Int32Array;
  constructor(extentM: number, cellM: number, tris: Float32Array, private readonly fallback: (x: number, z: number) => number) {
    this.grid = new Grid(extentM, cellM); this.tris = tris;
    const g = this.grid, per = this.tilesPerRow = Math.ceil(g.n / TILE), count = new Int32Array(per * per + 1);
    const range = (t: number, f: (k: number) => void) => {
      const q = t * 9, span = g.cell * TILE;
      const ti0 = Math.max(0, Math.floor((Math.min(tris[q], tris[q + 3], tris[q + 6]) - g.origin - g.cell) / span)), ti1 = Math.min(per - 1, Math.floor((Math.max(tris[q], tris[q + 3], tris[q + 6]) - g.origin + g.cell) / span));
      const tj0 = Math.max(0, Math.floor((Math.min(tris[q + 2], tris[q + 5], tris[q + 8]) - g.origin - g.cell) / span)), tj1 = Math.min(per - 1, Math.floor((Math.max(tris[q + 2], tris[q + 5], tris[q + 8]) - g.origin + g.cell) / span));
      for (let tj = tj0; tj <= tj1; tj++) for (let ti = ti0; ti <= ti1; ti++) f(tj * per + ti);
    };
    const n = tris.length / 9;
    for (let t = 0; t < n; t++) range(t, k => { count[k + 1]++; });
    for (let k = 0; k < per * per; k++) count[k + 1] += count[k];
    this.start = count.slice(); this.list = new Int32Array(count[per * per]); const fill = count.slice();
    for (let t = 0; t < n; t++) range(t, k => { this.list[fill[k]++] = t; });
  }
  get tilesMade() { return this.tiles.size; }
  private tile(ti: number, tj: number) {
    const key = tj * this.tilesPerRow + ti; let h = this.tiles.get(key);
    if (h) return h;
    h = new Float32Array(TILE * TILE).fill(NaN); const g = this.grid, t = this.tris, i0 = ti * TILE, j0 = tj * TILE, out = h;
    const put = (i: number, j: number, y: number) => { const k = (j - j0) * TILE + i - i0; if (!(out[k] >= y)) out[k] = y; };
    for (let s = this.start[key]; s < this.start[key + 1]; s++) {
      const q = this.list[s] * 9;
      g.triangle(t[q], t[q + 1], t[q + 2], t[q + 3], t[q + 4], t[q + 5], t[q + 6], t[q + 7], t[q + 8], put, i0, i0 + TILE - 1, j0, j0 + TILE - 1);
    }
    this.tiles.set(key, h); return h;
  }
  private value(i: number, j: number) {
    const g = this.grid; if (i < 0 || j < 0 || i >= g.n || j >= g.n) return NaN;
    const v = this.tile(Math.floor(i / TILE), Math.floor(j / TILE))[(j % TILE) * TILE + i % TILE];
    return v === v ? v : this.fallback(g.origin + (i + .5) * g.cell, g.origin + (j + .5) * g.cell);
  }
  /** The surface under (x, z), bilinear between cell centres; outside the raster the terrain grid. */
  at(x: number, z: number) {
    const {n, origin, cell} = this.grid, px = (x - origin) / cell - .5, pz = (z - origin) / cell - .5, i = Math.floor(px), j = Math.floor(pz);
    if (i < 0 || j < 0 || i >= n - 1 || j >= n - 1) return this.fallback(x, z);
    const fx = px - i, fz = pz - j;
    return (this.value(i, j) * (1 - fx) + this.value(i + 1, j) * fx) * (1 - fz) + (this.value(i, j + 1) * (1 - fx) + this.value(i + 1, j + 1) * fx) * fz;
  }
}

/** Street objects as circles, bucketed for lookups around the car. */
export class Obstacles {
  readonly x: Float32Array; readonly z: Float32Array; readonly r: Float32Array; readonly maxR: number;
  private readonly buckets = new Map<number, number[]>();
  constructor(flat: number[]) {
    const n = flat.length / 3; this.x = new Float32Array(n); this.z = new Float32Array(n); this.r = new Float32Array(n); let maxR = 0;
    for (let i = 0; i < n; i++) {
      this.x[i] = flat[i * 3]; this.z[i] = flat[i * 3 + 1]; this.r[i] = flat[i * 3 + 2]; maxR = Math.max(maxR, this.r[i]);
      const key = Obstacles.key(Math.floor(this.x[i] / BUCKET_M), Math.floor(this.z[i] / BUCKET_M)); const b = this.buckets.get(key);
      if (b) b.push(i); else this.buckets.set(key, [i]);
    }
    this.maxR = maxR;
  }
  private static key(i: number, j: number) { return (i + 4096) * 8192 + j + 4096; }
  get count() { return this.x.length; }
  /** The obstacles that can reach within `range` of (x, z), written into `out`. */
  near(x: number, z: number, range: number, out: number[]) {
    out.length = 0; const reach = range + this.maxR;
    const i0 = Math.floor((x - reach) / BUCKET_M), i1 = Math.floor((x + reach) / BUCKET_M), j0 = Math.floor((z - reach) / BUCKET_M), j1 = Math.floor((z + reach) / BUCKET_M);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) { const b = this.buckets.get(Obstacles.key(i, j)); if (b) for (const k of b) out.push(k); }
    return out;
  }
}

export type CarGround = {
  surface: Surface; obstacles: Obstacles;
  /** Hands the car's walls besides buildings (steps, water, high faces) to `mark` as points to fill and edges to draw. */
  blocked: (mark: (x: number, z: number) => void, line: (x0: number, z0: number, x1: number, z1: number) => void) => void;
  stats: {ms: number; fieldMs?: number; surfaceTriangles: number; walls: number; obstacles: number; objectCells: number};
};

export function buildCarGround(objects: THREE.Object3D[], layerOf: (o: THREE.Object3D) => string, groundAt: (x: number, z: number) => number, extentM = 275, cellM = .25): CarGround {
  const t0 = performance.now();
  const meshes = objects.filter((o): o is Mesh => o instanceof THREE.Mesh);
  const grid = new Grid(extentM, cellM);
  // Tops of the ground layers (faces that look up, near the terrain grid: not a canopy, not a pit) for the surface;
  // their steep faces that rise more than RISE_M, steps and water above the ground for the walls.
  const tops: number[] = [], wallPts: number[] = [], wallLines: number[] = [], water: number[] = [], steep: number[] = [];
  for (const o of meshes) {
    const layer = layerOf(o);
    if (SURFACE.has(layer)) eachTriangle(o, (p, a, b, c) => {
      const up = normalY(p, a, b, c);
      if (up > .5) { const x = (p[a] + p[b] + p[c]) / 3, y = (p[a + 1] + p[b + 1] + p[c + 1]) / 3, z = (p[a + 2] + p[b + 2] + p[c + 2]) / 3, g = groundAt(x, z); if (y > g - 1.5 && y < g + 1.2) tops.push(p[a], p[a + 1], p[a + 2], p[b], p[b + 1], p[b + 2], p[c], p[c + 1], p[c + 2]); }
      else if (up < .7 && Math.max(p[a + 1], p[b + 1], p[c + 1]) - Math.min(p[a + 1], p[b + 1], p[c + 1]) > RISE_M) steep.push(p[a], p[a + 1], p[a + 2], p[b], p[b + 1], p[b + 2], p[c], p[c + 1], p[c + 2]);
    });
    else if (STEPS.has(layer)) eachTriangle(o, (p, a, b, c) => { wallLines.push(p[a], p[a + 2], p[b], p[b + 2], p[b], p[b + 2], p[c], p[c + 2], p[c], p[c + 2], p[a], p[a + 2]); grid.triangle(p[a], 0, p[a + 2], p[b], 0, p[b + 2], p[c], 0, p[c + 2], (i, j) => { wallPts.push(grid.origin + (i + .5) * cellM, grid.origin + (j + .5) * cellM); }); });
    else if (WATER.has(layer)) eachTriangle(o, (p, a, b, c) => { if (normalY(p, a, b, c) > .5) water.push(p[a], p[a + 1], p[a + 2], p[b], p[b + 1], p[b + 2], p[c], p[c + 1], p[c + 2]); });
  }
  const surface = new Surface(extentM, cellM, new Float32Array(tops), groundAt);
  // A steep face is a wall where its top stands more than RISE_M over the surface beside it, judged every half metre
  // along it. Its own height says nothing: a pavement's edge face runs down past the road it meets (its skirt), and
  // one face can run the length of a sloping block, so a 0.2 m kerb became a wall wherever its face was 0.3 m tall.
  const STEP = .5;
  for (let q = 0; q < steep.length; q += 9) {
    let ex = 0, ez = 0;
    for (let k = 0; k < 3; k++) { const u = q + k * 3, v = q + ((k + 1) % 3) * 3, dx = steep[v] - steep[u], dz = steep[v + 2] - steep[u + 2]; if (Math.hypot(dx, dz) > Math.hypot(ex, ez)) { ex = dx; ez = dz; } }
    const len = Math.hypot(ex, ez); if (len < 1e-6) continue;
    const tx = ex / len, tz = ez / len, nx = -tz, nz = tx, ox = steep[q], oz = steep[q + 2];
    // the face's top along it: the highest point of its edges in each half-metre bin
    const bins = Math.max(1, Math.ceil(len / STEP)), top = new Float32Array(bins).fill(-Infinity);
    for (let k = 0; k < 3; k++) {
      const u = q + k * 3, v = q + ((k + 1) % 3) * 3, n = Math.max(1, Math.ceil(Math.hypot(steep[v] - steep[u], steep[v + 2] - steep[u + 2]) / (STEP / 2)));
      for (let m = 0; m <= n; m++) {
        const f = m / n, x = steep[u] + (steep[v] - steep[u]) * f, y = steep[u + 1] + (steep[v + 1] - steep[u + 1]) * f, z = steep[u + 2] + (steep[v + 2] - steep[u + 2]) * f;
        const b = Math.min(bins - 1, Math.max(0, Math.floor(((x - ox) * tx + (z - oz) * tz) / STEP))); if (y > top[b]) top[b] = y;
      }
    }
    for (let b = 0; b < bins; b++) {
      const s0 = b * STEP, s1 = Math.min(len, s0 + STEP), sm = (s0 + s1) / 2, x = ox + tx * sm, z = oz + tz * sm;
      const low = Math.min(surface.at(x + nx * .4, z + nz * .4), surface.at(x - nx * .4, z - nz * .4));
      if (top[b] - low > RISE_M) wallLines.push(ox + tx * s0, oz + tz * s0, ox + tx * s1, oz + tz * s1);
    }
  }
  // water whose top stands above the ground there is not driven into
  for (let q = 0; q < water.length; q += 9) grid.triangle(water[q], water[q + 1], water[q + 2], water[q + 3], water[q + 4], water[q + 5], water[q + 6], water[q + 7], water[q + 8], (i, j, y) => {
    const x = grid.origin + (i + .5) * cellM, z = grid.origin + (j + .5) * cellM; if (y > surface.at(x, z) + .05) wallPts.push(x, z);
  });
  // street objects in the bumper band
  const og = new Grid(extentM, OBJECT_CELL), occupied = new Set<number>(), M = og.n;
  for (const o of meshes) {
    const layer = layerOf(o);
    if (!OBJECTS.has(layer) || (layer === 'vegetation' && classOf(o.name).startsWith('leaf'))) continue;   // crowns stand above the car
    eachTriangle(o, (p, a, b, c) => {
      const g = groundAt((p[a] + p[b] + p[c]) / 3, (p[a + 2] + p[b + 2] + p[c + 2]) / 3), lo = g + BAND_LOW, hi = g + BAND_HIGH;
      if (Math.max(p[a + 1], p[b + 1], p[c + 1]) < lo || Math.min(p[a + 1], p[b + 1], p[c + 1]) > hi) return;
      for (let e = 0; e < 3; e++) {
        const u = e === 0 ? a : e === 1 ? b : c, v = e === 0 ? b : e === 1 ? c : a;
        // the part of the edge inside the band
        const y0 = p[u + 1], y1 = p[v + 1]; let t0 = 0, t1 = 1;
        if (Math.abs(y1 - y0) > 1e-9) { const ta = (lo - y0) / (y1 - y0), tb = (hi - y0) / (y1 - y0); t0 = Math.max(0, Math.min(ta, tb)); t1 = Math.min(1, Math.max(ta, tb)); if (t0 > t1) continue; }
        else if (y0 < lo || y0 > hi) continue;
        og.segment(p[u] + (p[v] - p[u]) * t0, 0, p[u + 2] + (p[v + 2] - p[u + 2]) * t0, p[u] + (p[v] - p[u]) * t1, 0, p[u + 2] + (p[v + 2] - p[u + 2]) * t1, (i, j) => { occupied.add(j * M + i); });
      }
    });
  }
  // blobs of touching cells, each covered by circles
  const flat: number[] = [], seen = new Set<number>(), pad = OBJECT_CELL * .75;
  for (const first of occupied) {
    if (seen.has(first)) continue;
    const blob: number[] = [first]; seen.add(first);
    for (let q = 0; q < blob.length; q++) {
      const k = blob[q], i = k % M, j = Math.floor(k / M);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ni = i + di, nj = j + dj; if (ni < 0 || nj < 0 || ni >= M || nj >= M) continue;
        const nk = nj * M + ni; if (occupied.has(nk) && !seen.has(nk)) { seen.add(nk); blob.push(nk); }
      }
    }
    const pts = blob.map(k => og.centre(k)); let mx = 0, mz = 0; for (const [x, z] of pts) { mx += x; mz += z; } mx /= pts.length; mz /= pts.length;
    let far = 0; for (const [x, z] of pts) far = Math.max(far, Math.hypot(x - mx, z - mz));
    if (far <= .45) { flat.push(mx, mz, far + pad); continue; }
    // a long blob (a bench, a fence run): farthest-point cover, so every cell lies within COVER_M of a circle's centre
    const d = pts.map(([x, z]) => Math.hypot(x - pts[0][0], z - pts[0][1])); flat.push(pts[0][0], pts[0][1], COVER_M + pad);
    for (;;) {
      let worst = 0; for (let i = 1; i < d.length; i++) if (d[i] > d[worst]) worst = i;
      if (d[worst] <= COVER_M) break;
      const [cx, cz] = pts[worst]; flat.push(cx, cz, COVER_M + pad);
      for (let i = 0; i < d.length; i++) d[i] = Math.min(d[i], Math.hypot(pts[i][0] - cx, pts[i][1] - cz));
    }
  }
  const obstacles = new Obstacles(flat);
  return {
    surface, obstacles,
    blocked: (mark, line) => { for (let q = 0; q < wallPts.length; q += 2) mark(wallPts[q], wallPts[q + 1]); for (let q = 0; q < wallLines.length; q += 4) line(wallLines[q], wallLines[q + 1], wallLines[q + 2], wallLines[q + 3]); },
    stats: {ms: Math.round(performance.now() - t0), surfaceTriangles: tops.length / 9, walls: wallLines.length / 4 + wallPts.length / 2, obstacles: obstacles.count, objectCells: occupied.size},
  };
}
