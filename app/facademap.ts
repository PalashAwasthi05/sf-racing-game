import * as THREE from 'three';
import type {GlazedWindow} from './glazing';

// The facade map: where the windows are, for the walls around them. A wall weathers around its openings: water off
// every sill carries the dirt it gathers down the wall below in streaks (most from the sill's two ends), and render
// cracks from the corners of an opening. A wall's shader knows nothing of its windows, so at load the windows found in
// the glazing (glazing.ts) are stacked into columns (one window above another in one wall plane, a storey apart: one
// entry however tall the building) and listed by the cells of a grid over the ground, so a fragment of wall finds the
// columns in front of which it stands with a few texel fetches. Nothing here is location specific.

const CELL = 3;   // metres
const COLS = 1024;   // texture width for the column and list tables

export function buildFacadeMap(windows: GlazedWindow[], reach: number) {
  const R = reach, cells = Math.ceil(2 * R / CELL);
  // 1. columns: windows of one wall plane and one width, one above another
  const n = windows.length, parent = new Int32Array(n); for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const grid = new Map<string, number[]>(), centre = (w: GlazedWindow) => [(w.u0 + w.u1) / 2 * w.ax - w.off * w.az, (w.u0 + w.u1) / 2 * w.az + w.off * w.ax];
  windows.forEach((w, i) => { const [x, z] = centre(w), k = Math.floor(x / CELL) + ',' + Math.floor(z / CELL); const c = grid.get(k); if (c) c.push(i); else grid.set(k, [i]); });
  windows.forEach((w, i) => {
    const [x, z] = centre(w), gx = Math.floor(x / CELL), gz = Math.floor(z / CELL);
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) for (const j of grid.get((gx + dx) + ',' + (gz + dz)) ?? []) {
      if (j <= i) continue; const q = windows[j];
      if (w.ax * q.ax + w.az * q.az < .995 || Math.abs(w.off - q.off) > .15 || Math.abs(w.u0 - q.u0) > .15 || Math.abs(w.u1 - q.u1) > .15) continue;
      const a = find(i), b = find(j); if (a !== b) parent[b] = a;
    }
  });
  const groups = new Map<number, number[]>(); for (let i = 0; i < n; i++) { const r = find(i); const g = groups.get(r); if (g) g.push(i); else groups.set(r, [i]); }
  // 2. each column split into runs a regular storey apart: a run is (first sill, pitch, count)
  type Run = {ax: number; az: number; off: number; u0: number; u1: number; y0: number; pitch: number; count: number; h: number};
  const runs: Run[] = [];
  for (const members of groups.values()) {
    const ws = members.map(i => windows[i]).sort((a, b) => a.y0 - b.y0);
    let run: Run | null = null;
    for (const w of ws) {
      const h = Math.min(w.y1 - w.y0, 4.2);
      if (run) {
        const gap = w.y0 - (run.y0 + run.pitch * (run.count - 1));
        if (run.count === 1 && gap > 1.5 && gap < 8) { run.pitch = gap; run.count = 2; run.u0 = Math.min(run.u0, w.u0); run.u1 = Math.max(run.u1, w.u1); continue; }
        if (run.count > 1 && Math.abs(gap - run.pitch) < .2) { run.count++; run.u0 = Math.min(run.u0, w.u0); run.u1 = Math.max(run.u1, w.u1); continue; }
        if (gap < .3) continue;   // a second pane group of one window
      }
      run = {ax: w.ax, az: w.az, off: w.off, u0: w.u0, u1: w.u1, y0: w.y0, pitch: 0, count: 1, h}; runs.push(run);
    }
  }
  // 3. the cells each run lies in: its sill line from a metre past either side, widened a metre either way across
  const lists = new Map<number, number[]>();
  runs.forEach((r, i) => {
    const seen = new Set<number>();
    for (let u = r.u0 - 1; u <= r.u1 + 1.001; u += .5) for (const o of [-1, 0, 1]) {
      const x = u * r.ax - (r.off + o) * r.az, z = u * r.az + (r.off + o) * r.ax;
      const cx = Math.floor((x + R) / CELL), cz = Math.floor((z + R) / CELL); if (cx < 0 || cz < 0 || cx >= cells || cz >= cells) continue;
      const c = cz * cells + cx; if (seen.has(c)) continue; seen.add(c);
      const l = lists.get(c); if (l) l.push(i); else lists.set(c, [i]);
    }
  });
  // 4. the tables: per run two texels (axis, offset, first sill; extent, pitch, count + height / 10), per cell its
  // start and count in the list, the list itself
  const colRows = Math.max(1, Math.ceil(2 * runs.length / COLS)), col = new Float32Array(COLS * colRows * 4);
  runs.forEach((r, i) => { col.set([r.ax, r.az, r.off, r.y0], 8 * i); col.set([r.u0, r.u1, r.pitch, r.count + Math.min(r.h, 9.9) / 10], 8 * i + 4); });
  const cellData = new Float32Array(cells * cells * 4), flat: number[] = [];
  for (const [c, l] of lists) { cellData[4 * c] = flat.length; cellData[4 * c + 1] = l.length; flat.push(...l); }
  const listRows = Math.max(1, Math.ceil(flat.length / COLS)), list = new Float32Array(COLS * listRows); list.set(flat);
  const tex = (data: Float32Array, w: number, h: number, format: THREE.PixelFormat) => { const t = new THREE.DataTexture(data, w, h, format, THREE.FloatType); t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true; return t; };
  const colTex = tex(col, COLS, colRows, THREE.RGBAFormat), cellTex = tex(cellData, cells, cells, THREE.RGBAFormat), listTex = tex(list, COLS, listRows, THREE.RedFormat);
  let longest = 0; for (const l of lists.values()) longest = Math.max(longest, l.length);
  return {
    textures: {uWmCol: colTex, uWmCell: cellTex, uWmList: listTex}, grid: new THREE.Vector4(R, CELL, cells, runs.length ? 1 : 0),
    stats: {windows: n, runs: runs.length, single: runs.filter(r => r.count === 1).length, stacked: runs.filter(r => r.count > 1).reduce((a, r) => a + r.count, 0), cells: lists.size, entries: flat.length, longest},
    dispose() { colTex.dispose(); cellTex.dispose(); listTex.dispose(); },
  };
}
export type FacadeMap = ReturnType<typeof buildFacadeMap>;

/** GLSL: the window columns in front of which a point of wall stands (WebGL2, texelFetch). wmEntries(P) gives the start
 *  and count of its cell's list; wmRun(i) the run's two texels. */
export const FACADEMAP_GLSL = /* glsl */`
uniform sampler2D uWmCol;uniform sampler2D uWmCell;uniform sampler2D uWmList;uniform vec4 uWmGrid;
ivec2 wmEntries(vec3 P){if(uWmGrid.w<.5)return ivec2(0);ivec2 c=ivec2(floor((P.xz+uWmGrid.x)/uWmGrid.y));int N=int(uWmGrid.z);
 if(c.x<0||c.y<0||c.x>=N||c.y>=N)return ivec2(0);vec4 e=texelFetch(uWmCell,c,0);return ivec2(int(e.x),int(e.y));}
int wmListAt(int e){return int(texelFetch(uWmList,ivec2(e%${COLS},e/${COLS}),0).r);}
vec4 wmRunA(int i){int t=2*i;return texelFetch(uWmCol,ivec2(t%${COLS},t/${COLS}),0);}
vec4 wmRunB(int i){int t=2*i+1;return texelFetch(uWmCol,ivec2(t%${COLS},t/${COLS}),0);}
`;
