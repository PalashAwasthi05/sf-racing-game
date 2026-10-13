import * as THREE from 'three';

// Sidewalk slabs. The generator scores sidewalk concrete with transverse joints of their own geometry: thin strips a
// pitch apart along each sidewalk, reaching a few metres in from the kerb. A slab's tone, its cracks and its stains
// change at those joints, never across a slab, so the shader has to know where they are. At load the strips are read
// back (each a connected group of triangles: its centre and its long axis) and drawn from above into a texture holding,
// per pixel, the direction along the nearest strip's run, that strip's phase on it, and the run's line and half length;
// a fragment then finds its slab exactly: t = dot(P, along) - phase, slab k = floor(t / pitch), and how far across the
// run it stands in the run's own straight frame (a kerb-distance contour wanders wherever the kerb does). The nearest
// strip wins each pixel (distance encoded as depth): first across its run (so each side of a street keeps its own),
// then along it (so each slab boundary is placed by its own joint). Where no run reaches the texture is empty and the
// shader falls back to a layout of its own.

export const PAVING_PITCH = 1.8;
const REACH_ACROSS = 10;   // how far past a strip's ends (into a wide sidewalk or a plaza) its run still lays slabs

const VERT = /* glsl */`
attribute vec4 runValue;varying vec4 vValue;
void main(){vValue=runValue;gl_Position=projectionMatrix*viewMatrix*vec4(position,1.);}`;
const FRAG = /* glsl */`precision highp float;varying vec4 vValue;void main(){gl_FragColor=vValue;}`;

type Strip = {cx: number; cz: number; ex: number; ez: number; len: number};

/** The joint strips of the given meshes, in world space. */
function strips(meshes: THREE.Mesh[]) {
  const out: Strip[] = []; const P = new THREE.Vector3();
  for (const mesh of meshes) {
    mesh.updateWorldMatrix(true, false);
    const g = mesh.geometry, pos = g.attributes.position, idx = g.index, n = pos.count;
    const w = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { P.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld); w[3 * i] = P.x; w[3 * i + 1] = P.y; w[3 * i + 2] = P.z; }
    const key = new Map<string, number>(), rep = new Int32Array(n);
    for (let i = 0; i < n; i++) { const k = Math.round(w[3 * i] * 1000) + ',' + Math.round(w[3 * i + 1] * 1000) + ',' + Math.round(w[3 * i + 2] * 1000); let r = key.get(k); if (r === undefined) { r = i; key.set(k, i); } rep[i] = r; }
    const parent = new Int32Array(n); for (let i = 0; i < n; i++) parent[i] = i;
    const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const tris = idx ? idx.count : n;
    for (let t = 0; t < tris; t += 3) { const a = find(rep[idx ? idx.getX(t) : t]), b = find(rep[idx ? idx.getX(t + 1) : t + 1]), c = find(rep[idx ? idx.getX(t + 2) : t + 2]); if (b !== a) parent[b] = a; if (find(c) !== a) parent[find(c)] = a; }
    const comps = new Map<number, number[]>();
    for (let i = 0; i < n; i++) { const r = find(rep[i]); let c = comps.get(r); if (!c) { c = []; comps.set(r, c); } c.push(i); }
    for (const verts of comps.values()) {
      // the strip's two ends: the vertex farthest from any one, then the one farthest from that
      const far = (from: number) => { let best = from, bd = -1; for (const i of verts) { const d = (w[3 * i] - w[3 * from]) ** 2 + (w[3 * i + 2] - w[3 * from + 2]) ** 2; if (d > bd) { bd = d; best = i; } } return best; };
      const e0 = far(verts[0]), e1 = far(e0);
      let ex = w[3 * e1] - w[3 * e0], ez = w[3 * e1 + 2] - w[3 * e0 + 2]; const len = Math.hypot(ex, ez); if (len < .3 || len > 6) continue;
      ex /= len; ez /= len;
      // a joint is a thin strip: every vertex close to the line between its ends
      let thick = 0; for (const i of verts) { const dx = w[3 * i] - w[3 * e0], dz = w[3 * i + 2] - w[3 * e0 + 2]; thick = Math.max(thick, Math.abs(dx * ez - dz * ex)); }
      if (thick > .06) continue;
      out.push({cx: (w[3 * e0] + w[3 * e1]) / 2, cz: (w[3 * e0 + 2] + w[3 * e1 + 2]) / 2, ex, ez, len});
    }
  }
  return out;
}

/** Draws the slab layout of the sidewalks scored by the given joint meshes. */
export function buildPavingField(renderer: THREE.WebGLRenderer, joints: THREE.Mesh[], reach: number, size = 1024) {
  const R = reach, list = strips(joints);
  // runs: strips along one sidewalk share their direction and their line across; each run gets its own random value
  // (for the slabs' tones), found by clustering rather than rounding so no run is split in two
  const along = list.map(s => { let ax = -s.ez, az = s.ex; if (ax < 0 || (ax === 0 && az < 0)) { ax = -ax; az = -az; } return [ax, az]; });
  const order = list.map((_, i) => i).sort((i, j) => Math.atan2(along[i][1], along[i][0]) - Math.atan2(along[j][1], along[j][0]));
  const runOf = new Int32Array(list.length).fill(-1); const runs: {ax: number; az: number; off: number; len: number; line: number; n: number}[] = [];
  for (const i of order) {
    const [ax, az] = along[i], off = list[i].cx * -az + list[i].cz * ax;   // the strip's line, across the run
    let r = runs.findIndex(u => Math.abs(u.ax * az - u.az * ax) < .03 && u.ax * ax + u.az * az > 0 && Math.abs(u.off - off) < .6);
    if (r < 0) { r = runs.length; runs.push({ax, az, off, len: 0, line: 0, n: 0}); }
    runOf[i] = r;
  }
  // each run's line and length from its whole strips (a strip cut short by the sidewalk's edge has its centre moved)
  list.forEach((s, i) => { const u = runs[runOf[i]]; u.len = Math.max(u.len, s.len); });
  list.forEach((s, i) => { const u = runs[runOf[i]]; if (s.len < .95 * u.len) return; u.line += s.cx * -u.az + s.cz * u.ax; u.n++; });
  runs.forEach(u => { u.line /= Math.max(u.n, 1); });
  const position: number[] = [], value: number[] = [], index: number[] = [];
  list.forEach((s, i) => {
    const [ax, az] = along[i], phase = ((s.cx * ax + s.cz * az) % PAVING_PITCH + PAVING_PITCH) % PAVING_PITCH;
    const run = runs[runOf[i]], angle = Math.atan2(az, ax);
    // a grid of 2 x 3 cells around the strip: along it, out to a full pitch either side; across, its own length and
    // REACH_ACROSS beyond each end. The cost (drawn as depth) is 0 on the strip, rising across past its ends (to 1) and
    // a little along (to .05 a pitch away)
    const us = [-PAVING_PITCH, 0, PAVING_PITCH], vs = [-s.len / 2 - REACH_ACROSS, -s.len / 2, s.len / 2, s.len / 2 + REACH_ACROSS];
    const base = position.length / 3;
    for (const v of vs) for (const u of us) {
      const x = s.cx + ax * u + s.ex * v, z = s.cz + az * u + s.ez * v;
      const cost = .05 * Math.abs(u) / PAVING_PITCH + Math.max(0, Math.abs(v) - s.len / 2) / REACH_ACROSS;
      position.push(x, -cost * 100, z); value.push(angle, phase, run.line, run.len / 2);
    }
    for (let r = 0; r < 3; r++) for (let c = 0; c < 2; c++) { const k = base + r * 3 + c; index.push(k, k + 1, k + 3, k + 1, k + 4, k + 3); }
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3)); geometry.setAttribute('runValue', new THREE.Float32BufferAttribute(value, 4)); geometry.setIndex(index);
  const material = new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide, depthTest: true, depthWrite: true});
  const mesh = new THREE.Mesh(geometry, material); mesh.frustumCulled = false;
  const scene = new THREE.Scene(); scene.add(mesh);
  // looking straight down with north (-z) up, as the road field: texture u = (x/R+1)/2, v = (-z/R+1)/2
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, 1, 4000); cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const target = new THREE.WebGLRenderTarget(size, size, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true});
  const prevTarget = renderer.getRenderTarget(), clear = renderer.getClearColor(new THREE.Color()), clearA = renderer.getClearAlpha();
  renderer.setClearColor(0x000000, 0); renderer.setRenderTarget(target); renderer.clear(); renderer.render(scene, cam);
  renderer.setRenderTarget(prevTarget); renderer.setClearColor(clear, clearA);
  geometry.dispose(); material.dispose();
  return {uniforms: {uPfTex: {value: target.texture}, uPfR: {value: R}}, texture: target.texture, strips: list.length, runs: runs.length, dispose() { target.dispose(); }};
}

/** GLSL: the slab layout at a world position. pfAt(P): .x the angle of the direction along the run (atan2(z, x)), .y the
 *  phase of the joints on it (m), .z the run's line (its offset across the run, m), .w half the strips' length (m); all
 *  zero where no run reaches. */
export const PAVING_GLSL = /* glsl */`
uniform sampler2D uPfTex;uniform float uPfR;
vec4 pfAt(vec3 P){vec2 uv=vec2(P.x,-P.z)/(2.*uPfR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return vec4(0.);return textureLod(uPfTex,uv,0.);}
`;
