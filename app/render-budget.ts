import * as THREE from 'three';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {FullScreenQuad} from 'three/addons/postprocessing/Pass.js';
import type {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';

// Frame budget. The viewer is bound by draw submission, not by pixels: a street view issued about 2,350 draws a
// frame, of which the image itself needed a quarter. The rest were depth-only passes redrawing the whole static world
// one material at a time (the shadow cascades, the occlusion pass's normals, the car glass's transmission copy),
// and the near cascade was drawn twice, once more inside the occlusion pass that never reads it. Everything here
// keeps the image as it was: the same triangles reach every depth pass, the multisampled scene is resolved exactly
// once, and the glass copies what lies behind the glass. Nothing here refers to a place.

/** Renders the scene into its own multisampled target and copies the resolved image, texel for texel, into the
 *  composer's read buffer. Post passes then run on plain (single-sample, depthless) targets: a full-screen quad
 *  gains nothing from four samples a pixel, and every pass written into one paid for them and for a resolve. */
export class MultisampledRenderPass extends RenderPass {
  readonly target: THREE.WebGLRenderTarget;
  private readonly copy: FullScreenQuad;
  constructor(scene: THREE.Scene, camera: THREE.Camera, samples = 4) {
    super(scene, camera);
    this.target = new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, samples, resolveDepthBuffer: false, resolveStencilBuffer: false});
    this.target.texture.name = 'scene.msaa';
    this.copy = new FullScreenQuad(new THREE.ShaderMaterial({
      uniforms: {tDiffuse: {value: this.target.texture}},
      vertexShader: 'void main(){gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader: 'uniform sampler2D tDiffuse;void main(){gl_FragColor=texelFetch(tDiffuse,ivec2(gl_FragCoord.xy),0);}',
      depthTest: false, depthWrite: false, blending: THREE.NoBlending,
    }));
  }
  setSize(width: number, height: number) { this.target.setSize(width, height); }
  render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget, deltaTime: number, maskActive: boolean) {
    if (this.renderToScreen) { super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive); return; }
    super.render(renderer, writeBuffer, this.target, deltaTime, maskActive); // resolved at the end of renderer.render
    renderer.setRenderTarget(readBuffer);
    this.copy.render(renderer);
  }
  dispose() { this.target.dispose(); (this.copy.material as THREE.Material).dispose(); this.copy.dispose(); }
}

type Layered = (o: THREE.Object3D) => string;
/** The layer toggles the viewer offers; every other layer is always shown. Signs switch with street objects. */
function toggleGroup(layer: string) { return layer === 'signs' ? 'street_objects' : ['facades', 'vegetation', 'street_objects', 'markings'].includes(layer) ? layer : 'other'; }

/** Merged position+normal copies of the static world, one mesh per 120 m cell, toggle group, shadow flag, side and
 *  world transform. Vertices are copied as stored and each copy carries its sources' world matrix, so the GPU puts them
 *  exactly where it puts the sources (baking the transform in would round a depth or two differently and move shadow
 *  edges by a texel). They are drawn only in the depth-only passes (shadow maps and the occlusion normals), which never
 *  read a material: a hundred-odd draws there instead of one per material and 60 m cell. Hidden everywhere else. */
export function buildDepthProxies(meshes: THREE.Mesh[], layerOf: Layered, cellM = 120) {
  type Group = {parts: THREE.BufferGeometry[]; verts: number; indices: number; cast: boolean; side: THREE.Side; shadowSide: THREE.Side | null; layer: string; transform: THREE.Matrix4};
  const groups = new Map<string, Group>(), transforms: THREE.Matrix4[] = [], box = new THREE.Box3(), centre = new THREE.Vector3();
  const transformId = (m: THREE.Matrix4) => { let i = transforms.findIndex(t => t.equals(m)); if (i < 0) i = transforms.push(m.clone()) - 1; return i; };
  const eligible = (m: THREE.Material) => !(m.alphaTest > 0 || m.alphaToCoverage || m.transparent || (m as THREE.MeshStandardMaterial).displacementMap || m.clippingPlanes?.length || (m as THREE.MeshBasicMaterial).wireframe);
  const skipped: THREE.Mesh[] = [];
  for (const o of meshes) {
    const g = o.geometry, m = o.material;
    if (Array.isArray(m) || !eligible(m) || !g.attributes.position || !g.attributes.normal || g.morphAttributes.position || o.customDepthMaterial) { skipped.push(o); continue; }
    o.updateWorldMatrix(true, false);
    box.setFromObject(o); box.getCenter(centre);
    const layer = toggleGroup(layerOf(o)), t = transformId(o.matrixWorld);
    const key = [Math.floor(centre.x / cellM), Math.floor(centre.z / cellM), layer, o.castShadow ? 1 : 0, m.side, m.shadowSide ?? '-', t].join('|');
    let grp = groups.get(key);
    if (!grp) { grp = {parts: [], verts: 0, indices: 0, cast: o.castShadow, side: m.side, shadowSide: m.shadowSide, layer, transform: transforms[t]}; groups.set(key, grp); }
    grp.parts.push(g);
    grp.verts += g.attributes.position.count; grp.indices += g.index ? g.index.count : g.attributes.position.count;
  }
  const root = new THREE.Group(); root.name = 'depth-proxies'; root.visible = false;
  const materials = new Map<string, THREE.MeshBasicMaterial>(), v = new THREE.Vector3();
  // Draw order by transform: the occlusion pass sorts by material and the shadow pass walks children in order, and
  // three.js re-sends an object's matrices only when they differ from the previous draw's.
  const ordered = [...groups].sort((a, b) => transforms.indexOf(a[1].transform) - transforms.indexOf(b[1].transform));
  for (const [key, grp] of ordered) {
    const pos = new Float32Array(grp.verts * 3), nrm = new Float32Array(grp.verts * 3);
    const index = grp.verts > 65535 ? new Uint32Array(grp.indices) : new Uint16Array(grp.indices);
    let vo = 0, io = 0;
    for (const g of grp.parts) {
      const p = g.attributes.position, n = g.attributes.normal, count = p.count;
      const plain = (a: THREE.BufferAttribute | THREE.InterleavedBufferAttribute) => (a as THREE.BufferAttribute).isBufferAttribute && a.array instanceof Float32Array && a.itemSize === 3 && !a.normalized && a.array.length === count * 3;
      if (plain(p)) pos.set(p.array as Float32Array, vo * 3); else for (let i = 0; i < count; i++) { v.fromBufferAttribute(p, i); v.toArray(pos, (vo + i) * 3); }
      if (plain(n)) nrm.set(n.array as Float32Array, vo * 3); else for (let i = 0; i < count; i++) { v.fromBufferAttribute(n, i); v.toArray(nrm, (vo + i) * 3); }
      if (g.index) { const src = g.index; for (let i = 0; i < src.count; i++) index[io + i] = src.getX(i) + vo; io += src.count; }
      else { for (let i = 0; i < count; i++) index[io + i] = vo + i; io += count; }
      vo += count;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geometry.setAttribute('normal', new THREE.BufferAttribute(nrm, 3)); geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.computeBoundingSphere(); geometry.computeBoundingBox();
    const mkey = grp.side + '|' + grp.shadowSide + '|' + transforms.indexOf(grp.transform);
    let material = materials.get(mkey);
    if (!material) { material = new THREE.MeshBasicMaterial({side: grp.side, colorWrite: false}); material.shadowSide = grp.shadowSide; material.name = 'depth-proxy'; materials.set(mkey, material); }
    const mesh = new THREE.Mesh(geometry, material); mesh.name = 'depth-proxy|' + key; mesh.castShadow = grp.cast; mesh.receiveShadow = false;
    mesh.userData.layer = grp.layer; mesh.matrixAutoUpdate = false; mesh.matrix.copy(grp.transform); mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
    root.add(mesh);
  }
  root.updateMatrixWorld(true);
  return {root, meshes: root.children as THREE.Mesh[], skipped, dispose() { for (const m of root.children as THREE.Mesh[]) m.geometry.dispose(); materials.forEach(m => m.dispose()); }};
}

/** Draws the depth-only passes with the proxies in place of the static world it stands for. `statics` are the roots
 *  that hold that world (hidden for the pass), `keep` the static meshes the proxies could not stand for. */
export function installDepthProxies(o: {renderer: THREE.WebGLRenderer; scene: THREE.Scene; ao: GTAOPass; proxies: THREE.Group; statics: THREE.Object3D[]; aoHidden: () => THREE.Object3D[]; enabled: () => boolean}) {
  const {renderer, scene, ao, proxies, statics} = o;
  const swap = (on: boolean, saved: boolean[]) => {
    if (on) { statics.forEach((s, i) => { saved[i] = s.visible; s.visible = false; }); proxies.visible = true; }
    else { statics.forEach((s, i) => { s.visible = saved[i]; }); proxies.visible = false; }
  };
  const shadowMap = renderer.shadowMap, renderShadows = shadowMap.render.bind(shadowMap), savedShadow: boolean[] = [];
  shadowMap.render = (lights, shadowScene, camera) => {
    // the pass returns early when nothing needs drawing; only swap in the proxies when a map will be drawn
    const draws = lights.length > 0 && shadowScene === scene && o.enabled() && (shadowMap.autoUpdate || shadowMap.needsUpdate);
    if (!draws) { renderShadows(lights, shadowScene, camera); return; }
    swap(true, savedShadow);
    try { renderShadows(lights, shadowScene, camera); } finally { swap(false, savedShadow); }
  };
  // The occlusion pass renders the scene once more for its normals and depth. The shadow maps were drawn for this
  // frame by the scene pass and the normal material never reads them, so they are not redrawn here; the static world
  // is drawn as its proxies; and only the scene's few loose lines are hidden instead of walking every node for them.
  const renderAo = ao.render.bind(ao), savedAo: boolean[] = [];
  const aoPass = ao as unknown as {_overrideVisibility: () => void; _visibilityCache: THREE.Object3D[]};
  aoPass._overrideVisibility = () => {
    for (const child of scene.children) {
      if (statics.includes(child) || child === proxies) continue;
      child.traverse(x => { const l = x as THREE.Line & {isLine2?: boolean}; if ((l.isLine || (x as THREE.Points).isPoints || l.isLine2) && x.visible) { x.visible = false; aoPass._visibilityCache.push(x); } });
    }
  };
  ao.render = (...args: Parameters<GTAOPass['render']>) => {
    const autoUpdate = shadowMap.autoUpdate, hidden = o.aoHidden(), was = hidden.map(h => h.visible), proxied = o.enabled();
    shadowMap.autoUpdate = false; hidden.forEach(h => { h.visible = false; }); if (proxied) swap(true, savedAo);
    try { renderAo(...args); } finally { if (proxied) swap(false, savedAo); hidden.forEach((h, i) => { h.visible = was[i]; }); shadowMap.autoUpdate = autoUpdate; }
  };
}

/** The glass's transmission copy: three.js redraws every opaque object into a full-size multisampled target whenever a
 *  transmissive surface is on screen, and samples it only behind that surface (at its own pixel, a mip or two blurred).
 *  So the copy is drawn for the objects that can reach the glass's screen rectangle, padded for the blur, and
 *  scissored to it. With a thickness (refraction offset) or a rough glass the whole copy is drawn as before. */
export function cullTransmission(o: {renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; scenePass: MultisampledRenderPass; candidates: () => THREE.Object3D[]; glassRoot: () => THREE.Object3D | null; enabled: () => boolean}) {
  const LAYER = 7, {renderer, camera, scenePass} = o;
  const frustum = new THREE.Frustum(), m = new THREE.Matrix4(), sphere = new THREE.Sphere(), p = new THREE.Vector3(), box = new THREE.Box3();
  const rect = new THREE.Vector4(); let active = false, inScene = false, entered: THREE.WebGLRenderTarget | null = null, savedMask = 1;
  const stats = {frames: 0, drawn: 0, candidates: 0, full: 0};
  function plane(i: number, a: number[], b: number[], s: number) { frustum.planes[i].setComponents(a[0] - s * b[0], a[1] - s * b[1], a[2] - s * b[2], a[3] - s * b[3]).normalize(); }
  /** Call once per frame before rendering. Marks the objects that can appear behind the glass this frame. */
  function prepare() {
    active = false;
    const root = o.glassRoot(); if (!o.enabled() || !root || !root.visible) return;
    const glass: THREE.Mesh[] = []; let pad = 8, refracts = false;
    root.traverse(x => { const mesh = x as THREE.Mesh; if (!mesh.isMesh || !mesh.visible) return; for (const mat of ([] as THREE.Material[]).concat(mesh.material)) { const t = mat as THREE.MeshPhysicalMaterial; if (t.transmission > 0) { glass.push(mesh); if (t.thickness > 0 || t.dispersion > 0 || t.transmissionMap) refracts = true; const lod = Math.log2(Math.max(1, scenePass.target.width, scenePass.target.height)) * t.roughness * THREE.MathUtils.clamp(t.ior * 2 - 2, 0, 1); pad = Math.max(pad, 2 ** (Math.ceil(lod) + 2)); } } });
    if (!glass.length || refracts) { if (glass.length) stats.full++; return; }
    camera.updateMatrixWorld(); m.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const g of glass) {
      if (!g.geometry.boundingBox) g.geometry.computeBoundingBox();
      box.copy(g.geometry.boundingBox!).applyMatrix4(g.matrixWorld);
      for (let c = 0; c < 8; c++) {
        p.set(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y, c & 4 ? box.max.z : box.min.z).applyMatrix4(camera.matrixWorldInverse);
        if (p.z > -camera.near) { stats.full++; return; } // a corner at or behind the near plane: draw the whole copy
        p.applyMatrix4(camera.projectionMatrix); x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }
    }
    // the copy is the size of the scene target it is drawn for
    const W = scenePass.target.width, H = scenePass.target.height;
    const px0 = Math.max(0, Math.floor((x0 + 1) / 2 * W) - pad), px1 = Math.min(W, Math.ceil((x1 + 1) / 2 * W) + pad);
    const py0 = Math.max(0, Math.floor((y0 + 1) / 2 * H) - pad), py1 = Math.min(H, Math.ceil((y1 + 1) / 2 * H) + pad);
    if (px1 <= px0 || py1 <= py0) return;
    rect.set(px0, py0, px1 - px0, py1 - py0);
    // Frustum through the padded rectangle: clip-space rows of projection x view, bounded at the pixel edges.
    const e = m.elements, row = (r: number) => [e[r], e[4 + r], e[8 + r], e[12 + r]];
    const r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3), nx0 = px0 / W * 2 - 1, nx1 = px1 / W * 2 - 1, ny0 = py0 / H * 2 - 1, ny1 = py1 / H * 2 - 1;
    plane(0, r0, r3, nx0); plane(1, r3.map(v => v * nx1), r0, 1); plane(2, r1, r3, ny0); plane(3, r3.map(v => v * ny1), r1, 1);
    plane(4, r3, r2.map(v => -v), 1); plane(5, r3, r2, 1);
    let drawn = 0, total = 0;
    for (const x of o.candidates()) {
      const mesh = x as THREE.Mesh, g = mesh.geometry as THREE.BufferGeometry | undefined; if (!g) continue; total++;
      if (!g.boundingSphere) g.computeBoundingSphere();
      sphere.copy(g.boundingSphere!).applyMatrix4(mesh.matrixWorld);
      if (!mesh.frustumCulled || frustum.intersectsSphere(sphere)) { mesh.layers.enable(LAYER); drawn++; } else mesh.layers.disable(LAYER);
    }
    root.traverse(x => { x.layers.enable(LAYER); });
    stats.frames++; stats.drawn += drawn; stats.candidates += total; active = true;
  }
  const renderScene = scenePass.render.bind(scenePass);
  scenePass.render = (...args: Parameters<MultisampledRenderPass['render']>) => { inScene = true; try { renderScene(...args); } finally { inScene = false; if (entered) { camera.layers.mask = savedMask; entered = null; } } };
  const setRenderTarget = renderer.setRenderTarget.bind(renderer);
  renderer.setRenderTarget = (target: THREE.WebGLRenderTarget | null, face?: number, mip?: number) => {
    if (entered && target !== entered) { camera.layers.mask = savedMask; entered = null; }
    const transmission = inScene && target !== null && target !== scenePass.target && target.samples > 0 && target.texture.generateMipmaps;
    if (transmission) {
      target.scissorTest = active;
      if (active) { target.scissor.copy(rect); savedMask = camera.layers.mask; camera.layers.mask = 1 << LAYER; entered = target; }
    }
    setRenderTarget(target, face, mip);
  };
  return {prepare, stats};
}

/** three.js caches single-valued uniforms per program but re-sends every array uniform on each material switch. The
 *  shadow cascades' matrices and split distances are arrays that hold one value for a whole frame, so they were sent
 *  once per material (about 160 times a frame) unchanged. This gives those uniforms the same per-program cache the
 *  single-valued ones have: a value is sent only when it differs from what that program already holds. */
export function cacheArrayUniforms(renderer: THREE.WebGLRenderer) {
  type Entry = {id: string; addr: WebGLUniformLocation; size?: number; seq?: Entry[]; setValue: (gl: WebGL2RenderingContext, v: unknown) => void; __cached?: boolean};
  const MATRIX_ARRAYS = new Set(['directionalShadowMatrix', 'spotLightMatrix', 'pointShadowMatrix']), VEC2_ARRAYS = new Set(['CSM_cascades']);
  const done = new WeakSet<object>(); let seen = 0;
  const patch = (e: Entry) => {
    if (e.seq) { e.seq.forEach(patch); return; }
    if (e.__cached || !e.size || e.size < 1) return;
    const n = e.size;
    if (MATRIX_ARRAYS.has(e.id)) {
      const data = new Float32Array(n * 16); let valid = false;
      e.setValue = function (gl, v) {
        const list = v as THREE.Matrix4[]; let same = valid;
        for (let i = 0; i < n; i++) { const m = list[i]?.elements; if (!m) return; for (let k = 0; k < 16; k++) { const x = Math.fround(m[k]); if (data[i * 16 + k] !== x) { same = false; data[i * 16 + k] = x; } } }
        if (same) return; gl.uniformMatrix4fv(this.addr, false, data); valid = true;
      };
    } else if (VEC2_ARRAYS.has(e.id)) {
      const data = new Float32Array(n * 2); let valid = false;
      e.setValue = function (gl, v) {
        const list = v as THREE.Vector2[]; let same = valid;
        for (let i = 0; i < n; i++) { const p = list[i]; if (!p) return; const x = Math.fround(p.x), y = Math.fround(p.y); if (data[i * 2] !== x) { same = false; data[i * 2] = x; } if (data[i * 2 + 1] !== y) { same = false; data[i * 2 + 1] = y; } }
        if (same) return; gl.uniform2fv(this.addr, data); valid = true;
      };
    } else return;
    e.__cached = true;
  };
  /** Call before rendering; patches programs created since the last call (a new program's first frame is unpatched). */
  return () => {
    const programs = renderer.info.programs as unknown as {getUniforms: () => {seq: Entry[]}}[] | null;
    if (!programs || programs.length === seen) return;
    for (const p of programs) { if (done.has(p)) continue; done.add(p); p.getUniforms().seq.forEach(patch); }
    seen = programs.length;
  };
}
