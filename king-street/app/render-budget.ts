import * as THREE from 'three';
import {OCTAHEDRAL_NORMAL_GLSL,renderPostQuad} from './postprocessing-utils.ts';
import {staticBoundsSteps, finishStaticSteps, finishStaticStepsAsync, shareStaticAttribute} from './static-detail-geometry.ts';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {FullScreenQuad} from 'three/addons/postprocessing/Pass.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';

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
  /** When set, the next pass reads `target.texture` itself and the copy into the composer's buffer is skipped. */
  direct = false;
  temporal?: {begin:()=>void;restore:()=>void};
  beforeBeauty?: () => void;
  constructor(scene: THREE.Scene, camera: THREE.Camera, samples = 4, resolveDepth=false) {
    super(scene, camera);
    this.target = new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, samples, depthTexture:resolveDepth?new THREE.DepthTexture(1,1,THREE.UnsignedIntType):null, resolveDepthBuffer: resolveDepth, resolveStencilBuffer: false});
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
    this.temporal?.begin();
    try{this.beforeBeauty?.();super.render(renderer, writeBuffer, this.target, deltaTime, maskActive);}
    finally{this.temporal?.restore();}
    if (this.direct) return;
    renderer.setRenderTarget(readBuffer);
    this.copy.render(renderer);
  }
  dispose() { this.target.dispose(); (this.copy.material as THREE.Material).dispose(); this.copy.dispose(); }
}

type Layered = (o: THREE.Object3D) => string;
/** The layer toggles the viewer offers; every other layer is always shown. Signs switch with street objects. */
function toggleGroup(layer: string) { return layer === 'signs' ? 'street_objects' : ['facades', 'vegetation', 'street_objects', 'markings'].includes(layer) ? layer : 'other'; }

/** Merged position+normal views of the static world, one mesh per 120 m cell, toggle group, shadow flag, side and
 *  world transform. Vertices are copied as stored and each copy carries its sources' world matrix, so the GPU puts them
 *  exactly where it puts the sources (baking the transform in would round a depth or two differently and move shadow
 *  edges by a texel). They are drawn only in the depth-only passes (shadow maps and the occlusion normals), which never
 *  read a material: a hundred-odd draws there instead of one per material and 60 m cell. Source attributes share the
 *  merged storage, and singleton groups reuse their source's GPU buffers too. Hidden everywhere else. */
function* buildDepthProxySteps(meshes: THREE.Mesh[], layerOf: Layered, cellM: number, isDetail: (o: THREE.Mesh) => boolean, shadowsOnly: boolean) {
  type Group = {parts: THREE.BufferGeometry[]; verts: number; indices: number; cast: boolean; side: THREE.Side; shadowSide: THREE.Side | null; layer: string; transform: THREE.Matrix4; detail: boolean};
  const groups = new Map<string, Group>(), transforms: THREE.Matrix4[] = [], box = new THREE.Box3(), centre = new THREE.Vector3();
  const transformId = (m: THREE.Matrix4) => { let i = transforms.findIndex(t => t.equals(m)); if (i < 0) i = transforms.push(m.clone()) - 1; return i; };
  const eligible = (m: THREE.Material) => !(m.alphaTest > 0 || m.alphaToCoverage || m.transparent || (m as THREE.MeshStandardMaterial).displacementMap || m.clippingPlanes?.length || (m as THREE.MeshBasicMaterial).wireframe);
  const skipped: THREE.Mesh[] = [];
  for (const o of meshes) {
    yield;
    const g = o.geometry, m = o.material;
    if (shadowsOnly && !o.castShadow) { skipped.push(o); continue; }
    // Spatial index tiles already share the source's position and normal buffers.
    // Draw them directly in depth passes instead of duplicating the entire area's
    // vertex arrays for each small tile.
    if (o.userData.sharedTile || Array.isArray(m) || !eligible(m) || !g.attributes.position || !g.attributes.normal || g.morphAttributes.position || o.customDepthMaterial) { skipped.push(o); continue; }
    o.updateWorldMatrix(true, false);
    box.setFromObject(o); box.getCenter(centre);
    // thin facade detail (frames, mullions, trims) in 40 m cells, drawn only near the camera (installDepthProxies)
    const layer = toggleGroup(layerOf(o)), t = transformId(o.matrixWorld), detail = isDetail(o), cell = detail ? 40 : cellM;
    const key = [Math.floor(centre.x / cell), Math.floor(centre.z / cell), layer, o.castShadow ? 1 : 0, m.side, m.shadowSide ?? '-', t, detail ? 'd' : ''].join('|');
    let grp = groups.get(key);
    if (!grp) { grp = {parts: [], verts: 0, indices: 0, cast: o.castShadow, side: m.side, shadowSide: m.shadowSide, layer, transform: transforms[t], detail}; groups.set(key, grp); }
    grp.parts.push(g);
    grp.verts += g.attributes.position.count; grp.indices += g.index ? g.index.count : g.attributes.position.count;
  }
  const root = new THREE.Group(); root.name = 'depth-proxies'; root.visible = false;
  const materials = new Map<string, THREE.MeshBasicMaterial>(), v = new THREE.Vector3();
  // Draw order by transform: the occlusion pass sorts by material and the shadow pass walks children in order, and
  // three.js re-sends an object's matrices only when they differ from the previous draw's.
  const ordered = [...groups].sort((a, b) => transforms.indexOf(a[1].transform) - transforms.indexOf(b[1].transform));
  for (const [key, grp] of ordered) {
    yield;
    const geometry = new THREE.BufferGeometry();
    if (grp.parts.length === 1) {
      const source = grp.parts[0];
      geometry.setAttribute('position', source.attributes.position);
      if (!shadowsOnly) geometry.setAttribute('normal', source.attributes.normal);
      if(source.attributes.streamTier)geometry.setAttribute('streamTier',source.attributes.streamTier);
      geometry.setIndex(source.index);
      geometry.boundingBox = source.boundingBox?.clone() ?? null;
      geometry.boundingSphere = source.boundingSphere?.clone() ?? null;
    } else {
      const pos = new Float32Array(grp.verts * 3), nrm = shadowsOnly ? null : new Float32Array(grp.verts * 3);
      const index = grp.verts > 65535 ? new Uint32Array(grp.indices) : new Uint16Array(grp.indices);
      const streamed=grp.parts.some(g=>!!g.attributes.streamTier);
      const StreamArray=grp.parts.some(g=>g.attributes.streamTier?.array instanceof Float32Array)?Float32Array:Uint16Array;
      const tiers=streamed?new StreamArray(grp.verts):null;
      let vo = 0, io = 0;
      for (const g of grp.parts) {
        yield;
        const p = g.attributes.position, n = g.attributes.normal, count = p.count;
        const plain = (a: THREE.BufferAttribute | THREE.InterleavedBufferAttribute) => (a as THREE.BufferAttribute).isBufferAttribute && a.array instanceof Float32Array && a.itemSize === 3 && !a.normalized && a.array.length === count * 3;
        if (plain(p)) pos.set(p.array as Float32Array, vo * 3); else for (let i = 0; i < count; i++) { if(i%4096===0)yield;v.fromBufferAttribute(p, i); v.toArray(pos, (vo + i) * 3); }
        if (nrm) {
          if (plain(n)) nrm.set(n.array as Float32Array, vo * 3);
          else for (let i = 0; i < count; i++) { if(i%4096===0)yield;v.fromBufferAttribute(n, i); v.toArray(nrm, (vo + i) * 3); }
        }
        if (g.index) { const src = g.index; for (let i = 0; i < src.count; i++) {if(i%16384===0)yield;index[io + i] = src.getX(i) + vo;} io += src.count; }
        else { for (let i = 0; i < count; i++) {if(i%16384===0)yield;index[io + i] = vo + i;} io += count; }
        if(tiers&&g.attributes.streamTier)for(let i=0;i<count;i++){if(i%16384===0)yield;tiers[vo+i]=g.attributes.streamTier.getX(i);}
        vo += count;
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      if (nrm) geometry.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
      if(tiers)geometry.setAttribute('streamTier',new THREE.BufferAttribute(tiers,1));
      geometry.setIndex(new THREE.BufferAttribute(index, 1));
      let offset = 0;
      for (const source of grp.parts) {
        for (const name of Object.keys(geometry.attributes)) {
          const attribute = source.attributes[name];
          if (attribute instanceof THREE.BufferAttribute) shareStaticAttribute(attribute, geometry.attributes[name] as THREE.BufferAttribute, offset*attribute.itemSize);
        }
        offset += source.attributes.position.count;
      }
    }
    yield* staticBoundsSteps(geometry);
    const mkey = grp.side + '|' + grp.shadowSide + '|' + transforms.indexOf(grp.transform);
    let material = materials.get(mkey);
    if (!material) { material = new THREE.MeshBasicMaterial({side: grp.side, colorWrite: false}); material.shadowSide = grp.shadowSide; material.name = 'depth-proxy'; materials.set(mkey, material); }
    const mesh = new THREE.Mesh(geometry, material); mesh.name = 'depth-proxy|' + key; mesh.castShadow = grp.cast; mesh.receiveShadow = false;
    mesh.userData.layer = grp.layer; mesh.userData.detail = grp.detail; mesh.matrixAutoUpdate = false; mesh.matrix.copy(grp.transform); mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
    if(geometry.attributes.streamTier)mesh.userData.streamShadow=true;
    root.add(mesh);
  }
  root.updateMatrixWorld(true);
  return {root, meshes: root.children as THREE.Mesh[], skipped, dispose() { for (const m of root.children as THREE.Mesh[]) m.geometry.dispose(); materials.forEach(m => m.dispose()); }};
}

/** Startup and ordinary tools retain their synchronous API. */
export function buildDepthProxies(meshes:THREE.Mesh[],layerOf:Layered,cellM=120,isDetail:(o:THREE.Mesh)=>boolean=()=>false,shadowsOnly=false) {
 return finishStaticSteps(buildDepthProxySteps(meshes,layerOf,cellM,isDetail,shadowsOnly));
}
/** Live tile preparation may yield without changing any proxy triangles or attributes. */
export function buildDepthProxiesAsync(meshes:THREE.Mesh[],layerOf:Layered,cellM:number,isDetail:(o:THREE.Mesh)=>boolean,shadowsOnly:boolean,yieldCpu:()=>Promise<void>) {
 return finishStaticStepsAsync(buildDepthProxySteps(meshes,layerOf,cellM,isDetail,shadowsOnly),yieldCpu);
}

export type DepthSwap = {enter(): void; leave(): void};
export function staticDepthSwap(o: {proxies: THREE.Group; statics: THREE.Object3D[]; enabled: () => boolean; near?: () => THREE.Vector3; detailM?: number}): DepthSwap {
  const {proxies,statics}=o;
  const saved: boolean[] = []; let swapped = false;
  // Thin detail stands in only within detailM of the camera: farther out its shadow is under a texel of the near cascade
  // and its occlusion under a pixel of the occlusion pass, and it was a third of their triangles.
  const details = (proxies.children as THREE.Mesh[]).filter(m => m.userData.detail), sphere = new THREE.Sphere();
  const detailVisibility: boolean[] = [];
  return {
    enter() {
      if (!o.enabled() || swapped) return; swapped = true; statics.forEach((s, i) => { saved[i] = s.visible; s.visible = false; }); proxies.visible = true;
      const at = o.near?.(), r = o.detailM ?? 50;
      for (let i = 0; i < details.length; i++) {
        const mesh = details[i];
        detailVisibility[i] = mesh.visible;
        if (!at || !mesh.visible) continue;
        sphere.copy(mesh.geometry.boundingSphere!).applyMatrix4(mesh.matrixWorld);
        mesh.visible = sphere.center.distanceTo(at) - sphere.radius < r;
      }
    },
    leave() { if (!swapped) return; swapped = false; details.forEach((mesh, i) => { mesh.visible = detailVisibility[i]; }); statics.forEach((s, i) => { s.visible = saved[i]; }); proxies.visible = false; },
  };
}

/** Draws the depth-only passes with the proxies in place of the static world it stands for. `statics` are the roots
 *  that hold that world (hidden for the pass), `keep` the static meshes the proxies could not stand for. More stand-ins
 *  (the car's) join through `addSwap`; every swap enters before a depth pass draws and leaves after it. */
export function installDepthProxies(o: {renderer: THREE.WebGLRenderer; scene: THREE.Scene; proxies: THREE.Group; statics: THREE.Object3D[]; enabled: () => boolean; near?: () => THREE.Vector3; detailM?: number}) {
  const {renderer, scene} = o;
  const world=staticDepthSwap(o);
  const swaps: DepthSwap[] = [world];
  const enter = () => { for (const s of swaps) s.enter(); };
  const leave = () => { for (let i = swaps.length - 1; i >= 0; i--) swaps[i].leave(); };
  const shadowMap = renderer.shadowMap, renderShadows = shadowMap.render.bind(shadowMap);
  shadowMap.render = (lights, shadowScene, camera) => {
    // the pass returns early when nothing needs drawing; only swap in the stand-ins when a map will be drawn
    if (!(lights.length > 0 && shadowScene === scene && shadowMap.enabled && (shadowMap.autoUpdate || shadowMap.needsUpdate))) { renderShadows(lights, shadowScene, camera); return; }
    enter();
    try { renderShadows(lights, shadowScene, camera); } finally { leave(); }
  };
  return {addSwap(s: DepthSwap) {
    swaps.push(s);
    return () => {const index=swaps.indexOf(s); if(index>=0)swaps.splice(index,1);};
  }};
}

/** Reconstruct the small AO depth/normal buffer from the resolved beauty depth.
 * Texture taps follow the most continuous neighbour on each axis, preserving
 * silhouettes at kerbs, window reveals and leaf cutouts. */
export function reuseDepthForOcclusion(ao:GTAOPass,depth:THREE.DepthTexture,inverseProjection:THREE.Matrix4,projection?:THREE.Matrix4){
  const target=(ao as GTAOPass & {normalRenderTarget:THREE.WebGLRenderTarget}).normalRenderTarget;
  // The source depth stays full precision. Only the reconstructed unit normal
  // attachment changes: RG8 octahedral normals use 2 bytes instead of RGBA16F's
  // 8, including every normal lookup in GTAO and the 12-tap denoiser.
  target.texture.format=THREE.RGFormat;target.texture.type=THREE.UnsignedByteType;
  target.texture.internalFormat='RG8';target.texture.needsUpdate=true;
  ao.setGBuffer(target.depthTexture!,target.texture);
  for(const pass of [ao.gtaoMaterial,ao.pdMaterial]){
    const signature='vec3 getViewNormal(const vec2 uv) {';
    if(!pass.fragmentShader.includes(signature)||!pass.fragmentShader.includes('#if NORMAL_VECTOR_TYPE == 2'))
      throw new Error('Expected Three GTAO normal sampling entry points');
    pass.fragmentShader=pass.fragmentShader
      .replace(signature,OCTAHEDRAL_NORMAL_GLSL+'\n'+signature)
      .replace('#if NORMAL_VECTOR_TYPE == 2','#if NORMAL_VECTOR_TYPE == 3\n return unpackOctNormal(textureLod(tNormal,uv,0.).rg);\n #elif NORMAL_VECTOR_TYPE == 2');
    pass.defines.NORMAL_VECTOR_TYPE=3;pass.needsUpdate=true;
  }
  // Retain GTAO's normal-buffer debug output as an RGB unit-normal image.
  ao.copyMaterial.uniforms.octNormalOutput={value:0};
  ao.copyMaterial.fragmentShader='uniform float octNormalOutput;\n'+OCTAHEDRAL_NORMAL_GLSL+ao.copyMaterial.fragmentShader.replace(
    'gl_FragColor = opacity * texel;',
    'gl_FragColor = opacity * (octNormalOutput>.5?vec4(unpackOctNormal(texel.rg)*.5+.5,1.):texel);');
  ao.copyMaterial.needsUpdate=true;
  const material=new THREE.ShaderMaterial({
    uniforms:{sourceDepth:{value:depth},inverseProjection:{value:inverseProjection}},
    vertexShader:'varying vec2 screenUv;void main(){screenUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:`
varying vec2 screenUv;
uniform sampler2D sourceDepth;
uniform mat4 inverseProjection;
${OCTAHEDRAL_NORMAL_GLSL}
float depthAt(ivec2 p,ivec2 size){return texelFetch(sourceDepth,clamp(p,ivec2(0),size-1),0).r;}
vec3 pointAt(ivec2 p,ivec2 size,float d){
 vec2 uv=(vec2(p)+.5)/vec2(size);
 vec4 view=inverseProjection*vec4(uv*2.-1.,d*2.-1.,1.);
 return view.xyz/view.w;
}
vec3 tangentAt(ivec2 p,ivec2 axis,ivec2 size,vec3 center,float depth){
 float d0=depthAt(p-axis,size),d1=depthAt(p+axis,size);
 float left=abs(2.*d0-depthAt(p-2*axis,size)-depth);
 float right=abs(2.*d1-depthAt(p+2*axis,size)-depth);
 return left<right?center-pointAt(p-axis,size,d0):pointAt(p+axis,size,d1)-center;
}
void main(){
 ivec2 size=textureSize(sourceDepth,0),p=ivec2(screenUv*vec2(size));
 float depth=depthAt(p,size);gl_FragDepth=depth;
 vec3 n=vec3(0.,0.,1.);
 if(depth<1.){
   vec3 center=pointAt(p,size,depth);
   vec3 crossN=cross(tangentAt(p,ivec2(1,0),size,center,depth),tangentAt(p,ivec2(0,1),size,center,depth));
   if(dot(crossN,crossN)>1e-14)n=normalize(crossN);
 }
 gl_FragColor=vec4(packOctNormal(n),0.,1.);
}`,
    depthTest:true,depthFunc:THREE.AlwaysDepth,depthWrite:true,blending:THREE.NoBlending,toneMapped:false,
  });
  const quad=new FullScreenQuad(material),render=ao.render.bind(ao);
  const camera=(ao as GTAOPass & {camera:THREE.PerspectiveCamera}).camera;
  const savedProjection=new THREE.Matrix4(),savedInverse=new THREE.Matrix4();
  ao.render=(...args:Parameters<GTAOPass['render']>)=>{
    const renderer=args[0];
    ao.copyMaterial.uniforms.octNormalOutput.value=Number(ao.output===GTAOPass.OUTPUT.Normal);
    renderPostQuad(renderer,target,quad);
    if(projection){
      savedProjection.copy(camera.projectionMatrix);savedInverse.copy(camera.projectionMatrixInverse);
      camera.projectionMatrix.copy(projection);camera.projectionMatrixInverse.copy(inverseProjection);
      try{render(...args);}finally{camera.projectionMatrix.copy(savedProjection);camera.projectionMatrixInverse.copy(savedInverse);}
    }else render(...args);
  };
  return ()=>{material.dispose();quad.dispose();};
}

/** The glass's transmission copy: three.js redraws every opaque object into a full-size multisampled target whenever a
 *  transmissive surface is on screen, and samples it only behind that surface (at its own pixel, a mip or two blurred).
 *  So the copy is drawn for the objects that can reach the glass's screen rectangle, padded for the blur, and
 *  scissored to it. With a thickness (refraction offset) or a rough glass the whole copy is drawn as before. */
export function cullTransmission(o: {renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; scenePass: MultisampledRenderPass; candidates: () => THREE.Object3D[]; glassRoot: () => THREE.Object3D | null; enabled: () => boolean}) {
  const LAYER = 7, {renderer, camera, scenePass} = o;
  const frustum = new THREE.Frustum(), m = new THREE.Matrix4(), p = new THREE.Vector3(), box = new THREE.Box3();
  // Driving changes the camera, not the street. Reuse world-space bounds across
  // frames; exact comparisons also retain support for moving objects, replaced
  // geometry and bounds recomputed in place. Weak keys do not retain old tiles.
  const bounds = new WeakMap<THREE.Object3D, {geometry: THREE.BufferGeometry; local: THREE.Sphere; matrix: THREE.Matrix4; world: THREE.Sphere}>();
  function worldSphere(mesh: THREE.Mesh, geometry: THREE.BufferGeometry) {
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    const local = geometry.boundingSphere!;
    let cached = bounds.get(mesh);
    if (!cached) {
      cached = {geometry, local: local.clone(), matrix: mesh.matrixWorld.clone(), world: local.clone().applyMatrix4(mesh.matrixWorld)};
      bounds.set(mesh, cached);
    } else if (cached.geometry !== geometry || !cached.local.equals(local) || !cached.matrix.equals(mesh.matrixWorld)) {
      cached.geometry = geometry; cached.local.copy(local); cached.matrix.copy(mesh.matrixWorld);
      cached.world.copy(local).applyMatrix4(mesh.matrixWorld);
    }
    return cached.world;
  }
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
      if (!mesh.frustumCulled || frustum.intersectsSphere(worldSphere(mesh, g))) { mesh.layers.enable(LAYER); drawn++; } else mesh.layers.disable(LAYER);
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
    const programs = renderer.info.programs as unknown as {isReady:()=>boolean;getUniforms: () => {seq: Entry[]}}[] | null;
    if (!programs || programs.length === seen) return;
    let pending=false;
    for (const p of programs) {
      if(done.has(p))continue;
      // Inspecting uniforms forces a synchronous link if streaming just queued
      // a program. Let the pending compile finish before asking for its layout.
      if(!p.isReady()){pending=true;continue;}
      p.getUniforms().seq.forEach(patch);done.add(p);
    }
    seen=pending?-1:programs.length;
  };
}
