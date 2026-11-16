import {BufferGeometry, type Object3D, type Material, type Sphere, type Texture, type Vector3} from 'three';
import * as THREE from 'three';

type StageBackpressureGL={
 fenceSync?:(condition:number,flags:number)=>WebGLSync|null;
 clientWaitSync?:(sync:WebGLSync,flags:number,timeout:number)=>number;
 deleteSync?:(sync:WebGLSync|null)=>void;
 isContextLost?:()=>boolean;
 SYNC_GPU_COMMANDS_COMPLETE?:number;TIMEOUT_EXPIRED?:number;WAIT_FAILED?:number;ALREADY_SIGNALED?:number;CONDITION_SATISFIED?:number;
};
export const STAGE_BACKPRESSURE_MAX_HELD_FRAMES=6;

/** Observe completed beauty frames without waiting for the GPU or flushing it.
 * The caller retains its one-stage-per-frame FIFO policy. At most two pending
 * frame markers are held. If they stay late for24 queued beauty frames, permit
 * one stage and restart that bounded starvation window; no marker is dropped
 * to make that escape appear signaled. Unsupported/lost/failed contexts fall
 * back to the caller's ordinary FIFO instead of stranding its promises. */
export function createStageBackpressure(gl:StageBackpressureGL|null|undefined){
 const fences:WebGLSync[]=[];
 let closed=false,failed=false,serial=0,blockedSince:number|undefined;
 const supported=!!gl&&typeof gl.fenceSync==='function'&&typeof gl.clientWaitSync==='function'&&typeof gl.deleteSync==='function'
  &&[gl.SYNC_GPU_COMMANDS_COMPLETE,gl.TIMEOUT_EXPIRED,gl.WAIT_FAILED,gl.ALREADY_SIGNALED,gl.CONDITION_SATISFIED].every(v=>typeof v==='number');
 function release(fence:WebGLSync){try{gl?.deleteSync?.(fence);}catch{/* Lost contexts may reject cleanup. */}}
 function clear(){for(const fence of fences)release(fence);fences.length=0;blockedSince=undefined;}
 function available(){
  if(closed||failed||!supported)return false;
  try{if(gl!.isContextLost?.()){clear();return false;}}catch{clear();failed=true;return false;}
  return true;
 }
 function poll(){
  if(!available())return;
  for(let i=0;i<fences.length;){
   let state:number;
   try{state=gl!.clientWaitSync!(fences[i],0,0);}catch{failed=true;clear();return;}
   if(state===gl!.TIMEOUT_EXPIRED){i++;continue;}
   if(state===gl!.ALREADY_SIGNALED||state===gl!.CONDITION_SATISFIED){release(fences[i]);fences.splice(i,1);continue;}
   failed=true;clear();return;
  }
  // The hold is counted from the first refusal to the next permit. It must not restart here when a marker completes:
  // the same frame adds a new one, so a GPU a couple of frames behind always shows two at the check, the count never
  // reached the bound and staged work was never released (detail tiers stuck on their stand-ins).
 }
 return {
  frame(pending:boolean){
   if(closed)return;serial++;
   if(!pending){clear();return;}
   poll();if(!available())return;
   if(fences.length<2){
    let fence:WebGLSync|null;
    try{fence=gl!.fenceSync!(gl!.SYNC_GPU_COMMANDS_COMPLETE!,0);}catch{failed=true;clear();return;}
    if(!fence){failed=true;clear();return;}fences.push(fence);
   }
   if(fences.length===2&&blockedSince===undefined)blockedSince=serial;
  },
  allow(){
   poll();if(!available())return true;
   if(fences.length<2){blockedSince=undefined;return true;}
   blockedSince??=serial;
   if(serial-blockedSince>=STAGE_BACKPRESSURE_MAX_HELD_FRAMES){blockedSince=serial;return true;}
   return false;
  },
  dispose(){if(closed)return;closed=true;clear();},
 };
}

/** Opt-in CPU submission/wall timings. Fixed storage avoids retaining an
 * unbounded history while a driver explores many streamed cells. */
export function createCellPreparationTrace(capacity=128,now:()=>number=()=>performance.now()){
  type Entry={cell:string;phase:string;startMs:number;durationMs:number};
  const limit=Math.max(1,Math.floor(capacity)),entries=new Array<Entry>(limit);
  let cursor=0,size=0;
  const record=(cell:string,phase:string,startMs:number,durationMs=now()-startMs)=>{
    entries[cursor]={cell,phase,startMs,durationMs};cursor=(cursor+1)%limit;size=Math.min(limit,size+1);
  };
  return {record,begin(cell:string,phase:string){const start=now();let done=false;return()=>{if(!done){done=true;record(cell,phase,start);}};},
    read(){return Array.from({length:size},(_,i)=>({...entries[(cursor-size+i+limit)%limit]}));}};
}

/** Keep the actual buffers, vertex layout and material variants, but submit only
 * one primitive per material while warming the renderer. WebGL uploads the full
 * attributes/index buffers before drawing; invisible startup triangles add no
 * readiness and can otherwise run millions of vertex shader invocations.
 * Restore before producing the opening frame, including when rendering fails. */
export function withWarmupPrimitives<T>(root: Object3D, render: () => T): T {
  const geometries = new Map<BufferGeometry, {start: number; count: number; groups: {start: number; count: number; materialIndex?: number}[]}>();
  const restoreBatches:(()=>void)[]=[];
  root.traverseVisible(object => {
    // Three 0.180 submits BatchedMesh's native multi-draw counts instead of
    // geometry.drawRange. Bound that submission after its normal culling/sync
    // hook, retaining the real buffers and instance textures for upload.
    if((object as THREE.BatchedMesh).isBatchedMesh){
      const mesh=object as THREE.BatchedMesh&{_multiDrawCount:number;_multiDrawCounts:Int32Array};
      const before=mesh.onBeforeRender,after=mesh.onAfterRender;
      const ownBefore=Object.hasOwn(mesh,'onBeforeRender'),ownAfter=Object.hasOwn(mesh,'onAfterRender');
      let saved:{count:number;first:number}|undefined;
      const restore=()=>{if(saved){mesh._multiDrawCount=saved.count;mesh._multiDrawCounts[0]=saved.first;saved=undefined;}};
      mesh.onBeforeRender=function(...args){
        restore();before.apply(this,args);
        saved={count:mesh._multiDrawCount,first:mesh._multiDrawCounts[0]};
        mesh._multiDrawCount=Math.min(1,mesh._multiDrawCount);
        if(mesh._multiDrawCount)mesh._multiDrawCounts[0]=Math.min(3,mesh._multiDrawCounts[0]);
      };
      mesh.onAfterRender=function(...args){restore();after.apply(this,args);};
      restoreBatches.push(()=>{
        restore();
        if(ownBefore)mesh.onBeforeRender=before;else delete (mesh as Partial<THREE.Mesh>).onBeforeRender;
        if(ownAfter)mesh.onAfterRender=after;else delete (mesh as Partial<THREE.Mesh>).onAfterRender;
      });
    }
    const geometry = (object as Object3D & {geometry?: BufferGeometry}).geometry;
    if (!geometry || geometries.has(geometry)) return;
    geometries.set(geometry, {...geometry.drawRange, groups: geometry.groups});
    // Preserve group offsets and material indices: a material starting beyond
    // the first triangle still needs its own draw and texture initialization.
    if (geometry.groups.length) geometry.groups = geometry.groups.map(group => {
      const start = Math.max(group.start, geometry.drawRange.start);
      const end = Math.min(group.start + group.count, geometry.drawRange.start + geometry.drawRange.count);
      return {...group, start, count: Math.max(0, Math.min(end - start, 3))};
    });
    else geometry.setDrawRange(geometry.drawRange.start, Math.min(geometry.drawRange.count, 3));
  });
  try { return render(); }
  finally {
    for(const restore of restoreBatches)restore();
    for (const [geometry, saved] of geometries) {
      geometry.setDrawRange(saved.start, saved.count);
      geometry.groups = saved.groups;
    }
  }
}

/** Compile a representative of every real material/vertex variant, without
 * repeatedly preparing the same material for thousands of static meshes. */
export function warmupCompileObjects(root: Object3D): Object3D[] {
  const seen = new Set<string>(), objects: Object3D[] = [];
  root.traverseVisible(object => {
    const mesh = object as Object3D & {geometry?: BufferGeometry; material?: Material | Material[];
      isInstancedMesh?: boolean; isBatchedMesh?: boolean; isSkinnedMesh?: boolean;
      instanceColor?: unknown; morphTexture?: unknown; _colorsTexture?: unknown};
    if (!mesh.material) return;
    const geometry = mesh.geometry;
    if (!geometry) { objects.push(object); return; }
    const attributes = Object.entries(geometry.attributes).sort(([a], [b]) => a.localeCompare(b))
      .map(([name, attribute]) => [name, attribute.itemSize, attribute.normalized]);
    const morphs = Object.entries(geometry.morphAttributes).sort(([a], [b]) => a.localeCompare(b))
      .map(([name, attributes]) => [name, attributes?.length, attributes?.[0]?.itemSize]);
    const key = JSON.stringify([
      (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).map(material => material.uuid),
      mesh.type, attributes, morphs, geometry.morphTargetsRelative,
      !!mesh.isInstancedMesh, !!mesh.instanceColor, !!mesh.morphTexture,
      !!mesh.isBatchedMesh, !!mesh._colorsTexture, !!mesh.isSkinnedMesh,
    ]);
    if (!seen.has(key)) { seen.add(key); objects.push(object); }
  });
  return objects;
}

/** Distant images stay on the CPU until exploration comes within the preload
 * radius. Shared textures are eligible when any of their meshes is nearby. */
export function nearbyTextureWarmup(candidates: Map<Texture, Sphere[]>, position: Vector3, radius: number): Texture[] {
  return [...candidates].map(([texture, spheres]) => ({texture,
    distance: spheres.reduce((nearest, sphere) => Math.min(nearest, sphere.distanceToPoint(position)), Infinity)}))
    .filter(({distance}) => distance <= radius).sort((a, b) => a.distance - b.distance)
    .map(({texture}) => texture);
}

/** Prepare a cell's program variants against the beauty pass target, then
 * force its geometry buffers onto the GPU in small offscreen batches. The
 * private staging scene references existing objects without reparenting them.
 * Hidden incoming tiers are visible only during the synchronous offscreen
 * submission; their exact visibility is restored before any frame yields. */
export function collectWarmupLights(root:THREE.Object3D){
 const result:THREE.Light[]=[];root.traverse(object=>{if((object as THREE.Light).isLight)result.push(object as THREE.Light);});return result;
}
export function visibleWarmupLights(lights:readonly THREE.Light[]){
 return lights.filter(light=>{for(let object:THREE.Object3D|null=light;object;object=object.parent)if(!object.visible)return false;return true;});
}
// Three keys light/render-list (and any transmission-target) caches by Scene.
// A bounded staging scene prevents every streamed batch acquiring a new key.
// Child references are cleared after each synchronous submission below.
const warmupStages=new WeakMap<THREE.WebGLRenderer,WeakMap<THREE.Scene,{stage:THREE.Scene;compileView:THREE.Group}>>();
function warmupStage(renderer:THREE.WebGLRenderer,liveScene:THREE.Scene){
 let scenes=warmupStages.get(renderer);if(!scenes){scenes=new WeakMap();warmupStages.set(renderer,scenes);}
 let context=scenes.get(liveScene);if(!context){const stage=new THREE.Scene();stage.updateMatrixWorld=()=>{};context={stage,compileView:new THREE.Group()};scenes.set(liveScene,context);}return context;
}
export async function prewarmCell(renderer: THREE.WebGLRenderer, liveScene: THREE.Scene, camera: THREE.Camera,
  target: THREE.WebGLRenderTarget, stagingTarget: THREE.WebGLRenderTarget, meshes: Object3D[], compileObjects: Object3D[], aborted: () => void, yieldFrame: () => Promise<void>,
  onPhase?:(phase:string,startMs:number,durationMs:number)=>void,options:{lights?:readonly THREE.Light[]}={}) {
  const withTarget = <T>(renderTarget: THREE.WebGLRenderTarget, render: () => T): T => {
    const previousTarget = renderer.getRenderTarget(), previousFace = renderer.getActiveCubeFace(), previousMip = renderer.getActiveMipmapLevel();
    const oldAutoClear = renderer.autoClear, oldShadowAuto = renderer.shadowMap.autoUpdate, oldShadowNeedsUpdate = renderer.shadowMap.needsUpdate;
    const cameraLocal = camera.matrix.clone(), cameraWorld = camera.matrixWorld.clone(), cameraInverse = camera.matrixWorldInverse.clone();
    const cameraNeedsUpdate = camera.matrixWorldNeedsUpdate;
    try {
      renderer.setRenderTarget(renderTarget); renderer.autoClear = true;
      renderer.shadowMap.autoUpdate = false; renderer.shadowMap.needsUpdate = false;
      return render();
    }
    finally {
      renderer.autoClear = oldAutoClear; renderer.shadowMap.autoUpdate = oldShadowAuto; renderer.shadowMap.needsUpdate = oldShadowNeedsUpdate;
      camera.matrix.copy(cameraLocal); camera.matrixWorld.copy(cameraWorld); camera.matrixWorldInverse.copy(cameraInverse); camera.matrixWorldNeedsUpdate = cameraNeedsUpdate;
      renderer.setRenderTarget(previousTarget, previousFace, previousMip);
    }
  };

  // compile() runs synchronously inside compileAsync(), so bind the actual
  // beauty target only for that call; its readiness promise must not hold any
  // renderer state across frames.
  const {stage,compileView}=warmupStage(renderer,liveScene);
  stage.background = null;
  stage.environment = liveScene.environment;
  stage.fog = liveScene.fog;
  stage.overrideMaterial = liveScene.overrideMaterial;
  const lights:Object3D[]=options.lights?visibleWarmupLights(options.lights):[];
  if(!options.lights)liveScene.traverseVisible(object => { if ((object as THREE.Light).isLight) lights.push(object); });
  // compileAsync still submits all of its programs synchronously. Only lights,
  // fog and the environment contribute to the compilation target. Walking the
  // whole neighborhood for every handful of section materials wastes a frame.
  for (let offset = 0; offset < compileObjects.length; offset += 8) {
    compileView.children = compileObjects.slice(offset, offset + 8);
    stage.children=lights;
    const started=onPhase?performance.now():0;
    const compiled=(()=>{try{return withTarget(target, () => renderer.compileAsync(compileView, camera, stage));}
      finally{compileView.children=[];stage.children=[];}})();
    onPhase?.('compile-submit',started,performance.now()-started);
    await compiled;
    onPhase?.('compile-ready',started,performance.now()-started);
    aborted();
    await yieldFrame();
    aborted();
  }
  // Renderer traversal reads Scene.children directly. Reuse the actual objects
  // so materials, render callbacks and instancing/batching data remain intact;
  // no source object is reparented, made visible, or has its world matrix updated.
  const queued = meshes.filter(source => !!(source as THREE.Mesh).geometry);
  const byteSize = (object: Object3D) => {
    const mesh = object as THREE.Mesh;
    let bytes = 0;
    for (const attribute of Object.values(mesh.geometry.attributes)) bytes += attribute.array?.byteLength ?? 0;
    if (mesh.geometry.index) bytes += mesh.geometry.index.array.byteLength;
    return Math.max(1, bytes);
  };
  let cursor = 0;
  while (cursor < queued.length) {
    aborted();
    let bytes = 0, end = cursor;
    while (end < queued.length && end - cursor < 8) {
      const nextBytes = byteSize(queued[end]);
      if (end > cursor && bytes + nextBytes > 1024 * 1024) break;
      bytes += nextBytes; end++;
    }
    const slice = queued.slice(cursor, end);
    stage.children = [...lights, ...slice];
    const hooks = slice.map(object => {
      const mesh = object as THREE.Mesh, before = mesh.onBeforeRender, after = mesh.onAfterRender;
      const culled = mesh.frustumCulled, visible=mesh.visible; mesh.frustumCulled = false;mesh.visible=true;
      const ownBefore = Object.hasOwn(mesh, 'onBeforeRender'), ownAfter = Object.hasOwn(mesh, 'onAfterRender');
      mesh.onBeforeRender = function(r, _scene, c, geometry, material, group) { before.call(this, r, liveScene, c, geometry, material, group); };
      mesh.onAfterRender = function(r, _scene, c, geometry, material, group) { after.call(this, r, liveScene, c, geometry, material, group); };
      return {mesh, before, after, ownBefore, ownAfter, culled,visible};
    });
    const started=onPhase?performance.now():0;
    try { withTarget(stagingTarget, () => withWarmupPrimitives(stage, () => renderer.render(stage, camera))); }
    finally {
      stage.children=[];
      // Three's render list retains its last active objects even after the
      // Scene child list is empty. A single staged mesh can retain its entire
      // detached cell through parent pointers. Clear this staging list only;
      // the live scene's lists and all GPU programs/buffers remain reusable.
      const renderList=renderer.renderLists?.get(stage,0);
      renderList?.init();renderList?.finish();
      for (const hook of hooks) {
        hook.mesh.frustumCulled = hook.culled;hook.mesh.visible=hook.visible;
        if (hook.ownBefore) hook.mesh.onBeforeRender = hook.before; else delete (hook.mesh as Partial<THREE.Mesh>).onBeforeRender;
        if (hook.ownAfter) hook.mesh.onAfterRender = hook.after; else delete (hook.mesh as Partial<THREE.Mesh>).onAfterRender;
      }
      onPhase?.('buffer-submit',started,performance.now()-started);
    }
    cursor = end;
    if (cursor < queued.length) await yieldFrame();
  }
  if (!queued.length) return;
  // Submitting buffers to WebGL does not mean the driver has finished them.
  // Keep drawing the existing street until the GPU signals that this section
  // is ready. A zero-timeout poll never waits on the main thread.
  const gl = renderer.getContext();
  if (!('fenceSync' in gl)) return;
  const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (!fence) throw Error('Could not check section upload completion');
  const fenceStarted=onPhase?performance.now():0;
  try {
    gl.flush();
    for (;;) {
      await yieldFrame();
      aborted();
      const status = gl.clientWaitSync(fence, 0, 0);
      if (status === gl.WAIT_FAILED) throw Error('Section upload completion check failed');
      if (status !== gl.TIMEOUT_EXPIRED) break;
    }
  } finally { gl.deleteSync(fence);onPhase?.('gpu-ready-wait',fenceStarted,performance.now()-fenceStarted); }
}
