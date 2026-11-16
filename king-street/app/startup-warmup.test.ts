import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {BufferGeometry, Float32BufferAttribute, Group, InstancedMesh, Mesh, MeshBasicMaterial, Sphere, Texture, Vector3} from 'three';
import {withWarmupPrimitives, warmupCompileObjects, nearbyTextureWarmup,collectWarmupLights,visibleWarmupLights,createStageBackpressure,STAGE_BACKPRESSURE_MAX_HELD_FRAMES} from './startup-warmup.ts';
import {WebGLRenderLists} from 'three/src/renderers/webgl/WebGLRenderLists.js';

function fenceFixture(){
 const created:WebGLSync[]=[],deleted:WebGLSync[]=[],waits:{flags:number;timeout:number}[]=[],states=new Map<WebGLSync,number>();let lost=false,nullFence=false,throwWait=false,flushes=0;
 const gl={SYNC_GPU_COMMANDS_COMPLETE:37143,ALREADY_SIGNALED:37146,TIMEOUT_EXPIRED:37147,CONDITION_SATISFIED:37148,WAIT_FAILED:37149,
  isContextLost:()=>lost,
  fenceSync:(condition:number,flags:number)=>{assert.equal(condition,37143);assert.equal(flags,0);if(nullFence)return null;const fence={} as WebGLSync;created.push(fence);return fence;},
  clientWaitSync:(sync:WebGLSync,flags:number,timeout:number)=>{waits.push({flags,timeout});if(throwWait)throw Error('Lost GPU');return states.get(sync)??37147;},
  deleteSync:(sync:WebGLSync|null)=>{if(sync)deleted.push(sync);},flush:()=>{flushes++;throw Error('Backpressure must never flush');}};
 return {gl,created,deleted,waits,states,get flushes(){return flushes;},set lost(value:boolean){lost=value;},set nullFence(value:boolean){nullFence=value;},set throwWait(value:boolean){throwWait=value;}};
}

test('stage backpressure polls real fence states without waiting/flushing and holds at two pending frames',()=>{
 const fixture=fenceFixture(),gate=createStageBackpressure(fixture.gl);
 gate.frame(false);assert.equal(fixture.created.length,0);assert.equal(gate.allow(),true);
 gate.frame(true);assert.equal(gate.allow(),true);gate.frame(true);assert.equal(gate.allow(),false);gate.frame(true);assert.equal(fixture.created.length,2);
 fixture.states.set(fixture.created[0],fixture.gl.ALREADY_SIGNALED);assert.equal(gate.allow(),true);assert.deepEqual(fixture.deleted,[fixture.created[0]]);
 gate.frame(true);assert.equal(fixture.created.length,3);assert.equal(gate.allow(),false);
 fixture.states.set(fixture.created[1],fixture.gl.CONDITION_SATISFIED);assert.equal(gate.allow(),true);
 gate.dispose();gate.dispose();assert.equal(fixture.deleted.length,3);assert.equal(new Set(fixture.deleted).size,3);assert.equal(fixture.flushes,0);assert.ok(fixture.waits.every(w=>w.flags===0&&w.timeout===0));
 const waits=fixture.waits.length;assert.equal(gate.allow(),true);gate.frame(true);assert.equal(fixture.waits.length,waits);assert.equal(fixture.created.length,3);
});

test('stage backpressure gives one bounded progress permit on very slow GPUs while retaining both markers',()=>{
 const fixture=fenceFixture(),gate=createStageBackpressure(fixture.gl);gate.frame(true);gate.frame(true);assert.equal(gate.allow(),false);
 for(let frame=1;frame<STAGE_BACKPRESSURE_MAX_HELD_FRAMES;frame++){gate.frame(true);assert.equal(gate.allow(),false);}
 gate.frame(true);assert.equal(gate.allow(),true);assert.equal(gate.allow(),false,'one escape must not drain the FIFO in a single frame');
 assert.equal(fixture.created.length,2);assert.equal(fixture.deleted.length,0,'late markers remain unsignaled and accounted for');
 for(let frame=1;frame<=STAGE_BACKPRESSURE_MAX_HELD_FRAMES;frame++){gate.frame(true);assert.equal(gate.allow(),frame===STAGE_BACKPRESSURE_MAX_HELD_FRAMES);}
 gate.frame(false);assert.equal(gate.allow(),true);assert.equal(fixture.deleted.length,2);gate.frame(true);assert.equal(fixture.created.length,3);gate.dispose();
});

test('stage backpressure still gives its bounded permit when markers complete but two are always outstanding at the check',()=>{
 // A GPU running about two frames behind: every frame the oldest marker completes after the gate was asked and before the
 // frame ends, and the frame adds a new one, so the gate always sees two. Observed in the viewer on 2026-11-12: frames
 // rendered, markers were created and completed steadily (593 made, 590 done), and no staged work was released for 15 s.
 const fixture=fenceFixture(),gate=createStageBackpressure(fixture.gl);gate.frame(true);gate.frame(true);
 let granted=0,done=0;
 for(let frame=0;frame<3*STAGE_BACKPRESSURE_MAX_HELD_FRAMES;frame++){
  if(gate.allow())granted++;
  fixture.states.set(fixture.created[done++],fixture.gl.CONDITION_SATISFIED);
  gate.frame(true);
 }
 assert.ok(granted>=2,`granted ${granted} permits in ${3*STAGE_BACKPRESSURE_MAX_HELD_FRAMES} frames`);gate.dispose();
});

test('stage backpressure recovers from lost contexts and fails open on unsupported/null/failed sync APIs',()=>{
 for(const gl of [null,undefined,{}, {fenceSync:()=>null}]){const gate=createStageBackpressure(gl);assert.equal(gate.allow(),true);gate.frame(true);gate.dispose();}
 const fixture=fenceFixture(),gate=createStageBackpressure(fixture.gl);gate.frame(true);gate.frame(true);fixture.lost=true;assert.equal(gate.allow(),true);assert.equal(fixture.deleted.length,2);gate.frame(true);assert.equal(fixture.created.length,2);
 fixture.lost=false;gate.frame(true);assert.equal(fixture.created.length,3);gate.dispose();assert.equal(fixture.deleted.length,3);
 for(const mode of ['null','failed','throw','unknown'] as const){
  const sample=fenceFixture(),g=createStageBackpressure(sample.gl);g.frame(true);
  if(mode==='null'){sample.nullFence=true;g.frame(true);}else if(mode==='throw'){sample.throwWait=true;}else sample.states.set(sample.created[0],mode==='failed'?sample.gl.WAIT_FAILED:-123);
  assert.equal(g.allow(),true);const created=sample.created.length;g.frame(true);assert.equal(sample.created.length,created);g.dispose();assert.equal(sample.deleted.length,1);assert.equal(new Set(sample.deleted).size,1);
 }
});

test('warmup retains actual upload buffers and every material group, then restores exact ranges', () => {
  const geometry = new BufferGeometry();
  const position = new Float32BufferAttribute(new Float32Array(300), 3);
  geometry.setAttribute('position', position);
  geometry.setIndex(Array.from({length: 90}, (_, i) => i));
  const index = geometry.index;
  geometry.setDrawRange(3, 81);
  geometry.addGroup(0, 30, 0); geometry.addGroup(30, 60, 1);
  const groups = geometry.groups;
  const root = new Group();
  root.add(new Mesh(geometry, [new MeshBasicMaterial(), new MeshBasicMaterial()]));
  root.add(new Mesh(geometry, new MeshBasicMaterial())); // Shared geometry saved once.
  assert.equal(withWarmupPrimitives(root, () => {
    assert.equal(geometry.attributes.position, position); assert.equal(geometry.index, index);
    assert.deepEqual(geometry.groups, [{start:3,count:3,materialIndex:0},{start:30,count:3,materialIndex:1}]);
    return 7;
  }), 7);
  assert.equal(geometry.groups, groups);
  assert.deepEqual(geometry.drawRange, {start:3,count:81});
});

test('warmup restores single-material ranges on failure and leaves hidden geometry alone', () => {
  const geometry = new BufferGeometry(), hidden = new BufferGeometry();
  geometry.setDrawRange(6, 21);
  const root = new Group(); root.add(new Mesh(geometry, new MeshBasicMaterial()));
  const mesh = new Mesh(hidden, new MeshBasicMaterial()); mesh.visible = false; root.add(mesh);
  assert.throws(() => withWarmupPrimitives(root, () => {
    assert.deepEqual(geometry.drawRange, {start:6,count:3});
    assert.deepEqual(hidden.drawRange, {start:0,count:Infinity});
    throw new Error('GPU failure');
  }), /GPU failure/);
  assert.deepEqual(geometry.drawRange, {start:6,count:21});
});

test('an incoming tier hidden by zero transition coverage still submits its buffers offscreen',async()=>{
 const {prewarmCell}=await import('./startup-warmup.ts');
 const source=new Mesh(new BufferGeometry().setAttribute('position',new Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3)),new MeshBasicMaterial());source.visible=false;
 const live=new THREE.Scene(),camera=new THREE.PerspectiveCamera(),beauty=new THREE.WebGLRenderTarget(2,2),staging=new THREE.WebGLRenderTarget(1,1);
 let target:THREE.WebGLRenderTarget|null=null,submitted=0;
 const renderer={autoClear:false,shadowMap:{autoUpdate:true,needsUpdate:true},getRenderTarget:()=>target,getActiveCubeFace:()=>0,getActiveMipmapLevel:()=>0,
  setRenderTarget:(value:THREE.WebGLRenderTarget|null)=>{target=value;},compileAsync:async()=>{},getContext:()=>({}),
  render:(stage:THREE.Scene)=>{assert.equal(target,staging);assert.ok(stage.children.includes(source));assert.equal(source.visible,true);assert.equal(source.frustumCulled,false);assert.equal(source.geometry.drawRange.count,3);submitted++;}} as unknown as THREE.WebGLRenderer;
 await prewarmCell(renderer,live,camera,beauty,staging,[source],[source],()=>{},async()=>{assert.equal(source.visible,false);assert.equal(source.frustumCulled,true);assert.equal(target,null);});
 assert.equal(submitted,1);assert.equal(source.visible,false);assert.equal(source.frustumCulled,true);assert.equal(source.geometry.drawRange.count,Infinity);
});

test('pre-indexed streaming lights avoid walking world meshes and follow hidden ancestors',async()=>{
 const {prewarmCell}=await import('./startup-warmup.ts');const live=new THREE.Scene(),branch=new THREE.Group(),visible=new THREE.DirectionalLight(),hidden=new THREE.PointLight();
 live.add(visible,branch);branch.add(hidden);branch.visible=false;
 const lights=collectWarmupLights(live);assert.equal(lights.length,2);assert.deepEqual(visibleWarmupLights(lights),[visible]);branch.visible=true;assert.deepEqual(visibleWarmupLights(lights),[visible,hidden]);branch.visible=false;
 live.traverseVisible=()=>{throw Error('Streaming must not walk the full neighborhood for lights');};
 const source=new Mesh(geometryForLightTest(),new MeshBasicMaterial()),camera=new THREE.PerspectiveCamera(),beauty=new THREE.WebGLRenderTarget(2,2),staging=new THREE.WebGLRenderTarget(1,1);
 let target:THREE.WebGLRenderTarget|null=null;const stages:THREE.Scene[]=[],rows:{object:unknown;geometry:unknown;material:unknown}[]=[];
 const renderLists=new WebGLRenderLists(undefined as never);
 const renderer={autoClear:false,shadowMap:{autoUpdate:true,needsUpdate:true},getRenderTarget:()=>target,getActiveCubeFace:()=>0,getActiveMipmapLevel:()=>0,
  setRenderTarget:(value:THREE.WebGLRenderTarget|null)=>{target=value;},compileAsync:async(_view:THREE.Object3D,_camera:THREE.Camera,stage:THREE.Scene)=>{assert.deepEqual(stage.children,[visible]);stages.push(stage);},getContext:()=>({}),renderLists,
  render:(stage:THREE.Scene)=>{const list=renderLists.get(stage,0);list.init();list.push(source,source.geometry,source.material,0,0,null);list.finish();rows.push(list.opaque[0]);}} as unknown as THREE.WebGLRenderer;
 await prewarmCell(renderer,live,camera,beauty,staging,[source],[source],()=>{},async()=>{},undefined,{lights});
 await prewarmCell(renderer,live,camera,beauty,staging,[source],[source],()=>{},async()=>{},undefined,{lights});
 assert.equal(stages[0],stages[1],'staging must reuse Three cache keys');assert.equal(stages[0].children.length,0,'idle staging cache must not retain streamed meshes or their parents');
 assert.ok(rows.every(row=>row.object===null&&row.geometry===null&&row.material===null),'Three internal render-item references must be cleared, not only Scene.children');
 assert.equal(renderLists.get(stages[0],0).opaque.length,0);
});

function geometryForLightTest(){return new BufferGeometry().setAttribute('position',new Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3));}

test('compilation deduplicates matching meshes but retains vertex and material variants', () => {
  const root = new Group(), material = new MeshBasicMaterial();
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute([0, 0, 0], 3));
  const first = new Mesh(geometry, material), duplicate = new Mesh(geometry.clone(), material);
  const colored = new Mesh(geometry.clone(), material);
  colored.geometry.setAttribute('color', new Float32BufferAttribute([1, 1, 1, 1], 4));
  const otherMaterial = new Mesh(geometry, material.clone());
  const instanced = new InstancedMesh(geometry, material, 1);
  const hidden = new Mesh(geometry, material.clone());hidden.visible = false;
  root.add(first, duplicate, colored, otherMaterial, instanced, hidden);
  assert.deepEqual(warmupCompileObjects(root), [first, colored, otherMaterial, instanced]);
  assert.deepEqual(root.children, [first, duplicate, colored, otherMaterial, instanced, hidden]);
});

test('texture preloading selects nearby bounds including shared textures, in distance order', () => {
  const near = new Texture(), shared = new Texture(), far = new Texture();
  const sphere = (x: number, radius = 1) => new Sphere(new Vector3(x, 0, 0), radius);
  const candidates = new Map([
    [far, [sphere(1000)]], [shared, [sphere(2000), sphere(325, 10)]], [near, [sphere(50)]],
  ]);
  assert.deepEqual(nearbyTextureWarmup(candidates, new Vector3(), 320), [near, shared]);
  assert.equal(candidates.size, 3, 'selection does not mark images as uploaded');
  assert.deepEqual(nearbyTextureWarmup(candidates, new Vector3(1000, 0, 0), 320), [far]);
});

test('cell prewarm stages source meshes with live lighting and restores state before each yield', async () => {
  const {prewarmCell} = await import('./startup-warmup.ts');
  const original = new THREE.WebGLRenderTarget(2, 2), beauty = new THREE.WebGLRenderTarget(4, 4), staging = new THREE.WebGLRenderTarget(8, 8);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(), light = new THREE.DirectionalLight(); scene.add(light);
  const material = new MeshBasicMaterial(), sources = Array.from({length: 65}, (_, i) => {
    const geometry = new BufferGeometry(); geometry.setAttribute('position', new Float32BufferAttribute([i, 0, 0, i+1, 0, 0, i, 1, 0], 3));
    const mesh = new Mesh(geometry, material); mesh.position.x = i; mesh.updateMatrixWorld();
    mesh.onBeforeRender=(_renderer,activeScene)=>assert.equal(activeScene,scene);
    mesh.onAfterRender=(_renderer,activeScene)=>assert.equal(activeScene,scene);
    return mesh;
  });
  let target: THREE.WebGLRenderTarget|null = original, autoClear = false, shadowAuto = true, shadowNeedsUpdate = true, resolveCompile!:()=>void;
  const seenStages: THREE.Scene[] = [];
  let polls=0, flushed=false, released=false;
  const fence={};
  const gl={SYNC_GPU_COMMANDS_COMPLETE:37143, TIMEOUT_EXPIRED:37147, WAIT_FAILED:37149,
    fenceSync:()=>fence, flush:()=>{flushed=true;},
    clientWaitSync:(sync:object,flags:number,timeout:number)=>{assert.equal(sync,fence);assert.ok(flushed);assert.equal(flags,0);assert.equal(timeout,0);return ++polls===1?37147:37148;},
    deleteSync:(sync:object)=>{assert.equal(sync,fence);released=true;}};
  const callbackGroup = new THREE.Group();
  const renderer = {
    autoClear, shadowMap: {autoUpdate: shadowAuto, needsUpdate: shadowNeedsUpdate},
    getRenderTarget:()=>target, getActiveCubeFace:()=>0, getActiveMipmapLevel:()=>0,
    setRenderTarget:(next:THREE.WebGLRenderTarget|null)=>{target=next;},
    getContext:()=>gl,
    compileAsync:(view:THREE.Object3D,_camera:THREE.Camera,targetScene:THREE.Scene)=>{assert.equal(view.children.length, 1);assert.deepEqual(targetScene.children,[light]);return new Promise<void>(resolve=>{resolveCompile=resolve;});},
    render:(stage:THREE.Scene)=>{seenStages.push(stage); assert.ok(stage.children.includes(light)); assert.ok(sources.some(source=>stage.children.includes(source))); assert.ok(sources.filter(source=>stage.children.includes(source)).every(source=>source.material===material));
      for(const source of sources)if(stage.children.includes(source)){source.onBeforeRender(renderer,stage,camera,source.geometry,material,callbackGroup);source.onAfterRender(renderer,stage,camera,source.geometry,material,callbackGroup);}},
  } as unknown as THREE.WebGLRenderer;
  Object.defineProperty(renderer, 'autoClear', {get:()=>autoClear, set:(v:boolean)=>{autoClear=v;}});
  Object.defineProperty(renderer.shadowMap, 'autoUpdate', {get:()=>shadowAuto, set:(v:boolean)=>{shadowAuto=v;}});
  Object.defineProperty(renderer.shadowMap, 'needsUpdate', {get:()=>shadowNeedsUpdate, set:(v:boolean)=>{shadowNeedsUpdate=v;}});
  const work = prewarmCell(renderer, scene, camera, beauty, staging, sources, [sources[0]], ()=>{}, async()=>{
    assert.equal(target, original); assert.equal(autoClear, false); assert.equal(shadowAuto, true); assert.equal(shadowNeedsUpdate,true);
  });
  assert.equal(target, original, 'compile target is restored before the readiness promise settles');
  assert.equal(autoClear, false); assert.equal(shadowAuto, true); assert.equal(shadowNeedsUpdate,true);
  resolveCompile(); await work;
  assert.equal(polls,2);assert.ok(released);
  assert.ok(seenStages.length > 0);
  assert.equal(target, original); assert.equal(autoClear, false); assert.equal(shadowAuto, true); assert.equal(shadowNeedsUpdate,true);
});

test('cell prewarm restores renderer state when aborting after a staged slice', async () => {
  const {prewarmCell} = await import('./startup-warmup.ts');
  const original = new THREE.WebGLRenderTarget(2, 2), beauty = new THREE.WebGLRenderTarget(4, 4), staging = new THREE.WebGLRenderTarget(8, 8);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(), material = new MeshBasicMaterial();
  const sources = Array.from({length: 65}, () => {const geometry=new BufferGeometry();geometry.setAttribute('position',new Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3));return new Mesh(geometry,material);});
  let target: THREE.WebGLRenderTarget|null = original, autoClear = false, shadowAuto = true, shadowNeedsUpdate = true, aborted = false;
  const renderer = {autoClear,shadowMap:{autoUpdate:shadowAuto,needsUpdate:shadowNeedsUpdate},getRenderTarget:()=>target,getActiveCubeFace:()=>0,getActiveMipmapLevel:()=>0,
    setRenderTarget:(next:THREE.WebGLRenderTarget|null)=>{target=next;},compileAsync:()=>Promise.resolve(),render:()=>{}} as unknown as THREE.WebGLRenderer;
  Object.defineProperty(renderer,'autoClear',{get:()=>autoClear,set:(v:boolean)=>{autoClear=v;}});
  Object.defineProperty(renderer.shadowMap,'autoUpdate',{get:()=>shadowAuto,set:(v:boolean)=>{shadowAuto=v;}});
  Object.defineProperty(renderer.shadowMap,'needsUpdate',{get:()=>shadowNeedsUpdate,set:(v:boolean)=>{shadowNeedsUpdate=v;}});
  await assert.rejects(prewarmCell(renderer,scene,camera,beauty,staging,sources,[],()=>{if(aborted)throw Error('aborted');},async()=>{assert.equal(target,original);assert.equal(autoClear,false);assert.equal(shadowAuto,true);assert.equal(shadowNeedsUpdate,true);aborted=true;}),/aborted/);
  assert.equal(target, original); assert.equal(autoClear, false); assert.equal(shadowAuto, true); assert.equal(shadowNeedsUpdate,true);
});

test('BatchedMesh warmup bounds native multi-draw primitives and restores cached counts and hooks',()=>{
 const geometry=new THREE.BoxGeometry(),material=new MeshBasicMaterial(),batch=new THREE.BatchedMesh(10,240,360,material);
 const id=batch.addGeometry(geometry);for(let i=0;i<10;i++)batch.addInstance(id);
 batch.perObjectFrustumCulled=false;batch.sortObjects=false;
 const root=new Group();root.add(batch);root.updateMatrixWorld(true);
 const camera=new THREE.PerspectiveCamera();camera.position.z=5;camera.updateMatrixWorld();
 const native=batch.onBeforeRender,after=batch.onAfterRender;
 const callbackGroup=new Group();
 const draw=batch as unknown as {_multiDrawCount:number;_multiDrawCounts:Int32Array};
 withWarmupPrimitives(root,()=>{
  for(let pass=0;pass<2;pass++){
   batch.onBeforeRender({} as THREE.WebGLRenderer,new THREE.Scene(),camera,batch.geometry,material,callbackGroup);
   assert.equal(draw._multiDrawCount,1);assert.equal(draw._multiDrawCounts[0],3);
   batch.onAfterRender({} as THREE.WebGLRenderer,new THREE.Scene(),camera,batch.geometry,material,callbackGroup);
   assert.equal(draw._multiDrawCount,10);assert.equal(draw._multiDrawCounts[0],36);
  }
 });
 assert.equal(batch.onBeforeRender,native);assert.equal(batch.onAfterRender,after);
 assert.equal(draw._multiDrawCount,10);assert.equal(draw._multiDrawCounts[0],36);
 assert.throws(()=>withWarmupPrimitives(root,()=>{
  batch.onBeforeRender({} as THREE.WebGLRenderer,new THREE.Scene(),camera,batch.geometry,material,callbackGroup);throw Error('failed draw');
 }),/failed draw/);
 assert.equal(draw._multiDrawCount,10);assert.equal(draw._multiDrawCounts[0],36);assert.equal(batch.onBeforeRender,native);
 batch.dispose();geometry.dispose();material.dispose();
});

test('optional preparation trace is bounded, chronological, isolated, and one-shot',async()=>{
 const {createCellPreparationTrace}=await import('./startup-warmup.ts');let now=10;
 const trace=createCellPreparationTrace(2,()=>now),finish=trace.begin('first','parse');
 now=15;finish();finish();assert.equal(trace.read().length,1);
 trace.record('second','compile-submit',20,2);trace.record('third','attach',30,1);
 const entries=trace.read();assert.deepEqual(entries,[
  {cell:'second',phase:'compile-submit',startMs:20,durationMs:2},
  {cell:'third',phase:'attach',startMs:30,durationMs:1}]);
 entries[0].phase='changed';assert.equal(trace.read()[0].phase,'compile-submit');
});
