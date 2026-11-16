import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import * as THREE from 'three';
import {buildTreeCell,packTreeCell,hydrateTreeCell,validateTreeCell,treeDetailUrl,treeDetailBinding,createTreeDetailLayer,type TreeCellAsset} from './tree-detail.ts';
import {finishCanopyImpostor} from './canopy-impostor.ts';
import {createTreeMaterials} from './tree-detail-geometry.ts';
import {buildLeafTwigs} from './tree-architecture.ts';
import {coarseLeafSources} from './tree-foliage-lod.ts';
import {estimateTreeCell} from './tree-detail-budget.ts';

const publicDir=new URL('../public/',import.meta.url);
const index=JSON.parse(readFileSync(new URL('compare/neighborhood-stream.json',publicDir),'utf8'));
const manifestUrl=await treeDetailUrl(index);
const manifest=JSON.parse(readFileSync(new URL(manifestUrl.slice(1),publicDir),'utf8'));
const skeletonRaw=gunzipSync(readFileSync(new URL(manifest.skeleton.binary.slice(1),publicDir)));
const sourceSkeleton=new Float32Array(skeletonRaw.buffer.slice(skeletonRaw.byteOffset,skeletonRaw.byteOffset+skeletonRaw.byteLength));
const decode=(asset:TreeCellAsset)=>{
 const bytes=gunzipSync(readFileSync(new URL(asset.binary.slice(1),publicDir)));
 return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength) as ArrayBuffer;
};

type TreeOptions=NonNullable<Parameters<typeof createTreeDetailLayer>[3]>;
async function controllerHarness(cells:{id:string;boundsEN:number[];source?:TreeCellAsset}[],options:TreeOptions={}){
 const fixture=[...manifest.treeCells].filter(c=>c.leafSamples>0).sort((a,b)=>a.leafSamples-b.leafSamples)[0],texture=new THREE.Texture();
 const assets=cells.map(c=>({...fixture,...c.source,...c,contentHash:c.id+'-source'}));
 const index={sourceWorldHash:'race-test',cells:assets.map(c=>({id:c.id,contentHash:c.contentHash}))};
 const fake={...treeDetailBinding(index),treeCells:assets,stats:{},skeleton:manifest.skeleton};
 const originalWorker=globalThis.Worker,originalFetch=globalThis.fetch;let builds=0;
 class CpuWorker{
  onmessage:((event:{data:unknown})=>void)|undefined;
  postMessage({id,asset,skeleton,quality}:{id:number;asset:TreeCellAsset;skeleton?:ArrayBuffer;quality?:'fine'|'coarse'}){
   if(skeleton)return;builds++;
   const cell=buildTreeCell(asset,decode(asset),texture,()=>{},{skeleton:sourceSkeleton,quality}),packed=packTreeCell(cell);
   queueMicrotask(()=>this.onmessage?.({data:{id,packed}}));cell.dispose();
  }
  terminate(){}
 }
 globalThis.Worker=CpuWorker as unknown as typeof Worker;
 globalThis.fetch=(async(url)=>String(url)===manifest.skeleton.binary?new Response(Uint8Array.from(readFileSync(new URL(manifest.skeleton.binary.slice(1),publicDir)))):new Response(JSON.stringify(fake))) as typeof fetch;
 try{
  const layer=await createTreeDetailLayer(index,texture,()=>{},options);
  return {layer,builds:()=>builds,tick:()=>new Promise<void>(resolve=>setImmediate(resolve)),
   dispose(){layer.dispose();globalThis.Worker=originalWorker;globalThis.fetch=originalFetch;texture.dispose();}};
 }catch(error){globalThis.Worker=originalWorker;globalThis.fetch=originalFetch;texture.dispose();throw error;}
}

void test('shipped tree package represents all production source samples and branch faces',()=>{
 assert.equal(manifest.stats.sourceSamples,index.base.stats.canopies.sourceSamples);
 assert.equal(manifest.stats.barkTriangles,index.base.stats.canopies.barkTriangles);
 assert.equal(manifest.treeCells.length,313);
 assert.ok(manifest.stats.maxInputCellBytes<11*1024**2);
 for(const asset of manifest.treeCells){
  assert.equal(asset.contentHash,index.cells.find((c:{id:string})=>c.id===asset.id).contentHash);
  validateTreeCell(asset,decode(asset));
 }
});

void test('near and remote source trees use identical geometry paths; hydration does no sample expansion',t=>{
 const ordered=[...manifest.treeCells].sort((a,b)=>b.sourceSamples-a.sourceSamples);
 const selected=[ordered[0],manifest.treeCells.find((c:{id:string})=>c.id==='grid-0-0'),ordered.at(-1)];
 const texture=new THREE.Texture();
 for(const asset of selected){
  const buffer=decode(asset),before=buffer.slice(0),start=performance.now(),cell=buildTreeCell(asset,buffer,texture,()=>{},{skeleton:sourceSkeleton});
  const buildMs=performance.now()-start,packed=packTreeCell(cell),hydrateStart=performance.now();
  const hydrated=hydrateTreeCell(packed,texture,()=>{}),hydrateMs=performance.now()-hydrateStart;
  assert.equal(hydrated.foliageMeshes.reduce((n,m)=>n+m.geometry.getAttribute('position').count/4,0),asset.leafSamples);
  assert.equal(hydrated.renderables.filter(m=>m.name.endsWith('___bark')).reduce((n,m)=>n+m.geometry.index!.count/3,0),asset.barkTriangles+cell.twigSegments*6);
  assert.ok(hydrated.renderables.length<=3);assert.ok(hydrated.foliageMeshes.length<=1);
  assert.equal(hydrated.geometryBytes,cell.geometryBytes);
  assert.equal(hydrated.bytes,cell.geometryBytes*2+buffer.byteLength+cell.transitionBytes);
  assert.equal(cell.transitionBytes,cell.renderables.reduce((n,m)=>n+m.geometry.getAttribute('position').count*4,0));
  assert.ok(cell.twigSegments<=2048);
  assert.ok(hydrated.renderables.every(m=>m.castShadow&&m.receiveShadow&&m.userData.streamCell===asset.id));
  if(asset.id==='grid-0-0'){
   const bark=asset.entries.find((e:{kind:string})=>e.kind==='bark');
   assert.notEqual(bark.indices%12,0,'fixture must include boundary-clipped source cylinders');
   const original=new Float32Array(before,bark.offset,bark.vertices*3),finished=new Float32Array(buffer,bark.offset,bark.vertices*3);
   assert.ok(original.some((v,i)=>Math.abs(v-finished[i])>1e-4),'intact limbs in a clipped cell remained blunt cylinders');
  }
  for(let i=0;i<cell.renderables.length;i++)assert.equal(hydrated.renderables[i].geometry.getAttribute('position').array,cell.renderables[i].geometry.getAttribute('position').array);
  t.diagnostic(JSON.stringify({id:asset.id,samples:asset.sourceSamples,triangles:asset.barkTriangles,workerBuildMs:+buildMs.toFixed(2),mainHydrateMs:+hydrateMs.toFixed(2),residentBytes:cell.bytes,draws:cell.renderables.length}));
  hydrated.dispose();cell.dispose();
 }
 texture.dispose();
});

void test('tree byte validation rejects corrupt ranges before renderer allocation',()=>{
 const asset=manifest.treeCells[0],buffer=decode(asset);
 assert.throws(()=>validateTreeCell({...asset,byteLength:buffer.byteLength+4},buffer));
 assert.throws(()=>validateTreeCell({...asset,entries:[{...asset.entries[0],offset:4}]},buffer));
 const corrupt=buffer.slice(0);new Float32Array(corrupt,asset.entries[0].offset,1)[0]=NaN;
 assert.throws(()=>validateTreeCell(asset,corrupt));
});

void test('fallback canopy shader removes intersecting plane lighting and preserves bark',()=>{
 const material=new THREE.MeshStandardMaterial(),texture=new THREE.Texture();finishCanopyImpostor(material,texture);
 const shader={uniforms:{},vertexShader:'#include <common>\n#include <project_vertex>',fragmentShader:'#include <common>\n#include <map_fragment>\n#include <normal_fragment_maps>'};
 material.onBeforeCompile(shader as never,{} as never);
 assert.ok(shader.fragmentShader.includes('mix(normal,canopySoftNormal,canopyLeafShare)'));
 assert.ok(shader.fragmentShader.includes('mix(diffuseColor.rgb,canopyOlive,canopyLeafShare)'));
 assert.ok(shader.fragmentShader.includes('canopyDetail'));
 material.dispose();texture.dispose();
});

void test('saturated slots yield to nearer cells after retirement; staged and fading bytes stay counted',async()=>{
 const fixture=manifest.treeCells.at(-1),buffer=decode(fixture),texture=new THREE.Texture();
 const cells=[{...fixture,id:'behind',contentHash:'a',boundsEN:[-1,-1,1,1]},{...fixture,id:'ahead',contentHash:'b',boundsEN:[89,-1,91,1]}];
 const fakeIndex={sourceWorldHash:'test-source',cells:cells.map(c=>({id:c.id,contentHash:c.contentHash}))};
 const fakeManifest={...treeDetailBinding(fakeIndex),treeCells:cells,stats:{},skeleton:manifest.skeleton};
 const originalWorker=globalThis.Worker,originalFetch=globalThis.fetch;
 let builds=0,prepared=0,finishRetirement:(()=>void)|undefined;
 class CpuWorker{
  onmessage:((event:{data:unknown})=>void)|undefined;
  postMessage({id,asset,skeleton,quality}:{id:number;asset:TreeCellAsset;skeleton?:ArrayBuffer;quality?:'fine'|'coarse'}){
   if(skeleton)return;
   builds++;const cell=buildTreeCell(asset,buffer.slice(0),texture,()=>{},{skeleton:sourceSkeleton,quality}),packed=packTreeCell(cell);
   queueMicrotask(()=>this.onmessage?.({data:{id,packed}}));
  }
  terminate(){}
 }
 globalThis.Worker=CpuWorker as unknown as typeof Worker;
 globalThis.fetch=(async(url)=>String(url)===manifest.skeleton.binary?new Response(Uint8Array.from(readFileSync(new URL(manifest.skeleton.binary.slice(1),publicDir)))):new Response(JSON.stringify(fakeManifest))) as typeof fetch;
 const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
 try{
  const layer=await createTreeDetailLayer(fakeIndex,texture,()=>{},{maxCells:1,loadM:105,unloadM:145,
   prepare:async()=>{prepared++;assert.ok(layer.stats.residentBytes>0);assert.equal(layer.readyCells.size,0);},
   retire:()=>new Promise<void>(resolve=>{finishRetirement=resolve;}),
  });
  assert.deepEqual([...layer.availableCells],['behind','ahead']);
  layer.update({x:0,z:0},{x:1,z:0});await tick();
  assert.deepEqual([...layer.readyCells.keys()],['behind']);assert.equal(prepared,1);
  const resident=layer.stats.residentBytes;
  layer.update({x:70,z:0},{x:1,z:0});await tick();
  assert.equal(layer.stats.evictions,1);assert.deepEqual([...layer.readyCells.keys()],['behind']);
  assert.equal(layer.stats.residentBytes,resident);assert.equal(builds,1);
  finishRetirement!();await tick();layer.update({x:70,z:0},{x:1,z:0});await tick();
  assert.deepEqual([...layer.readyCells.keys()],['ahead']);assert.equal(builds,2);
  for(let i=0;i<100;i++)layer.update({x:70,z:0},{x:1,z:0});await tick();
  assert.equal(builds,2,'parked camera caused repeated geometry work');
  layer.dispose();assert.equal(layer.stats.residentBytes,0);
 }finally{globalThis.Worker=originalWorker;globalThis.fetch=originalFetch;texture.dispose();}
});

void test('older neighborhoods and browsers without workers keep all native tree classes available',async()=>{
 const originalWorker=globalThis.Worker,originalFetch=globalThis.fetch,texture=new THREE.Texture();
 try{
  globalThis.Worker=undefined as unknown as typeof Worker;
  globalThis.fetch=(async()=>new Response(JSON.stringify(manifest))) as typeof fetch;
  let layer=await createTreeDetailLayer(index,texture,()=>{});
  assert.equal(layer.availableCells.size,0);assert.equal(layer.coveredClasses.size,0);
  layer.update({x:0,z:0});assert.equal(layer.stats.loading,false);layer.dispose();
  globalThis.Worker=originalWorker;
  globalThis.fetch=(async()=>new Response('',{status:404})) as typeof fetch;
  layer=await createTreeDetailLayer(index,texture,()=>{});
  assert.equal(layer.availableCells.size,0);assert.equal(layer.readyCells.size,0);layer.dispose();
 }finally{globalThis.Worker=originalWorker;globalThis.fetch=originalFetch;texture.dispose();}
});

void test('coarse leaves retain existing points/extrema with stable source-order-independent selection',()=>{
 const points:number[]=[],colors:number[]=[];
 for(let x=-4;x<=4;x+=.5)for(let y=4;y<=10;y+=.5)for(let z=-4;z<=4;z+=.5){points.push(x,y,z);colors.push(.1,.2,.03);}
 const make=(reverse:boolean)=>{
  const rows=Array.from({length:points.length/3},(_,i)=>i);if(reverse)rows.reverse();
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(rows.flatMap(i=>points.slice(i*3,i*3+3)),3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(rows.flatMap(i=>colors.slice(i*3,i*3+3)),3));
  return new THREE.Mesh(geometry,new THREE.MeshStandardMaterial());
 };
 const segments=[{root:new THREE.Vector3(0,0,0),tip:new THREE.Vector3(0,5,0),radius:.2,owner:'source-tree'}];
 const a=make(false),b=make(true),first=coarseLeafSources([a],segments),second=coarseLeafSources([b],segments);
 const p=first[0].geometry.getAttribute('position');assert.ok(p.count<=112);assert.ok(p.count>50);
 assert.deepEqual(Array.from(p.array),Array.from(second[0].geometry.getAttribute('position').array));
 const original=new Set(Array.from({length:points.length/3},(_,i)=>points.slice(i*3,i*3+3).join(',')));
 for(let i=0;i<p.count;i++)assert.ok(original.has([p.getX(i),p.getY(i),p.getZ(i)].join(',')));
 first[0].geometry.computeBoundingBox();a.geometry.computeBoundingBox();assert.deepEqual(first[0].geometry.boundingBox,a.geometry.boundingBox);
 const before=Array.from(a.geometry.getAttribute('position').array),twig=buildLeafTwigs([a],segments,100);
 assert.ok(twig);assert.ok(twig.userData.twigSegments<=100);assert.deepEqual(Array.from(a.geometry.getAttribute('position').array),before);
 for(const source of [a,b,...first,...second]){source.geometry.dispose();(source.material as THREE.Material).dispose();}twig.dispose();
});

void test('all hydrated cells share exactly three layer-owned materials; cell retirement releases only geometry',()=>{
 const asset=manifest.treeCells.at(-1),texture=new THREE.Texture(),source=buildTreeCell(asset,decode(asset),texture,()=>{},{skeleton:sourceSkeleton});
 let setups=0,disposals=0;const materials=createTreeMaterials(texture,()=>{setups++;});
 for(const material of [materials.bark,materials.leaf,materials.frond])material.addEventListener('dispose',()=>disposals++);
 const a=hydrateTreeCell(packTreeCell(source),texture,()=>{throw Error('shared materials were registered again');},materials);
 const b=hydrateTreeCell(packTreeCell(source),texture,()=>{throw Error('shared materials were registered again');},materials);
 assert.equal(setups,3);assert.equal(a.foliageMeshes[0].material,b.foliageMeshes[0].material);
 a.dispose();b.dispose();assert.equal(disposals,0);materials.dispose();assert.equal(disposals,3);source.dispose();texture.dispose();
});

void test('LOD replacement holds both displays/budgets until fade resolves and never removes ready source identity',async()=>{
 const source=manifest.treeCells.at(-1),asset={...source,id:'one-cell',contentHash:'one-hash',boundsEN:[89,-1,91,1]},buffer=decode(source),texture=new THREE.Texture();
 const index={sourceWorldHash:'test-lod',cells:[{id:asset.id,contentHash:asset.contentHash}]},fake={...treeDetailBinding(index),treeCells:[asset],stats:{},skeleton:manifest.skeleton};
 const originalWorker=globalThis.Worker,originalFetch=globalThis.fetch;let finishFade:(()=>void)|undefined,oldDisplay:ReturnType<typeof buildTreeCell>|undefined;
 class CpuWorker{onmessage:((event:{data:unknown})=>void)|undefined;postMessage({id,asset,skeleton,quality}:{id:number;asset:TreeCellAsset;skeleton?:ArrayBuffer;quality?:'fine'|'coarse'}){if(skeleton)return;const cell=buildTreeCell(asset,buffer.slice(0),texture,()=>{},{skeleton:sourceSkeleton,quality});queueMicrotask(()=>this.onmessage?.({data:{id,packed:packTreeCell(cell)}}));}terminate(){}}
 globalThis.Worker=CpuWorker as unknown as typeof Worker;
 globalThis.fetch=(async(url)=>String(url)===manifest.skeleton.binary?new Response(Uint8Array.from(readFileSync(new URL(manifest.skeleton.binary.slice(1),publicDir)))):new Response(JSON.stringify(fake))) as typeof fetch;
 const tick=()=>new Promise<void>(resolve=>setImmediate(resolve));
 try{
  const layer=await createTreeDetailLayer(index,texture,()=>{},{replace:async(incoming,previous)=>{
   assert.equal(incoming.quality,'fine');assert.equal(previous.quality,'coarse');oldDisplay=previous;
   assert.equal(layer.root.children.length,2);assert.ok(layer.stats.residentBytes>=incoming.bytes+previous.bytes);assert.equal(layer.stats.drawCalls,incoming.renderables.length+previous.renderables.length);
   await new Promise<void>(resolve=>{finishFade=resolve;});
  }});
  layer.update({x:0,z:0});await tick();assert.equal(layer.readyCells.get(asset.id)?.quality,'coarse');
  layer.update({x:70,z:0});await tick();assert.equal(layer.readyCells.get(asset.id)?.quality,'fine');
  assert.equal(layer.root.children.length,2);assert.ok(oldDisplay?.root.parent);assert.equal(layer.readyCells.size,1);
  finishFade!();await tick();assert.equal(layer.root.children.length,1);assert.equal(oldDisplay?.root.parent,null);assert.equal(layer.readyCells.size,1);
  layer.dispose();
 }finally{globalThis.Worker=originalWorker;globalThis.fetch=originalFetch;texture.dispose();}
});

void test('renewed distance demand restores the same cell; stale retirement cannot delete a later retirement',async()=>{
 const retirements:{signal:AbortSignal;finish:()=>void}[]=[];let restores=0;
 const h=await controllerHarness([{id:'tree',boundsEN:[-1,-1,1,1]}],{loadM:100,unloadM:145,
  retire:(_cell,signal)=>new Promise<void>(finish=>retirements.push({signal,finish})),restore:()=>{restores++;},
 });
 try{
  h.layer.update({x:0,z:0});await h.tick();const cell=h.layer.readyCells.get('tree')!;
  let disposals=0;cell.renderables[0].geometry.addEventListener('dispose',()=>disposals++);
  const bytes=h.layer.stats.residentBytes;
  h.layer.update({x:200,z:0});await h.tick();assert.equal(retirements.length,1);
  h.layer.update({x:0,z:0});await h.tick();
  assert.equal(retirements[0].signal.aborted,true);assert.equal(restores,1);assert.equal(h.builds(),1);
  assert.equal(h.layer.readyCells.get('tree'),cell);assert.equal(h.layer.stats.residentBytes,bytes);assert.equal(h.layer.root.children.length,1);assert.equal(disposals,0);
  h.layer.update({x:200,z:0});await h.tick();assert.equal(retirements.length,2);
  retirements[0].finish();await h.tick();
  assert.equal(h.layer.readyCells.get('tree'),cell);assert.equal(disposals,0);assert.equal(retirements[1].signal.aborted,false);
  retirements[1].finish();await h.tick();assert.equal(h.layer.readyCells.size,0);assert.equal(disposals,1);
 }finally{h.dispose();}
});

void test('turning toward an evicted tree cancels budget retirement without an immediate re-eviction or reload',async()=>{
 let retireSignal:AbortSignal|undefined,finishRetire:(()=>void)|undefined,retireCalls=0,restores=0;
 const h=await controllerHarness([{id:'a',boundsEN:[-1,-1,1,1]},{id:'b',boundsEN:[89,-1,91,1]}],{maxCells:1,loadM:105,unloadM:145,
  retire:(_cell,signal)=>{retireCalls++;retireSignal=signal;return new Promise<void>(resolve=>{finishRetire=resolve;});},restore:()=>{restores++;},
 });
 try{
  h.layer.update({x:0,z:0},{x:1,z:0});await h.tick();const cell=h.layer.readyCells.get('a')!,bytes=h.layer.stats.residentBytes;
  h.layer.update({x:70,z:0},{x:1,z:0});await h.tick();assert.equal(retireCalls,1);
  h.layer.update({x:70,z:0},{x:-1,z:0});await h.tick();
  assert.equal(retireSignal?.aborted,true);assert.equal(restores,1);assert.equal(h.layer.readyCells.get('a'),cell);assert.equal(h.layer.stats.residentBytes,bytes);
  assert.equal(h.builds(),1);assert.equal(retireCalls,1);assert.equal(h.layer.root.children.length,1);
  finishRetire!();await h.tick();for(let i=0;i<20;i++)h.layer.update({x:70,z:0},{x:-1,z:0});await h.tick();
  assert.equal(h.layer.readyCells.get('a'),cell);assert.equal(retireCalls,1);assert.equal(h.layer.stats.restoredEvictions,1);
 }finally{h.dispose();}
});

void test('reversing an in-flight LOD fade retains both complementary displays until restoration finishes',async()=>{
 const replaces:{signal:AbortSignal;finish:()=>void}[]=[];let finishRestore:(()=>void)|undefined;
 const h=await controllerHarness([{id:'tree',boundsEN:[89,-1,91,1]}],{
  replace:(_incoming,_previous,signal)=>new Promise<void>(finish=>replaces.push({signal,finish})),
  restore:()=>new Promise<void>(resolve=>{finishRestore=resolve;}),
 });
 try{
  h.layer.update({x:0,z:0});await h.tick();const previous=h.layer.readyCells.get('tree')!;
  h.layer.update({x:70,z:0});await h.tick();const incoming=h.layer.readyCells.get('tree')!;
  let oldDisposed=0,newDisposed=0;previous.renderables[0].geometry.addEventListener('dispose',()=>oldDisposed++);incoming.renderables[0].geometry.addEventListener('dispose',()=>newDisposed++);
  assert.equal(previous.quality,'coarse');assert.equal(incoming.quality,'fine');const bytes=h.layer.stats.residentBytes;
  h.layer.update({x:0,z:0});await h.tick();assert.equal(replaces[0].signal.aborted,true);
  assert.equal(h.layer.readyCells.get('tree'),previous);assert.equal(h.layer.root.children.length,2);assert.equal(h.layer.stats.residentBytes,bytes);
  replaces[0].finish();await h.tick();assert.equal(oldDisposed,0);assert.equal(newDisposed,0);assert.equal(h.layer.root.children.length,2);
  finishRestore!();await h.tick();assert.equal(h.layer.readyCells.get('tree'),previous);assert.equal(h.layer.root.children.length,1);assert.equal(newDisposed,1);assert.equal(oldDisposed,0);
  h.layer.update({x:70,z:0});await h.tick();assert.equal(replaces.length,2);assert.equal(h.layer.root.children.length,2);
  h.layer.dispose();assert.equal(replaces[1].signal.aborted,true);assert.equal(oldDisposed,1);assert.equal(h.layer.stats.residentBytes,0);
  replaces[1].finish();await h.tick();assert.equal(h.layer.readyCells.size,0);assert.equal(h.layer.root.children.length,0);
 }finally{h.dispose();}
});

void test('a now-unwanted LOD warmed asynchronously is discarded before attachment without failing the source cell',async()=>{
 let finishPrepare:(()=>void)|undefined,replaceCalls=0;
 const h=await controllerHarness([{id:'tree',boundsEN:[89,-1,91,1]}],{
  prepare:cell=>cell.quality==='fine'?new Promise<void>(resolve=>{finishPrepare=resolve;}):Promise.resolve(),
  replace:async()=>{replaceCalls++;},
 });
 try{
  h.layer.update({x:0,z:0});await h.tick();const previous=h.layer.readyCells.get('tree')!,bytes=h.layer.stats.residentBytes;
  h.layer.update({x:70,z:0});await h.tick();assert.ok(finishPrepare);assert.equal(h.layer.readyCells.get('tree'),previous);
  h.layer.update({x:0,z:0});finishPrepare!();await h.tick();
  assert.equal(h.layer.readyCells.get('tree'),previous);assert.equal(h.layer.root.children.length,1);assert.equal(h.layer.stats.residentBytes,bytes);
  assert.equal(replaceCalls,0);assert.equal(h.layer.availableCells.has('tree'),true);assert.equal(h.layer.stats.errors,0);
 }finally{h.dispose();}
});

void test('final disposal aborts outstanding retirement and ignores its late completion',async()=>{
 let signal:AbortSignal|undefined,finish:(()=>void)|undefined;
 const h=await controllerHarness([{id:'tree',boundsEN:[-1,-1,1,1]}],{loadM:100,unloadM:145,
  retire:(_cell,s)=>{signal=s;return new Promise<void>(resolve=>{finish=resolve;});},
 });
 try{
  h.layer.update({x:0,z:0});await h.tick();const cell=h.layer.readyCells.get('tree')!;let disposals=0;cell.renderables[0].geometry.addEventListener('dispose',()=>disposals++);
  h.layer.update({x:200,z:0});await h.tick();h.layer.dispose();assert.equal(signal?.aborted,true);assert.equal(disposals,1);
  finish!();await h.tick();assert.equal(disposals,1);assert.equal(h.layer.readyCells.size,0);assert.equal(h.layer.stats.residentBytes,0);
 }finally{h.dispose();}
});

void test('requested-quality bounds cover actual source buffers and make coarse admission cheaper than fine',()=>{
 const texture=new THREE.Texture();
 for(const asset of [manifest.treeCells.find((c:{id:string})=>c.id==='grid-1-1'),manifest.treeCells.find((c:{id:string})=>c.id==='grid-0-0'),manifest.treeCells.at(-1)]){
  const owners=new Set(asset.skeletonIds.map((id:number)=>sourceSkeleton[id*8+7])).size;
  const fine=estimateTreeCell(asset,'fine',owners),coarse=estimateTreeCell(asset,'coarse',owners);
  assert.ok(coarse.triangles<=fine.triangles);assert.ok(coarse.bytes<=fine.bytes);
  for(const quality of ['fine','coarse'] as const){
   const cell=buildTreeCell(asset,decode(asset),texture,()=>{},{skeleton:sourceSkeleton,quality}),bound=quality==='fine'?fine:coarse;
   assert.ok(cell.bytes<=bound.bytes);assert.ok(cell.triangles<=bound.triangles);assert.ok(cell.renderables.length<=bound.draws);cell.dispose();
  }
 }
 const dense=manifest.treeCells.find((c:{id:string})=>c.id==='grid-1-1');
 const bound=estimateTreeCell(dense,'coarse',new Set(dense.skeletonIds.map((id:number)=>sourceSkeleton[id*8+7])).size);
 assert.equal(bound.leafCards,3920);assert.equal(bound.twigSegments,768);assert.equal(bound.triangles,16964);assert.ok(371000+bound.triangles<=400000);
 texture.dispose();
});

void test('coarse request fits its remaining global budget without reserving or evicting a fine variant',async()=>{
 const source=manifest.treeCells.find((c:{id:string})=>c.id==='grid-1-1'),owners=new Set(source.skeletonIds.map((id:number)=>sourceSkeleton[id*8+7])).size;
 const fine=estimateTreeCell(source,'fine',owners),coarse=estimateTreeCell(source,'coarse',owners);
 const h=await controllerHarness([{id:'near',boundsEN:[-1,-1,1,1],source},{id:'middle',boundsEN:[159,-1,161,1],source}],{maxTriangles:fine.triangles+coarse.triangles});
 try{
  h.layer.update({x:0,z:0},{x:1,z:0});await h.tick();h.layer.update({x:0,z:0},{x:1,z:0});await h.tick();
  assert.equal(h.layer.readyCells.get('near')?.quality,'fine');assert.equal(h.layer.readyCells.get('middle')?.quality,'coarse');
  assert.equal(h.layer.stats.evictions,0);assert.equal(h.builds(),2);assert.ok(h.layer.stats.triangles<=fine.triangles+coarse.triangles);
 }finally{h.dispose();}
});

void test('global high-priority member holds admission while a non-member would fit transient free space',async()=>{
 const source=manifest.treeCells.find((c:{id:string})=>c.id==='grid-1-1'),owners=new Set(source.skeletonIds.map((id:number)=>sourceSkeleton[id*8+7])).size;
 const cap=estimateTreeCell(source,'fine',owners).triangles;let finishRetire:(()=>void)|undefined;
 const h=await controllerHarness([{id:'old',boundsEN:[139,-1,141,1],source},{id:'priority',boundsEN:[339,-1,341,1],source},{id:'cheap',boundsEN:[344,-1,346,1]}],{maxTriangles:cap,
  retire:()=>new Promise<void>(resolve=>{finishRetire=resolve;}),
 });
 try{
  h.layer.update({x:0,z:0},{x:1,z:0});await h.tick();const old=h.layer.readyCells.get('old')!;assert.equal(old.quality,'coarse');
  h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();
  assert.deepEqual([...h.layer.desiredCells.keys()],['priority']);assert.ok(finishRetire);assert.equal(h.builds(),1);
  assert.ok(old.triangles+300<cap,'cheap non-member should physically fit while the incumbent fades');
  for(let i=0;i<20;i++){h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();}
  assert.equal(h.builds(),1);assert.equal(h.layer.readyCells.has('cheap'),false);
  finishRetire!();await h.tick();h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();
  assert.deepEqual([...h.layer.readyCells.keys()],['priority']);assert.equal(h.builds(),2);assert.equal(h.layer.stats.evictions,1);
 }finally{h.dispose();}
});

void test('budget victim is not reinserted when actual cost frees space; a real view turn can restore admission',async()=>{
 const source=manifest.treeCells.find((c:{leafSamples:number})=>c.leafSamples>=180&&c.leafSamples<500),owners=new Set(source.skeletonIds.map((id:number)=>sourceSkeleton[id*8+7])).size;
 const cap=estimateTreeCell(source,'fine',owners).triangles;
 const h=await controllerHarness([{id:'cheap-victim',boundsEN:[139,-1,141,1]},{id:'priority',boundsEN:[339,-1,341,1],source}],{maxTriangles:cap});
 try{
  h.layer.update({x:0,z:0},{x:1,z:0});await h.tick();const old=h.layer.readyCells.get('cheap-victim')!;
  h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();
  const priority=h.layer.readyCells.get('priority')!;assert.ok(priority);
  assert.ok(priority.triangles+old.triangles<=cap,'retired cheap variant should now fit actual remaining space');
  for(let i=0;i<25;i++){h.layer.update({x:340,z:0},{x:1,z:0});await h.tick();}
  assert.deepEqual([...h.layer.readyCells.keys()],['priority']);assert.equal(h.layer.stats.reloads,0);assert.equal(h.builds(),2);
  h.layer.update({x:340,z:0},{x:-1,z:0});await h.tick();
  assert.equal(h.layer.readyCells.has('cheap-victim'),true);assert.equal(h.layer.stats.reloads,1);
 }finally{h.dispose();}
});

void test('small distance jitter preserves global membership instead of alternating cheap source-cell admissions',async()=>{
 const h=await controllerHarness([{id:'a',boundsEN:[-1,-1,1,1]},{id:'b',boundsEN:[9,-1,11,1]}],{maxCells:1});
 try{
  h.layer.update({x:0,z:0});await h.tick();
  for(let i=0;i<30;i++){h.layer.update({x:i%2?4.4:5.6,z:0});await h.tick();}
  assert.deepEqual([...h.layer.readyCells.keys()],['a']);assert.equal(h.builds(),1);assert.equal(h.layer.stats.evictions,0);
  h.layer.update({x:9,z:0});await h.tick();h.layer.update({x:9,z:0});await h.tick();
  assert.deepEqual([...h.layer.readyCells.keys()],['b']);assert.equal(h.builds(),2);
  for(let i=0;i<30;i++){h.layer.update({x:i%2?5.1:6.1,z:0});await h.tick();}
  assert.deepEqual([...h.layer.readyCells.keys()],['b']);assert.equal(h.builds(),2);assert.equal(h.layer.stats.reloads,0);
 }finally{h.dispose();}
});
