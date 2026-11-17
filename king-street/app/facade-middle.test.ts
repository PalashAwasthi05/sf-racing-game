import assert from 'node:assert/strict';
import test from 'node:test';
import {gzipSync} from 'node:zlib';
import * as THREE from 'three';
import {createFacadeMiddleController,decodeFacadeBatch,validateFacadeMiddleIndex,loadFacadeMiddleIndex,type FacadeCell,type FacadeMiddleIndex} from './facade-middle.ts';

function fixture(id='a',east=0){
 const batch={offset:0,vertices:3,indices:3,min:[east,0,0],step:[.001,.001,.001],glazing:true,residentBytes:60,quantizationErrorM:.0005};
 const raw=new Uint8Array(44);const view=new DataView(raw.buffer);
 [0,0,0,1000,0,0,0,1000,0].forEach((n,i)=>view.setUint16(i*2,n,true));
 raw.set([0,0,127,0,0,127,0,0,127],18);raw.fill(128,27,36);[0,1,2].forEach((n,i)=>view.setUint16(36+i*2,n,true));
 const cell:FacadeCell={id,boundsEN:[east,-1,east+1,1],tileEN:[east,-1,east+1,1],sourceHash:'a'.repeat(12),url:'/compare/facade-cell-0123456789abcdef.bin.gz',sha256:'a'.repeat(64),compressedBytes:30,byteLength:44,batches:[batch],
  stats:{triangles:1,vertices:3,residentBytes:60,batches:1,buildings:1,sourceGlazingTriangles:1,glazingTriangles:1}};
 return {cell,raw};
}
function index(cells:FacadeCell[]):FacadeMiddleIndex{return {version:1,format:'facade-middle-v1',sourceWorldHash:'source',complete:true,sourceCells:cells.map(c=>({id:c.id,contentHash:c.sourceHash})),cells};}
const sight=(east=0)=>({east,north:0,forwardE:1,forwardN:0,visible:()=>true});
const tick=()=>new Promise<void>(r=>setTimeout(r,0));

test('quantized real boundaries decode with source normals and exact triangle topology',()=>{
 const {raw,cell}=fixture();const geometry=decodeFacadeBatch(raw,cell.batches[0]);
 assert.deepEqual([...geometry.getAttribute('position').array],[0,0,0,1,0,0,0,1,0]);
 assert.deepEqual([...geometry.index!.array],[0,1,2]);assert.equal(geometry.getAttribute('normal').normalized,true);
 const invalid=raw.slice();new DataView(invalid.buffer).setUint16(36,3,true);assert.throws(()=>decodeFacadeBatch(invalid,cell.batches[0]),/outside vertices/);
 geometry.dispose();
});

test('index rejects source mismatch and inconsistent memory accounting',()=>{
 const {cell}=fixture(),data=index([cell]),source={sourceWorldHash:'source',cells:[{id:cell.id,contentHash:cell.sourceHash}]};
 assert.equal(validateFacadeMiddleIndex(data,source),data);
 assert.throws(()=>validateFacadeMiddleIndex(data,{...source,sourceWorldHash:'other'}));
 const bad=structuredClone(data);bad.cells[0].batches[0].residentBytes++;assert.throws(()=>validateFacadeMiddleIndex(bad,source),/batch/);
});

test('index loading authenticates the hash-pinned metadata before accepting cells',async()=>{
 const {cell}=fixture(),data=index([cell]),source={sourceWorldHash:'source',cells:[{id:cell.id,contentHash:cell.sourceHash}]};
 const bytes=new TextEncoder().encode(JSON.stringify(data));const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('');
 const original=globalThis.fetch;globalThis.fetch=async()=>new Response(bytes);
 try{assert.equal((await loadFacadeMiddleIndex('/compare/neighborhood-facades-'+hash.slice(0,16)+'.json',source)).cells[0].id,cell.id);
  await assert.rejects(()=>loadFacadeMiddleIndex('/compare/neighborhood-facades-'+('0'.repeat(16))+'.json',source),/hash mismatch/);
 }finally{globalThis.fetch=original;}
});

test('bounded gzip reader stages multiple aligned batches and rejects excess or missing payload before attach',async()=>{
 const {cell,raw}=fixture();cell.batches.push({...cell.batches[0],offset:raw.length});cell.byteLength=raw.length*2;
 const original=globalThis.fetch;
 try{
  for(const length of [cell.byteLength,cell.byteLength+1,cell.byteLength-1]){
   const payload=Buffer.concat([raw,raw,Buffer.alloc(1)]).subarray(0,length);
   const gzip=gzipSync(payload);cell.compressedBytes=gzip.length;
   cell.sha256=Buffer.from(await crypto.subtle.digest('SHA-256',gzip)).toString('hex');
   globalThis.fetch=async()=>new Response(new Uint8Array(gzip));
   let prepared=0,disposed=0;
   const controller=createFacadeMiddleController(index([cell]),{scene:new THREE.Group(),yieldTask:async()=>{},prepare:async mesh=>{
    assert.deepEqual([...mesh.geometry.index!.array],[0,1,2]);prepared++;mesh.geometry.addEventListener('dispose',()=>disposed++);
   }});
   controller.update(sight());await controller.waitForIdle();
   if(length===cell.byteLength){assert.deepEqual(controller.status().active,['a']);assert.equal(prepared,2);}
   else {assert.deepEqual(controller.status().active,[]);assert.match(controller.status().lastError,/exceeds|Incomplete/);}
   await controller.dispose();assert.equal(disposed,prepared);
  }
 }finally{globalThis.fetch=original;}
});

test('camera-first geometry stages every batch before one atomic attach',async()=>{
 const near=fixture('near',10),far=fixture('far',100);const scene=new THREE.Group(),loads:string[]=[],attached:string[]=[];
 let continueUpload=()=>{};const upload=new Promise<void>(r=>continueUpload=r);
 const controller=createFacadeMiddleController(index([far.cell,near.cell]),{vertexOverheadBytes:0,scene,load:async cell=>{loads.push(cell.id);return near.raw;},prepare:async()=>upload,onAttach:cell=>attached.push(cell.id),yieldTask:async()=>{},maxCells:1});
 controller.update(sight());await tick();assert.deepEqual(loads,['near']);assert.equal(scene.children.length,0);assert.deepEqual(attached,[]);
 continueUpload();await controller.waitForIdle();assert.deepEqual(attached,['near']);assert.equal(scene.children.length,1);assert.equal(controller.status().bytes,60);
 const group=controller.groupFor('near')!;let disposed=0;(group.children[0] as THREE.Mesh).geometry.addEventListener('dispose',()=>disposed++);
 await controller.dispose();await controller.dispose();assert.equal(disposed,1);assert.equal(scene.children.length,0);
});

test('aborted staged geometry never attaches and disposes each buffer once',async()=>{
 const {cell,raw}=fixture();let finish=()=>{};const pending=new Promise<void>(r=>finish=r);let disposed=0,attached=0;
 const controller=createFacadeMiddleController(index([cell]),{scene:new THREE.Group(),load:async()=>raw,yieldTask:async()=>{},prepare:async mesh=>{mesh.geometry.addEventListener('dispose',()=>disposed++);await pending;},onAttach:()=>attached++});
 controller.update(sight());await tick();controller.update({...sight(),visible:()=>false});finish();await controller.waitForIdle();
 assert.equal(attached,0);assert.equal(disposed,1);assert.equal(controller.status().bytes,0);await controller.dispose();assert.equal(disposed,1);
});

test('outgoing fades retain memory reservations and no replacement exceeds the hard cap',async()=>{
 const a=fixture('a',0),b=fixture('b',200);let releaseFade=()=>{};let fade:Promise<void>|undefined;const loads:string[]=[];
 const controller=createFacadeMiddleController(index([a.cell,b.cell]),{vertexOverheadBytes:0,scene:new THREE.Group(),load:async c=>{loads.push(c.id);return a.raw;},prepare:async()=>{},yieldTask:tick,maxResidentBytes:60,maxCells:1,maxDistanceM:50,
  onDetach:(_c,_g,reason)=>reason==='evict'?(fade??=new Promise<void>(r=>releaseFade=r)):undefined});
 controller.update(sight());await controller.waitForIdle();controller.update(sight(200));await tick();assert.deepEqual(loads,['a']);assert.equal(controller.status().bytes,60);assert.deepEqual(controller.status().retiring,['a']);
 releaseFade();await controller.waitForIdle();assert.deepEqual(loads,['a','b']);assert.equal(controller.status().bytes,60);await controller.dispose();
});

test('a rapid turn back restores the exact retiring facade without another load, upload or reservation',async()=>{
 const a=fixture('a',0),b=fixture('b',200),scene=new THREE.Group();let loads=0,uploads=0,restores=0,disposes=0,cleanups=0,finish=()=>{};let fadeSignal:AbortSignal|undefined;
 const controller=createFacadeMiddleController(index([a.cell,b.cell]),{vertexOverheadBytes:0,scene,maxCells:1,maxResidentBytes:60,maxDistanceM:50,yieldTask:tick,
  load:async()=>{loads++;return a.raw;},prepare:async mesh=>{uploads++;mesh.geometry.addEventListener('dispose',()=>disposes++);},
  onDetach:async(_cell,_group,reason,signal)=>{if(reason==='dispose')return;fadeSignal=signal;await new Promise<void>(resolve=>finish=resolve);if(!signal.aborted)cleanups++;},
  onRestore:(_cell,group)=>{restores++;assert.equal(group,original);}});
 controller.update(sight());await controller.waitForIdle();const original=controller.groupFor('a')!;
 controller.update(sight(200));assert.deepEqual(controller.status().retiring,['a']);assert.equal(controller.groupFor('a'),original);assert.equal(controller.status().bytes,60);assert.equal(controller.status().cells,1);
 controller.update(sight());assert.equal(fadeSignal?.aborted,true);assert.equal(restores,1);assert.equal(controller.groupFor('a'),original);assert.deepEqual(controller.status().active,['a']);assert.deepEqual(controller.status().retiring,[]);
 await controller.waitForIdle();assert.equal(loads,1);assert.equal(uploads,1);assert.equal(controller.status().cells,1);assert.equal(controller.status().bytes,60);assert.equal(scene.children[0],original);
 finish();await tick();assert.equal(cleanups,0);assert.equal(disposes,0);assert.equal(controller.groupFor('a'),original);assert.equal(controller.status().lastError,'');
 await controller.dispose();assert.equal(disposes,1);
});

test('stale fade completion cannot release a newer retirement or same-turn restored resource',async()=>{
 const {cell,raw}=fixture(),phases:{signal:AbortSignal;finish:()=>void}[]=[];let disposes=0,cleanups=0;
 const controller=createFacadeMiddleController(index([cell]),{scene:new THREE.Group(),load:async()=>raw,prepare:async mesh=>{mesh.geometry.addEventListener('dispose',()=>disposes++);},yieldTask:tick,
  onDetach:async(_cell,_group,reason,signal)=>{if(reason==='dispose')return;await new Promise<void>(resolve=>phases.push({signal,finish:resolve}));if(!signal.aborted)cleanups++;},onRestore:()=>{}});
 controller.update(sight());await controller.waitForIdle();const original=controller.groupFor(cell.id);
 controller.update({...sight(),visible:()=>false});controller.update(sight());controller.update({...sight(),visible:()=>false});
 assert.equal(phases.length,2);assert.equal(phases[0].signal.aborted,true);assert.equal(phases[1].signal.aborted,false);
 phases[0].finish();await tick();assert.equal(disposes,0);assert.deepEqual(controller.status().retiring,[cell.id]);assert.equal(controller.groupFor(cell.id),original);assert.equal(controller.status().cells,1);
 phases[1].finish();controller.update(sight());await tick();assert.equal(phases[1].signal.aborted,true);assert.equal(disposes,0);assert.equal(cleanups,0);assert.equal(controller.groupFor(cell.id),original);
 await controller.dispose();assert.equal(disposes,1);
});

test('closing during eviction aborts stale cleanup and releases geometry and shared materials exactly once',async()=>{
 const {cell,raw}=fixture(),scene=new THREE.Group();let finish=()=>{},oldSignal:AbortSignal|undefined,disposes=0,materials=0,oldCleanups=0;
 const controller=createFacadeMiddleController(index([cell]),{scene,load:async()=>raw,prepare:async mesh=>{mesh.geometry.addEventListener('dispose',()=>disposes++);(mesh.material as THREE.Material).addEventListener('dispose',()=>materials++);},yieldTask:tick,
  onDetach:async(_cell,_group,reason,signal)=>{if(reason==='dispose')return;oldSignal=signal;await new Promise<void>(resolve=>finish=resolve);if(!signal.aborted)oldCleanups++;}});
 controller.update(sight());await controller.waitForIdle();controller.update({...sight(),visible:()=>false});
 const closing=controller.dispose();assert.equal(controller.dispose(),closing);await closing;assert.equal(oldSignal?.aborted,true);assert.equal(disposes,1);assert.equal(materials,1);assert.equal(scene.children.length,0);assert.equal(controller.status().cells,0);
 finish();await tick();assert.equal(oldCleanups,0);assert.equal(disposes,1);assert.equal(materials,1);assert.deepEqual(controller.status().retiring,[]);controller.update(sight());assert.equal(controller.status().loading,null);
});

test('older detach callbacks restore through onAttach and presentation waiters see the same prepared group',async()=>{
 const {cell,raw}=fixture();let attached=0,finish=()=>{};
 const controller=createFacadeMiddleController(index([cell]),{scene:new THREE.Group(),load:async()=>raw,prepare:async()=>{},yieldTask:tick,onAttach:()=>attached++,
  onDetach:(_cell,_group,reason)=>reason==='evict'?new Promise<void>(resolve=>finish=resolve):undefined});
 controller.update(sight());await controller.waitForIdle();const original=controller.groupFor(cell.id);controller.update({...sight(),visible:()=>false});controller.update(sight());
 assert.equal(attached,2);assert.equal(controller.groupFor(cell.id),original);assert.equal(await controller.waitForPresentation(cell.id,new AbortController().signal),true);
 finish();await tick();assert.equal(controller.groupFor(cell.id),original);await controller.dispose();
});

test('resident middle facades survive full detail attachment while unopened detailed owners are skipped',async()=>{
 const a=fixture('a'),b=fixture('b',10);const controller=createFacadeMiddleController(index([a.cell,b.cell]),{scene:new THREE.Group(),load:async()=>a.raw,prepare:async()=>{},yieldTask:async()=>{}});
 controller.update({...sight(),detailed:new Set(['b'])});await controller.waitForIdle();assert.deepEqual(controller.status().active,['a']);
 controller.update({...sight(),detailed:new Set(['a','b'])});await controller.waitForIdle();assert.deepEqual(controller.status().active,['a']);await controller.dispose();
});

test('a visible handoff keeps the outgoing owner through the actual upload and cancels on a turn back',async()=>{
 const near=fixture('near',1),replacement=fixture('swap',50);
 let finish=()=>{};const staged=new Promise<void>(resolve=>finish=resolve);
 const controller=createFacadeMiddleController(index([near.cell,replacement.cell]),{scene:new THREE.Group(),maxCells:1,load:async()=>replacement.raw,yieldTask:async()=>{},prepare:async()=>staged});
 controller.update({...sight(),handoffs:new Set(['swap'])});
 assert.equal(controller.status().loading,'swap');
 const cancelled=new AbortController(),pending=new AbortController();
 let published=false;const prepared=controller.waitForPresentation('swap',pending.signal).then(v=>{published=v;});
 const returning=controller.waitForPresentation('swap',cancelled.signal);cancelled.abort();
 assert.equal(await returning,false);await tick();assert.equal(published,false);
 finish();await prepared;assert.equal(published,true);
 assert.equal(await controller.waitForPresentation('near',pending.signal),false);
 await controller.dispose();
});

test('building partitions validate complete parent coverage and inherit parent demand',async()=>{
 const a=fixture('p@front',0),b=fixture('p@back',90);
 for(const [item,id] of [[a,'front'],[b,'back']] as const){item.cell.parentId='p';item.cell.buildingId=id;}
 const data={...index([a.cell,b.cell]),sourceCells:[{id:'p',contentHash:a.cell.sourceHash}]};
 const source={sourceWorldHash:'source',cells:[{id:'p',contentHash:a.cell.sourceHash,stats:{buildingIds:['front','back']}}]};
 validateFacadeMiddleIndex(data,source);
 assert.throws(()=>validateFacadeMiddleIndex({...data,cells:[a.cell]},source),/partition/);
 const controller=createFacadeMiddleController(data,{scene:new THREE.Group(),maxCells:1,maxDistanceM:25,load:async()=>a.raw,prepare:async()=>{},yieldTask:async()=>{}});
 controller.update({...sight(),detailed:new Set(['p'])});await controller.waitForIdle();assert.equal(controller.status().active.length,0);
 controller.update({...sight(),handoffs:new Set(['p'])});await controller.waitForIdle();assert.deepEqual(controller.status().active,['p@front']);
 controller.update({...sight(90),handoffs:new Set(['p'])});await controller.waitForIdle();assert.deepEqual(controller.status().active,['p@back']);
 await controller.dispose();
});
