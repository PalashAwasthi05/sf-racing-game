import * as THREE from 'three';
import {finishStaticSteps,finishStaticStepsAsync} from './static-detail-geometry.ts';

export type FacadeBatch = {offset:number;vertices:number;indices:number;min:number[];step:number[];glazing:boolean;residentBytes:number;quantizationErrorM:number};
export type FacadeCell = {id:string;parentId?:string;buildingId?:string;boundsEN:number[];tileEN:number[];sourceHash:string;url:string;sha256:string;compressedBytes:number;byteLength:number;batches:FacadeBatch[];
 stats:{triangles:number;vertices:number;residentBytes:number;batches:number;buildings:number;sourceGlazingTriangles:number;glazingTriangles:number}};
export type FacadeMiddleIndex = {version:1;format:'facade-middle-v1';sourceWorldHash:string;complete:boolean;sourceCells:{id:string;contentHash:string}[];cells:FacadeCell[]};
type SourceIndex = {sourceWorldHash:string;cells:{id:string;contentHash?:string;stats?:{buildingIds?:string[]}}[]};
const MIB=1024**2;
const aborted=()=>new DOMException('Facade preparation cancelled','AbortError');
function check(signal:AbortSignal){if(signal.aborted)throw aborted();}

export function validateFacadeMiddleIndex(index:FacadeMiddleIndex,source:SourceIndex){
 const sourceCells=new Map(source.cells.map(c=>[c.id,c.contentHash]));
 if(index?.version!==1||index.format!=='facade-middle-v1'||!index.complete||index.sourceWorldHash!==source.sourceWorldHash||
  index.sourceCells.length!==sourceCells.size||index.sourceCells.some(c=>!c.contentHash||sourceCells.get(c.id)!==c.contentHash)||
  new Set(index.sourceCells.map(c=>c.id)).size!==sourceCells.size||new Set(index.cells.map(c=>c.id)).size!==index.cells.length)throw Error('Facade geometry belongs to a different neighborhood');
 for(const c of index.cells){
  if(sourceCells.get(c.parentId??c.id)!==c.sourceHash||!/^\/compare\/facade-cell-[a-f0-9]{16}\.bin\.gz$/.test(c.url)||!/^[a-f0-9]{64}$/.test(c.sha256)||
   !Number.isSafeInteger(c.byteLength)||c.byteLength<=0||c.byteLength>32*MIB||!Number.isSafeInteger(c.compressedBytes)||c.compressedBytes<=0||c.compressedBytes>16*MIB||
   c.boundsEN.length!==4||!c.boundsEN.every(Number.isFinite)||!c.batches.length||c.stats.glazingTriangles!==c.stats.sourceGlazingTriangles)throw Error('Invalid facade cell');
  let end=0,vertices=0,indices=0,resident=0;
  for(const b of c.batches){
   if(!Number.isSafeInteger(b.offset)||b.offset!==end||!Number.isSafeInteger(b.vertices)||b.vertices<=0||b.vertices>60000||
    !Number.isSafeInteger(b.indices)||b.indices<=0||b.indices%3||b.min.length!==3||b.step.length!==3||!b.min.every(Number.isFinite)||!b.step.every(s=>Number.isFinite(s)&&s>0)||
    b.residentBytes!==b.vertices*18+b.indices*2)throw Error('Invalid facade batch');
   end=Math.ceil((b.offset+b.vertices*12)/4)*4+b.indices*2;end=Math.ceil(end/4)*4;
   vertices+=b.vertices;indices+=b.indices;resident+=b.residentBytes;
  }
  if(end!==c.byteLength||vertices!==c.stats.vertices||indices/3!==c.stats.triangles||resident!==c.stats.residentBytes||c.batches.length!==c.stats.batches)throw Error('Facade accounting mismatch');
 }
 const parents=new Set(index.cells.filter(c=>c.parentId!==undefined).map(c=>c.parentId!));
 for(const parent of parents){
  const parts=index.cells.filter(c=>c.parentId===parent);
  const ids=source.cells.find(c=>c.id===parent)?.stats?.buildingIds;
  if(!ids||index.cells.some(c=>c.id===parent)||parts.length!==ids.length||
    parts.some(c=>!c.buildingId||!ids.includes(c.buildingId)||c.id!==parent+'@'+c.buildingId||c.stats.buildings!==1))throw Error('Incomplete middle facade partition');
 }
 return index;
}

export async function loadFacadeMiddleIndex(url:string,source:SourceIndex,signal?:AbortSignal){
 if(!/^\/compare\/neighborhood-facades-[a-f0-9]{16}\.json$/.test(url))throw Error('Unpinned facade index');
 const response=await fetch(url,{signal});if(!response.ok)throw Error('Facade index unavailable');
 const bytes=await boundedBytes(response.body!,4*MIB,signal);
 const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes as Uint8Array<ArrayBuffer>))].map(v=>v.toString(16).padStart(2,'0')).join('');
 if(hash.slice(0,16)!==url.slice(-21,-5))throw Error('Facade index hash mismatch');
 return validateFacadeMiddleIndex(JSON.parse(new TextDecoder().decode(bytes)),source);
}

async function boundedBytes(stream:ReadableStream<Uint8Array>,limit:number,signal?:AbortSignal,exact=false){
 const reader=stream.getReader();const parts:Uint8Array[]=[];let total=0;
 const cancel=()=>{void reader.cancel(aborted()).catch(()=>{});};signal?.addEventListener('abort',cancel,{once:true});
 try{
  if(signal?.aborted)throw aborted();
  for(;;){const {value,done}=await reader.read();if(done)break;if(signal?.aborted)throw aborted();total+=value.length;if(total>limit)throw Error('Facade data exceeds declared length');parts.push(value);}
  if(signal?.aborted)throw aborted();if(exact&&total!==limit)throw Error('Incomplete facade geometry');
  const result=new Uint8Array(total);let offset=0;for(const part of parts){result.set(part,offset);offset+=part.length;}return result;
 }finally{signal?.removeEventListener('abort',cancel);await reader.cancel().catch(()=>{});reader.releaseLock();}
}

// Feed the inflater on demand. Limiting compressed input per pull also bounds
// the size of an inflated chunk held while a mesh waits for its GPU upload.
const compressedChunkBytes=512;
const inflaterWorkingBytes=1024*1024;
function batchLength(cell:FacadeCell,i:number){
 return (cell.batches[i+1]?.offset??cell.byteLength)-cell.batches[i].offset;
}
async function* verifiedBatches(cell:FacadeCell,signal:AbortSignal){
 const response=await fetch(cell.url,{signal});if(!response.ok||!response.body)throw Error('Facade geometry unavailable');
 const compressed=await boundedBytes(response.body,cell.compressedBytes,signal,true);check(signal);
 const digest=await crypto.subtle.digest('SHA-256',compressed);
 const hash=Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
 if(hash!==cell.sha256)throw Error('Facade geometry hash mismatch');check(signal);
 let offset=0;
 const compressedStream=new ReadableStream<Uint8Array<ArrayBuffer>>({pull(controller){
   if(offset===compressed.length){controller.close();return;}
   const part=compressed.subarray(offset,offset+compressedChunkBytes);
   controller.enqueue(part);offset+=part.length;
 }});
 const reader=compressedStream.pipeThrough(new DecompressionStream('gzip')).getReader();
 let chunk:Uint8Array<ArrayBufferLike>=new Uint8Array(0);
 try{
   for(let i=0;i<cell.batches.length;i++){
     const raw=new Uint8Array(batchLength(cell,i));
     for(let written=0;written<raw.length;){
       check(signal);
       if(!chunk.length){const next=await reader.read();if(next.done)throw Error('Incomplete facade geometry');chunk=next.value;}
       const count=Math.min(chunk.length,raw.length-written);
       raw.set(chunk.subarray(0,count),written);chunk=chunk.subarray(count);written+=count;
     }
     yield {raw,batch:{...cell.batches[i],offset:0}};
   }
   if(chunk.length||!(await reader.read()).done)throw Error('Facade data exceeds declared length');
 }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
}

function* facadeBatchSteps(raw:Uint8Array,b:FacadeBatch):Generator<void,THREE.BufferGeometry>{
 const view=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);
 const positions=new Float32Array(b.vertices*3);
 const box=new THREE.Box3();
 for(let first=0;first<b.vertices;first+=2048){
   const end=Math.min(b.vertices,first+2048);
   for(let i=first;i<end;i++){
     const at=b.offset+6*i,j=3*i;
     const x=positions[j]=Math.fround(b.min[0]+view.getUint16(at,true)*b.step[0]);
     const y=positions[j+1]=Math.fround(b.min[1]+view.getUint16(at+2,true)*b.step[1]);
     const z=positions[j+2]=Math.fround(b.min[2]+view.getUint16(at+4,true)*b.step[2]);
     box.min.x=Math.min(box.min.x,x);box.min.y=Math.min(box.min.y,y);box.min.z=Math.min(box.min.z,z);
     box.max.x=Math.max(box.max.x,x);box.max.y=Math.max(box.max.y,y);box.max.z=Math.max(box.max.z,z);
   }
   yield;
 }
 const normalsAt=b.offset+6*b.vertices,colorsAt=normalsAt+3*b.vertices;
 const normals=new Int8Array(raw.buffer,raw.byteOffset+normalsAt,b.vertices*3).slice();
 const colors=raw.slice(colorsAt,colorsAt+3*b.vertices);
 const at=Math.ceil((colorsAt+3*b.vertices)/4)*4;
 const indices=new Uint16Array(b.indices);
 for(let first=0;first<b.indices;first+=8192){
   const end=Math.min(b.indices,first+8192);
   for(let i=first;i<end;i++){const v=view.getUint16(at+i*2,true);if(v>=b.vertices)throw Error('Facade index outside vertices');indices[i]=v;}
   yield;
 }
 const geometry=new THREE.BufferGeometry();
 geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3,true));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3,true));
 geometry.setIndex(new THREE.BufferAttribute(indices,1));
 geometry.boundingBox=box;geometry.boundingSphere=box.getBoundingSphere(new THREE.Sphere());
 return geometry;
}

export function decodeFacadeBatch(raw:Uint8Array,b:FacadeBatch){return finishStaticSteps(facadeBatchSteps(raw,b));}

export type FacadeSight={east:number;north:number;forwardE:number;forwardN:number;visible:(boundsEN:number[])=>boolean;detailed?:ReadonlySet<string>;handoffs?:ReadonlySet<string>;
 /** Optional normalized projected frontage demand; larger values load first. */
 screenDemand?:(cell:FacadeCell)=>number};
export type FacadeMiddleOptions={scene:THREE.Object3D;
 /** Stage one bounded mesh's real GPU buffers/program before publishing it. */
 prepare:(mesh:THREE.Mesh,signal:AbortSignal)=>Promise<void>;
 onAttach?:(cell:FacadeCell,group:THREE.Group)=>void;
 /** Resolve after an outgoing transition. Check signal.aborted after awaiting
  * before external cleanup: renewed demand cancels that eviction in place. */
 onDetach?:(cell:FacadeCell,group:THREE.Group,reason:'evict'|'dispose',signal:AbortSignal)=>void|Promise<void>;
 /** Reverse a cancelled outgoing transition using the same prepared group.
  * Older callers fall back to onAttach; no fetch, upload or budget is added. */
 onRestore?:(cell:FacadeCell,group:THREE.Group)=>void;
 maxResidentBytes?:number;maxWorkingBytes?:number;maxTriangles?:number;maxDraws?:number;maxCells?:number;maxDistanceM?:number;
 /** Reserve transition/other attributes installed by prepare, per vertex.
  * Defaults to the streamTier integer/float layout for this source cell count. */
 vertexOverheadBytes?:number;
 yieldTask?:()=>Promise<void>;
 load?:(cell:FacadeCell,signal:AbortSignal)=>Promise<Uint8Array>;
};

export function createFacadeMiddleController(index:FacadeMiddleIndex,options:FacadeMiddleOptions){
 const caps={bytes:options.maxResidentBytes??16*MIB,workingBytes:options.maxWorkingBytes??6*MIB,triangles:options.maxTriangles??300000,draws:options.maxDraws??40,cells:options.maxCells??16,distance:options.maxDistanceM??225};
 const vertexOverhead=options.vertexOverheadBytes??(index.sourceCells.length+index.cells.filter(c=>c.parentId).length<4096?2:4);
 const cellBytes=(cell:FacadeCell)=>cell.stats.residentBytes+cell.stats.vertices*vertexOverhead;
 const workingBytes=(cell:FacadeCell)=>options.load?2*(cell.compressedBytes+cell.byteLength):
  2*cell.compressedBytes+Math.max(...cell.batches.map((_,i)=>batchLength(cell,i)))+inflaterWorkingBytes;
 type Resource={cell:FacadeCell;group:THREE.Group;disposed:boolean};
 type Retirement={abort:AbortController;reason:'evict'|'dispose';done:Promise<void>};
 // Keep one resident entry throughout retirement, so reentry can restore the
 // existing object before the outgoing transition has completed.
 const live=new Map<string,Resource>(),retiring=new Map<Resource,Retirement>(),retry=new Map<string,number>();
 let desired:FacadeCell[]=[],closed=false,lastError='',work:{cell:FacadeCell;abort:AbortController;done:Promise<void>}|undefined;
 let disposal:Promise<void>|undefined;
 const waiting=new Set<()=>void>();
 function notify(){for(const check of waiting)check();}
 function waitForPresentation(id:string,signal:AbortSignal){
  return new Promise<boolean>(resolve=>{
   const check=()=>{
    const resource=live.get(id),ready=!!resource&&!retiring.has(resource),eligible=desired.some(c=>c.id===id)&&(retry.get(id)??0)<=Date.now();
    if(!signal.aborted&&!closed&&!ready&&eligible)return;
    waiting.delete(check);signal.removeEventListener('abort',check);resolve(ready&&!signal.aborted&&!closed);
   };
   waiting.add(check);signal.addEventListener('abort',check,{once:true});check();
  });
 }
 const matte=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.85,metalness:0,side:THREE.DoubleSide});
 // Same dielectric response as the viewer's full-detail glazing finish. Source
 // linear glass tints remain dark; sky/probe reflection supplies their sheen.
 const glass=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.08,metalness:0,envMapIntensity:1.3,side:THREE.DoubleSide});
 matte.name='facade_middle_solid';glass.name='facade_middle_glazing';
 for(const material of [matte,glass])material.userData={class:'facade_middle',bakedAppearance:true,layer:'facades'};
 const yieldTask=options.yieldTask??(()=>new Promise<void>(r=>setTimeout(r,0)));
 const usage=()=>{
  const cells=[...live.values()].map(r=>r.cell);if(work&&!live.has(work.cell.id))cells.push(work.cell);
  return {bytes:cells.reduce((n,c)=>n+cellBytes(c),0),workingBytes:work?workingBytes(work.cell):0,
   triangles:cells.reduce((n,c)=>n+c.stats.triangles,0),draws:cells.reduce((n,c)=>n+c.batches.length,0),cells:cells.length};
 };
 function release(r:Resource){if(r.disposed)return;r.disposed=true;r.group.removeFromParent();r.group.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});r.group.clear();}
 function retire(r:Resource,reason:'evict'|'dispose'){
  if(r.disposed)return Promise.resolve();
  const previous=retiring.get(r);if(previous?.reason===reason)return previous.done;
  if(previous){retiring.delete(r);previous.abort.abort();}
  const retirement:Retirement={abort:new AbortController(),reason,done:Promise.resolve()};retiring.set(r,retirement);
  const signal=retirement.abort.signal;let finishAbort=()=>{};
  const cancelled=new Promise<void>(resolve=>{finishAbort=resolve;signal.addEventListener('abort',finishAbort,{once:true});});
  let transition:Promise<void>;
  try{transition=Promise.resolve(options.onDetach?.(r.cell,r.group,reason,signal));}catch(e){lastError=String(e);transition=Promise.resolve();}
  retirement.done=Promise.race([transition.catch(e=>{if(!signal.aborted)lastError=String(e);}),cancelled]).finally(()=>{
   signal.removeEventListener('abort',finishAbort);
   if(retiring.get(r)!==retirement||signal.aborted)return;
   retiring.delete(r);if(live.get(r.cell.id)===r)live.delete(r.cell.id);release(r);notify();schedule();
  });
  return retirement.done;
 }
 function restore(r:Resource){
  const retirement=retiring.get(r);if(!retirement||retirement.reason!=='evict'||closed||r.disposed)return;
  retiring.delete(r);retirement.abort.abort();options.scene.add(r.group);
  try{(options.onRestore??options.onAttach)?.(r.cell,r.group);}catch(error){lastError=String(error);}
 }
 async function* cellBatches(cell:FacadeCell,signal:AbortSignal){
  if(!options.load){yield* verifiedBatches(cell,signal);return;}
  const raw=await options.load(cell,signal);
  if(raw.byteLength!==cell.byteLength)throw Error('Facade length mismatch');
  for(const batch of cell.batches)yield {raw,batch};
 }
 async function load(cell:FacadeCell,signal:AbortSignal){
  const group=new THREE.Group();group.name='facade_middle__'+cell.id;group.userData={streamCell:cell.id,facadeMiddle:true,displayOnly:true};
  const resource:Resource={cell,group,disposed:false};
  try{
   for await(const {raw,batch} of cellBatches(cell,signal)){
    check(signal);await yieldTask();check(signal);
    const geometry=await finishStaticStepsAsync(facadeBatchSteps(raw,batch),async()=>{await yieldTask();check(signal);});
    const mesh=new THREE.Mesh(geometry,batch.glazing?glass:matte);mesh.name=group.name+(batch.glazing?'__glazing':'__solid');
    mesh.userData={...group.userData,layer:'facades',class:'facade_middle',bakedAppearance:true};mesh.castShadow=!batch.glazing;mesh.receiveShadow=true;group.add(mesh);
    await options.prepare(mesh,signal);check(signal);
   }
   return resource;
  }catch(error){release(resource);throw error;}
 }
 function schedule(){
  if(closed||work)return;
  const candidate=desired.find(c=>!live.has(c.id)&&(retry.get(c.id)??0)<=Date.now());if(!candidate)return;
  const used=usage();if(used.bytes+cellBytes(candidate)>caps.bytes||used.triangles+candidate.stats.triangles>caps.triangles||used.draws+candidate.batches.length>caps.draws||used.cells>=caps.cells)return;
  const task={cell:candidate,abort:new AbortController(),done:Promise.resolve()};work=task;
  task.done=load(candidate,task.abort.signal).then(resource=>{
   if(closed||task.abort.signal.aborted||!desired.some(c=>c.id===candidate.id)){release(resource);return;}
   live.set(candidate.id,resource);options.scene.add(resource.group);
   try{options.onAttach?.(candidate,resource.group);}catch(error){live.delete(candidate.id);release(resource);throw error;}
   retry.delete(candidate.id);
  }).catch(error=>{if(!task.abort.signal.aborted&&!closed){lastError=String(error);retry.set(candidate.id,Date.now()+5000);}}).finally(()=>{if(work===task)work=undefined;notify();schedule();});
 }
 return {
  update(sight:FacadeSight){
   if(closed)return;
   const direction=Math.hypot(sight.forwardE,sight.forwardN)||1;
   const scored=index.cells.flatMap(cell=>{
    const b=cell.boundsEN,dx=Math.max(b[0]-sight.east,0,sight.east-b[2]),dn=Math.max(b[1]-sight.north,0,sight.north-b[3]),distance=Math.hypot(dx,dn);
    if(distance>caps.distance||workingBytes(cell)>caps.workingBytes||!sight.visible(b)||(!live.has(cell.id)&&sight.detailed?.has(cell.parentId??cell.id)))return [];
    const e=(b[0]+b[2])*.5-sight.east,n=(b[1]+b[3])*.5-sight.north;
    const facing=(e*sight.forwardE+n*sight.forwardN)/(Math.hypot(e,n)*direction||1);
    const demand=sight.screenDemand?.(cell)??0;
    return [{cell,score:distance*(1-.4*Math.max(0,facing))-Math.min(200,Math.max(0,demand)*100)+(sight.detailed?.has(cell.parentId??cell.id)?35:0)-(live.has(cell.id)?12:0)}];
   }).sort((a,b)=>Number(!!sight.handoffs?.has(b.cell.parentId??b.cell.id))-Number(!!sight.handoffs?.has(a.cell.parentId??a.cell.id))||a.score-b.score||a.cell.id.localeCompare(b.cell.id));
   const selected:FacadeCell[]=[];let bytes=0,triangles=0,draws=0;
   for(const {cell} of scored){const s=cell.stats;if(bytes+cellBytes(cell)>caps.bytes||triangles+s.triangles>caps.triangles||draws+cell.batches.length>caps.draws||selected.length>=caps.cells)continue;
    selected.push(cell);bytes+=cellBytes(cell);triangles+=s.triangles;draws+=cell.batches.length;}
   desired=selected;
   if(work&&!desired.some(c=>c.id===work!.cell.id))work.abort.abort();
   for(const r of [...live.values()]){if(desired.some(c=>c.id===r.cell.id))restore(r);else void retire(r,'evict');}
   notify();schedule();
  },
  waitForPresentation,
  groupFor:(id:string)=>live.get(id)?.group,
  status:()=>({active:[...live.values()].filter(r=>!retiring.has(r)).map(r=>r.cell.id),retiring:[...retiring.keys()].map(r=>r.cell.id),wanted:desired.map(c=>c.id),loading:work?.cell.id??null,...usage(),caps,lastError}),
  async waitForIdle(){while(work||retiring.size){if(work)await work.done;else await yieldTask();}},
  dispose(){
   if(disposal)return disposal;closed=true;desired=[];notify();work?.abort.abort();
   disposal=(async()=>{await work?.done;await Promise.all([...live.values()].map(r=>retire(r,'dispose')));matte.dispose();glass.dispose();})();
   return disposal;
  },
 };
}
