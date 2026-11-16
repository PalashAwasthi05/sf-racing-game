import * as THREE from 'three';
import {hydrateTreeCell,loadTreeSkeleton,createTreeMaterials,type TreeCell,type TreeCellAsset,type PackedCell,type TreeSkeletonAsset} from './tree-detail-geometry.ts';
import {estimateTreeCell,type TreeEstimate} from './tree-detail-budget.ts';
export {buildTreeCell,loadTreeCell,validateTreeCell,packTreeCell,hydrateTreeCell,type TreeCell,type TreeCellAsset} from './tree-detail-geometry.ts';

type TreeIndex={sourceWorldHash:string;cells:{id:string;contentHash?:string}[]};
type Manifest={version:number;sourceWorldHash:string;cells:{id:string;contentHash:string}[];treeCells:TreeCellAsset[];stats:Record<string,number>;skeleton?:TreeSkeletonAsset};
const hash=async(value:Uint8Array)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value as Uint8Array<ArrayBuffer>)),b=>b.toString(16).padStart(2,'0')).join('');

export function treeDetailBinding(index:TreeIndex){
 return {version:3,sourceWorldHash:index.sourceWorldHash,cells:index.cells.map(c=>({id:c.id,contentHash:c.contentHash??''}))};
}
export async function treeDetailUrl(index:TreeIndex){
 return '/compare/tree-detail-'+(await hash(new TextEncoder().encode(JSON.stringify(treeDetailBinding(index))))).slice(0,16)+'.json';
}

function treeWorker(leafTexture:THREE.Texture,setupMaterial:(material:THREE.Material)=>void,onFailure:()=>void,skeleton:Float32Array){
 const worker=new Worker(new URL('./tree-detail-worker.ts',import.meta.url),{type:'module',name:'source-tree-geometry'});
 try{worker.postMessage({skeleton:skeleton.buffer},[skeleton.buffer as ArrayBuffer]);}catch(error){worker.terminate();throw error;}
 const materials=createTreeMaterials(leafTexture,setupMaterial);
 let serial=0,broken=false;const requests=new Map<number,{resolve:(cell:TreeCell)=>void;reject:(error:Error)=>void}>();
 worker.onmessage=(event:MessageEvent<{id:number;packed?:PackedCell;error?:string}>)=>{
  const response=event.data,pending=requests.get(response.id);if(!pending)return;requests.delete(response.id);
  if(response.error||!response.packed){pending.reject(Error(response.error??'Invalid tree worker response'));return;}
  try{pending.resolve(hydrateTreeCell(response.packed,leafTexture,setupMaterial,materials));}catch(error){pending.reject(error as Error);}
 };
 worker.onerror=()=>{broken=true;for(const pending of requests.values())pending.reject(Error('Tree geometry worker failed'));requests.clear();onFailure();};
 return {load(asset:TreeCellAsset,quality:'fine'|'coarse'){if(broken)return Promise.reject(Error('Tree geometry worker failed'));return new Promise<TreeCell>((resolve,reject)=>{const id=++serial;requests.set(id,{resolve,reject});worker.postMessage({id,asset,quality});});},
  dispose(){worker.terminate();for(const pending of requests.values())pending.reject(Error('Tree geometry worker stopped'));requests.clear();materials.dispose();}};
}

/** One tree-only cell in flight; ready cells replace both base crowns and source
 * vegetation atomically. Distance is to the actual source tree bounds, so detail
 * works everywhere in the world. Original cells remain authoritative for data. */
export async function createTreeDetailLayer(index:TreeIndex,leafTexture:THREE.Texture,
 setupMaterial:(material:THREE.Material)=>void,options:{onChange?:()=>void;loadM?:number;unloadM?:number;fineM?:number;maxCells?:number;maxBytes?:number;maxTriangles?:number;maxDraws?:number;
 prepare?:(cell:TreeCell,signal:AbortSignal)=>Promise<void>;retire?:(cell:TreeCell,signal:AbortSignal)=>Promise<void>;
 restore?:(cell:TreeCell,signal:AbortSignal)=>void|Promise<void>;
 replace?:(incoming:TreeCell,previous:TreeCell,signal:AbortSignal)=>Promise<void>}={}){
 const binding=treeDetailBinding(index),response=await fetch(await treeDetailUrl(index));
 if(!response.ok&&response.status!==404)throw Error('Tree detail package unavailable');
 const manifest:Manifest=response.status===404?{...binding,treeCells:[],stats:{}}:await response.json();
 if(manifest.version!==3||manifest.sourceWorldHash!==binding.sourceWorldHash||JSON.stringify(manifest.cells)!==JSON.stringify(binding.cells)||
  !Array.isArray(manifest.treeCells)||manifest.treeCells.length>binding.cells.length)throw Error('Tree detail binding mismatch');
 const ids=new Set<string>(),coveredClasses=new Map<string,ReadonlySet<string>>();
 for(const asset of manifest.treeCells){
  if(ids.has(asset.id)||!binding.cells.some(c=>c.id===asset.id&&c.contentHash===asset.contentHash)||
   !/^\/compare\/tree-detail-cell-[a-f0-9]{16}\.bin$/.test(asset.binary)||!/^[a-f0-9]{64}$/.test(asset.binaryHash)||
   !Number.isSafeInteger(asset.byteLength)||asset.byteLength<=0||asset.byteLength>16*1024**2||
   !Number.isSafeInteger(asset.compressedBytes)||asset.compressedBytes<=0||asset.compressedBytes>asset.byteLength+65536||
   !Array.isArray(asset.boundsEN)||asset.boundsEN.length!==4||!asset.boundsEN.every(Number.isFinite)||
   asset.boundsEN[0]>asset.boundsEN[2]||asset.boundsEN[1]>asset.boundsEN[3]||!Array.isArray(asset.classes)||
   asset.classes.some(c=>!(/^(leaf\w*|frond|bark|wood|trunk)$/).test(c))||
   !Array.isArray(asset.skeletonIds)||asset.skeletonIds.length>4096||asset.skeletonIds.some(id=>!Number.isSafeInteger(id)||id<0||id>=(manifest.skeleton?.segments??0)))throw Error('Invalid tree cell descriptor');
  ids.add(asset.id);coveredClasses.set(asset.id,new Set(asset.classes));
 }
 const loadM=options.loadM??260,unloadM=options.unloadM??310,fineM=options.fineM??60,maxCells=options.maxCells??64,maxBytes=options.maxBytes??64*1024**2,maxTriangles=options.maxTriangles??400000,maxDraws=options.maxDraws??128;
 type Retirement={cell:TreeCell;abort:AbortController;epoch:number;challenger?:TreeCellAsset};
 type Restoration={cell:TreeCell;abort:AbortController;done:Promise<void>};
 type LodTransition={incoming:TreeCell;previous:TreeCell;abort:AbortController;reversing?:Promise<void>};
 type Deferred={priority:number;x:number;z:number;quality:'fine'|'coarse'};
 const root=new THREE.Group();root.name='tree-detail';const readyCells=new Map<string,TreeCell>(),failed=new Set<string>(),budgetSkipped=new Set<string>(),oversized=new Set<string>(),retiring=new Map<string,Retirement>(),restoring=new Map<string,Restoration>(),staging=new Set<TreeCell>(),overlaps=new Set<TreeCell>(),protectedUntil=new Map<string,number>(),coarseOwners=new Map<string,number>();
 const assets=new Map(manifest.treeCells.map(asset=>[asset.id,asset]));
 const deferred=new Map<string,Deferred>(),desiredCells=new Map<string,'fine'|'coarse'>(),actualCosts=new Map<string,TreeEstimate>(),loadedVariants=new Set<string>();
 const stats={...manifest.stats,readyCells:0,residentBytes:0,loading:false,errors:0,drawCalls:0,triangles:0,pendingBytes:0,pendingTriangles:0,evictions:0,distanceRetirements:0,restoredEvictions:0,reversedLods:0,oversizedVariants:0,overheadBytes:0,fineCells:0,coarseCells:0,sharedMaterials:0,targetCells:0,targetTriangles:0,targetBytes:0,targetDraws:0,targetPlanChanges:0,lodReserveBytes:0,lodReserveTriangles:0,loads:0,uniqueVariants:0,reloads:0};
 const controller=new AbortController();let decoder:ReturnType<typeof treeWorker>|null=null;
 if(manifest.treeCells.length&&typeof Worker==='function')try{
  if(!manifest.skeleton)throw Error('Source tree skeleton missing');
  const skeleton=await loadTreeSkeleton(manifest.skeleton);
  for(const asset of manifest.treeCells)coarseOwners.set(asset.id,new Set(asset.skeletonIds!.map(id=>skeleton[id*8+7])).size);
  stats.overheadBytes=skeleton.byteLength+manifest.treeCells.reduce((n,c)=>n+(c.skeletonIds?.length??0)*8+c.entries.length*80+512,0);
  decoder=treeWorker(leafTexture,setupMaterial,()=>{ids.clear();coveredClasses.clear();options.onChange?.();},skeleton);
  stats.sharedMaterials=3;
 }catch{/* Native/source foliage remains available without workers. */}
 if(!decoder){ids.clear();coveredClasses.clear();manifest.treeCells=[];stats.overheadBytes=0;}
 let disposed=false,visible=true,pending=false,dirty=true,demandEpoch=0,lodTransition:LodTransition|null=null;
 const position={x:Infinity,z:Infinity},rankedPosition={x:Infinity,z:Infinity},look={x:0,z:0};
 const planPosition={x:Infinity,z:Infinity},planLook={x:0,z:0};
 const distance=(asset:TreeCellAsset)=>Math.hypot(Math.max(asset.boundsEN[0]-position.x,0,position.x-asset.boundsEN[2]),Math.max(asset.boundsEN[1]+position.z,0,-position.z-asset.boundsEN[3]));
 const rank=(asset:TreeCellAsset)=>{
  const d=distance(asset),dx=(asset.boundsEN[0]+asset.boundsEN[2])/2-position.x,dz=-(asset.boundsEN[1]+asset.boundsEN[3])/2-position.z;
  const centerM=Math.hypot(dx,dz),facing=(dx*look.x+dz*look.z)/Math.max(1,centerM);return {asset,distance:d,facing,priority:(d+centerM*.16)*(1-.40*facing)};
 };
 const quality=(asset:TreeCellAsset,d:number,previous=readyCells.get(asset.id)?.quality):'fine'|'coarse'=>
  !asset.entries.some(e=>e.kind==='leaves')||d<(previous==='coarse'?fineM-10:previous==='fine'?fineM+25:fineM)?'fine':'coarse';
 const changed=()=>{const residents=[...readyCells.values(),...staging,...overlaps];stats.readyCells=readyCells.size;stats.residentBytes=(disposed?0:stats.overheadBytes)+residents.reduce((n,c)=>n+c.bytes,0);stats.drawCalls=residents.reduce((n,c)=>n+c.renderables.length,0);stats.triangles=residents.reduce((n,c)=>n+c.triangles,0);stats.fineCells=[...readyCells.values()].filter(c=>c.quality==='fine').length;stats.coarseCells=readyCells.size-stats.fineCells;options.onChange?.();};
 stats.residentBytes=stats.overheadBytes;
 const estimates=new Map(manifest.treeCells.map(asset=>[asset.id,{fine:estimateTreeCell(asset,'fine',coarseOwners.get(asset.id)),coarse:estimateTreeCell(asset,'coarse',coarseOwners.get(asset.id))}] as const));
 const variantKey=(id:string,quality:'fine'|'coarse')=>id+':'+quality;
 const cost=(id:string,quality:'fine'|'coarse')=>actualCosts.get(variantKey(id,quality))??estimates.get(id)![quality];
 const fitsAlone=(estimate:TreeEstimate)=>estimate.bytes+stats.overheadBytes<=maxBytes&&estimate.triangles<=maxTriangles&&estimate.draws<=maxDraws&&maxCells>=1;
 const fitsNow=(id:string,estimate:TreeEstimate)=>(readyCells.has(id)||readyCells.size<maxCells)&&stats.residentBytes+estimate.bytes<=maxBytes&&stats.triangles+estimate.triangles<=maxTriangles&&stats.drawCalls+estimate.draws<=maxDraws;
 // The overlap adds the smaller variant on top of max(current,target). Leave
 // that bounded allowance in the global plan instead of repeatedly ejecting
 // cheap cells solely to make a brief fine/coarse transition fit.
 const reserve={bytes:0,triangles:0,draws:0};
 if(options.replace)for(const [id,e] of estimates)if(assets.get(id)!.entries.some(entry=>entry.kind==='leaves')){
  reserve.bytes=Math.max(reserve.bytes,Math.min(e.fine.bytes,e.coarse.bytes));reserve.triangles=Math.max(reserve.triangles,Math.min(e.fine.triangles,e.coarse.triangles));reserve.draws=Math.max(reserve.draws,Math.min(e.fine.draws,e.coarse.draws));
 }
 reserve.bytes=Math.min(reserve.bytes,Math.max(0,maxBytes-stats.overheadBytes)*.12);reserve.triangles=Math.min(reserve.triangles,maxTriangles*.12);reserve.draws=Math.min(reserve.draws,Math.floor(maxDraws*.12));
 stats.lodReserveBytes=Math.ceil(reserve.bytes);stats.lodReserveTriangles=Math.ceil(reserve.triangles);
 // Resolve promptly on abort even when an older caller does not consume the
 // signal. Its late completion still has a rejection handler and no ownership.
 const waitForHook=(promise:Promise<void>|void,signal:AbortSignal)=>new Promise<void>((resolve,reject)=>{
  const cleanup=()=>signal.removeEventListener('abort',aborted),aborted=()=>{cleanup();resolve();};
  if(signal.aborted){Promise.resolve(promise).catch(()=>{});resolve();return;}
  signal.addEventListener('abort',aborted,{once:true});
  Promise.resolve(promise).then(()=>{cleanup();resolve();},error=>{cleanup();reject(error);});
 });
 const restore=(cell:TreeCell)=>{
  restoring.get(cell.id)?.abort.abort();
  const state:Restoration={cell,abort:new AbortController(),done:Promise.resolve()};restoring.set(cell.id,state);
  try{state.done=waitForHook(options.restore?.(cell,state.abort.signal),state.abort.signal).catch(()=>{});}
  catch{state.done=Promise.resolve();}
  void state.done.finally(()=>{if(restoring.get(cell.id)===state)restoring.delete(cell.id);});
  return state.done;
 };
 const retire=(cell:TreeCell,challenger?:TreeCellAsset)=>{
  if(retiring.has(cell.id))return;
  restoring.get(cell.id)?.abort.abort();restoring.delete(cell.id);
  const state:Retirement={cell,abort:new AbortController(),epoch:demandEpoch,challenger};retiring.set(cell.id,state);
  void Promise.resolve().then(()=>{
   if(!state.abort.signal.aborted)return waitForHook(options.retire?.(cell,state.abort.signal),state.abort.signal);
  }).catch(()=>{}).finally(()=>{
   if(disposed||state.abort.signal.aborted||retiring.get(cell.id)!==state)return;
   retiring.delete(cell.id);
   if(readyCells.get(cell.id)===cell){readyCells.delete(cell.id);cell.dispose();budgetSkipped.clear();changed();dirty=true;}
  });
 };
 function renewDemand(){
  for(const [id,state] of retiring){
   if(state.epoch===demandEpoch)continue;
   const asset=assets.get(id);if(!asset)continue;const demand=rank(asset);if(demand.distance>loadM)continue;
   const challenger=state.challenger?rank(state.challenger):null;
   const wanted=!challenger||demand.distance<35||demand.priority<=challenger.priority*1.05+8||demand.facing>.15&&challenger.facing<-.1;
   if(!wanted)continue;
   retiring.delete(id);state.abort.abort();deferred.delete(id);protectedUntil.set(id,performance.now()+450);stats.restoredEvictions++;void restore(state.cell);dirty=true;changed();
  }
  const transition=lodTransition;
  if(transition&&!transition.reversing&&options.restore){
   const asset=assets.get(transition.incoming.id);
   if(asset&&distance(asset)<=loadM&&quality(asset,distance(asset),transition.previous.quality)===transition.previous.quality){
    // Reverse complementary masks while BOTH displays still exist. Merely
    // aborting then disposing incoming would leave uncovered stipple samples.
    readyCells.set(transition.previous.id,transition.previous);overlaps.delete(transition.previous);overlaps.add(transition.incoming);
    transition.reversing=Promise.resolve();transition.reversing=restore(transition.previous);transition.abort.abort();stats.reversedLods++;dirty=true;changed();
   }
  }
 }
 function pump(){
  if(disposed||pending||!visible||!dirty)return;dirty=false;
  if(!Number.isFinite(planPosition.x)||Math.hypot(position.x-planPosition.x,position.z-planPosition.z)>=8||Math.abs(look.x-planLook.x)+Math.abs(look.z-planLook.z)>=.3){planPosition.x=position.x;planPosition.z=position.z;planLook.x=look.x;planLook.z=look.z;}
  const candidates=manifest.treeCells.map(asset=>{
   const candidate=rank(asset),b=asset.boundsEN;
   const d=Math.hypot(Math.max(b[0]-planPosition.x,0,planPosition.x-b[2]),Math.max(b[1]+planPosition.z,0,-planPosition.z-b[3]));
   const dx=(b[0]+b[2])/2-planPosition.x,dz=-(b[1]+b[3])/2-planPosition.z,center=Math.hypot(dx,dz),facing=(dx*planLook.x+dz*planLook.z)/Math.max(1,center);
   const incumbent=readyCells.has(asset.id)||desiredCells.has(asset.id),protectedCell=performance.now()<(protectedUntil.get(asset.id)??0);
   return {...candidate,score:Math.max(0,(d+center*.16)*(1-.4*facing)*(protectedCell ? .10 : incumbent ? .86 : 1))};
  }).sort((a,b)=>a.score-b.score||Number(readyCells.has(b.asset.id))-Number(readyCells.has(a.asset.id))||a.asset.id.localeCompare(b.asset.id));
  for(const {asset,distance:d} of candidates)if(d>unloadM&&readyCells.has(asset.id)&&!retiring.has(asset.id)){
   stats.distanceRetirements++;retire(readyCells.get(asset.id)!);
  }
  const demands=candidates.flatMap(candidate=>{
   const {asset,distance:d}=candidate,previous=readyCells.get(asset.id);
   if(!ids.has(asset.id)||d>(previous?unloadM:loadM)||failed.has(asset.id))return [];
   let desired=previous&&!options.replace?previous.quality:quality(asset,d),estimate=cost(asset.id,desired);
   if(!fitsAlone(estimate)){
    oversized.add(variantKey(asset.id,desired));stats.oversizedVariants=oversized.size;
    // An impossible fine variant must not churn other residents. A bounded
    // coarse display can still cover this source until fine can be supported
    // under a different configured hard ceiling; normal fine demand is kept.
    if(desired!=='fine')return [];
    desired='coarse';estimate=cost(asset.id,'coarse');if(!fitsAlone(estimate))return [];
   }
   if(budgetSkipped.has(variantKey(asset.id,desired)))return [];
   const held=deferred.get(asset.id);
   if(held){
    const moved=Math.hypot(position.x-held.x,position.z-held.z);
    const improved=d<35||candidate.priority<held.priority*.78||moved>=24&&candidate.priority<=held.priority*1.05||desired==='fine'&&held.quality==='coarse';
    if(!improved)return [];deferred.delete(asset.id);
   }
   const peak={bytes:Math.max(estimate.bytes,previous?.bytes??0),triangles:Math.max(estimate.triangles,previous?.triangles??0),draws:Math.max(estimate.draws,previous?.renderables.length??0)};
   return [{...candidate,quality:desired,estimate,peak}];
  });
  let plannedBytes=stats.overheadBytes,plannedTriangles=0,plannedDraws=0;
  const selected:typeof demands=[];
  for(const demand of demands){
   if(selected.length>=maxCells||plannedBytes+demand.peak.bytes>maxBytes-reserve.bytes||plannedTriangles+demand.peak.triangles>maxTriangles-reserve.triangles||plannedDraws+demand.peak.draws>maxDraws-reserve.draws)continue;
   selected.push(demand);plannedBytes+=demand.peak.bytes;plannedTriangles+=demand.peak.triangles;plannedDraws+=demand.peak.draws;
  }
  const signature=(values:Iterable<[string,'fine'|'coarse']>)=>[...values].map(([id,q])=>id+':'+q).join('|');
  const previousPlan=signature(desiredCells);desiredCells.clear();for(const demand of selected)desiredCells.set(demand.asset.id,demand.quality);
  if(previousPlan!==signature(desiredCells))stats.targetPlanChanges++;
  stats.targetCells=selected.length;stats.targetBytes=plannedBytes;stats.targetTriangles=plannedTriangles;stats.targetDraws=plannedDraws;
  const challenger=selected.find(d=>!readyCells.has(d.asset.id)||readyCells.get(d.asset.id)!.quality!==d.quality)?.asset;
  for(const candidate of candidates){
   const cell=readyCells.get(candidate.asset.id);
   if(!cell||desiredCells.has(cell.id)||retiring.has(cell.id)||candidate.distance>unloadM)continue;
   // Budget victims do not re-enter merely because their fade freed a cheap
   // slot. Admission must improve materially in distance/view direction.
   deferred.set(cell.id,{priority:candidate.priority,x:position.x,z:position.z,quality:cell.quality});
   stats.evictions++;retire(cell,challenger);
  }
  // A smaller member can warm while another selected member awaits an old
  // fade, but a non-member can never opportunistically steal that space.
  const next=selected.find(d=>!retiring.has(d.asset.id)&&(!readyCells.has(d.asset.id)||readyCells.get(d.asset.id)!.quality!==d.quality)&&fitsNow(d.asset.id,d.estimate));
  if(!next)return;
  const nextQuality=next.quality,estimate=next.estimate;
  pending=true;stats.loading=true;stats.pendingBytes=estimate.bytes;stats.pendingTriangles=estimate.triangles;
  void decoder!.load(next.asset,nextQuality).then(async cell=>{
   if(disposed){cell.dispose();return;}
   const key=variantKey(cell.id,cell.quality);stats.loads++;if(loadedVariants.has(key))stats.reloads++;else loadedVariants.add(key);stats.uniqueVariants=loadedVariants.size;
   actualCosts.set(key,{bytes:cell.bytes,geometryBytes:cell.geometryBytes,vertices:cell.renderables.reduce((n,m)=>n+m.geometry.getAttribute('position').count,0),triangles:cell.triangles,draws:cell.renderables.length,leafCards:cell.foliageMeshes.reduce((n,m)=>n+m.geometry.getAttribute('position').count/4,0),twigSegments:cell.twigSegments});
   if(disposed||distance(next.asset)>unloadM||stats.residentBytes+cell.bytes>maxBytes||stats.triangles+cell.triangles>maxTriangles||stats.drawCalls+cell.renderables.length>maxDraws){cell.dispose();if(!disposed)budgetSkipped.add(variantKey(cell.id,cell.quality));return;}
   stats.pendingBytes=0;stats.pendingTriangles=0;
   staging.add(cell);changed();
   try{await options.prepare?.(cell,controller.signal);controller.signal.throwIfAborted();}
   catch(error){staging.delete(cell);cell.dispose();changed();throw error;}
   const previous=readyCells.get(cell.id);
   // Camera demand can change while geometry is being warmed. Keep the valid
   // previous display rather than starting a now-unwanted LOD transition.
   if(distance(next.asset)>unloadM||previous&&quality(next.asset,distance(next.asset),previous.quality)!==cell.quality){staging.delete(cell);cell.dispose();changed();return;}
   staging.delete(cell);cell.root.visible=visible;root.add(cell.root);readyCells.set(cell.id,cell);
   const transition:LodTransition|null=previous?{incoming:cell,previous,abort:new AbortController()}:null;
   if(transition){overlaps.add(transition.previous);lodTransition=transition;}changed();
   if(transition){
    // Caller supplies complementary 200–300ms masks. Keep the old display
    // resident until the blend resolves, including every overlapping byte,
    // draw and triangle; the ready source-cell identity never disappears.
    try{if(!transition.abort.signal.aborted)await waitForHook(options.replace?.(cell,transition.previous,transition.abort.signal),transition.abort.signal);}catch{/* Keep the valid display if a fade is interrupted. */}
    if(transition.reversing){await transition.reversing;overlaps.delete(cell);cell.dispose();}
    else{overlaps.delete(transition.previous);transition.previous.dispose();}
    if(lodTransition===transition)lodTransition=null;changed();
   }
  }).catch(()=>{if(!disposed){failed.add(next.asset.id);ids.delete(next.asset.id);coveredClasses.delete(next.asset.id);stats.errors++;options.onChange?.();}}).finally(()=>{pending=false;stats.loading=false;stats.pendingBytes=0;stats.pendingTriangles=0;dirty=true;});
 }
 return {root,readyCells,desiredCells,availableCells:ids,coveredClasses,stats,update(p:Pick<THREE.Vector3,'x'|'z'>,direction?:Pick<THREE.Vector3,'x'|'z'>){
   position.x=p.x;position.z=p.z;
   let renewed=false;
   if(Math.hypot(p.x-rankedPosition.x,p.z-rankedPosition.z)>2){rankedPosition.x=p.x;rankedPosition.z=p.z;dirty=true;budgetSkipped.clear();renewed=true;}
   if(direction&&Math.abs(direction.x-look.x)+Math.abs(direction.z-look.z)>.15){look.x=direction.x;look.z=direction.z;dirty=true;renewed=true;}
   if(renewed)demandEpoch++;renewDemand();
   pump();},
  setLayer(on:boolean){visible=on;root.visible=on;},
  dispose(){disposed=true;controller.abort();lodTransition?.abort.abort();for(const state of retiring.values())state.abort.abort();for(const state of restoring.values())state.abort.abort();retiring.clear();restoring.clear();decoder?.dispose();for(const cell of new Set([...readyCells.values(),...staging,...overlaps]))cell.dispose();readyCells.clear();staging.clear();overlaps.clear();desiredCells.clear();actualCosts.clear();deferred.clear();protectedUntil.clear();stats.targetCells=stats.targetTriangles=stats.targetBytes=stats.targetDraws=0;stats.sharedMaterials=0;root.removeFromParent();root.clear();changed();}};
}
