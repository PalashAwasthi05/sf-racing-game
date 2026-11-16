// Deterministic CPU simulation: actual planner/transition code and rounded circuit geometry.
// No browser, geometry decoding, GPU, car physics, rendering or FPS measurement.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {setImmediate as turn} from 'node:timers/promises';
import {manageDriveCells} from '../king-street/app/drive-stream.ts';
import {createStreamTransitions,coverageRange,drawTier} from '../king-street/app/stream-transition.ts';
import {treeDetailUrl} from '../king-street/app/tree-detail.ts';
const root=path.resolve(import.meta.dirname,'../..');
const destination=process.argv[2]??path.join(root,'agent-handoff/reviews/2026-11-12/clarity');
const json=async p=>JSON.parse(await readFile(p,'utf8'));
const fixture=await json(path.join(root,'agent-handoff/reviews/2026-11-12/full-map-1080/route.json'));
const manifest=await json(path.join(root,'work/king-street/public/compare/neighborhood-stream.json'));
const latencies=await json(path.join(destination,'arrival-latency-fixture.json'));
const publicPath=url=>path.join(root,'work/king-street/public',url);
const facadeManifest=await json(publicPath(manifest.facadeMiddle));
const treeManifest=await json(publicPath(await treeDetailUrl(manifest)));
const facadeIds=new Set(facadeManifest.cells.map(c=>c.id)),treeIds=new Set(treeManifest.treeCells.map(c=>c.id));
assert.equal(manifest.sourceWorldHash,fixture.sourceWorldHash);
assert.equal(facadeManifest.sourceWorldHash,manifest.sourceWorldHash);assert.equal(treeManifest.sourceWorldHash,manifest.sourceWorldHash);
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const toward=(a,b,m)=>{const t=m/distance(a,b);return a.map((v,i)=>v+t*(b[i]-v));};
// Same 11m quadratic corners and 1.4m sampling as neighborhood-circuit.mjs.
const way=[{p:fixture.centerlineEN[0],corner:false}],turns=[];
function line(a,b,corner=false){const d=distance(a,b),n=Math.ceil(d/1.4);for(let i=1;i<=n;i++)way.push({p:toward(a,b,d*i/n),corner});}
let prev=fixture.centerlineEN[0];
for(let i=1;i<fixture.centerlineEN.length-1;i++){
 const point=fixture.centerlineEN[i],a=toward(point,fixture.centerlineEN[i-1],fixture.cornerRadiusM),b=toward(point,fixture.centerlineEN[i+1],fixture.cornerRadiusM);
 line(prev,a);let last=a;for(let j=1;j<=14;j++){const t=j/14,s=1-t,q=a.map((v,k)=>s*s*v+2*s*t*point[k]+t*t*b[k]);line(last,q,true);last=q;}
 turns.push(way.length-1);prev=b;
}
line(prev,fixture.centerlineEN.at(-1));
const stations=way.map((w,i)=>i?distance(w.p,way[i-1].p):0);for(let i=1;i<stations.length;i++)stations[i]+=stations[i-1];
const plannedM=stations.at(-1);assert.ok(Math.abs(plannedM-960)<10,`Planned distance ${plannedM}`);assert.equal(turns.length,4);

async function replay(name,stress=false,supplement=true,reentry=false){
 let now=0,closed=false,serial=0,frame=0,mileage=0,cursor=0,checks=0,maxDelta=0,maxActive=0,maxRetiring=0,maxResidentBytes=0,maxFacades=0,maxTrees=0,retireCalls=0,restores=0,disposals=0,hazards=0;
 let reentryOwner,reentryUntil=-Infinity,injectedReentries=0;
 const jobs=[],full=new Map(),extras={facade:new Map(),tree:new Map()},generation=new Map(),resources=[],previous=new Map();
 const allIds=manifest.cells.map(c=>c.id),transitions=createStreamTransitions(allIds,()=>{},()=>now);
 if(supplement)for(const id of treeIds)transitions.independentTree(id);
 const addJob=(due,run)=>jobs.push({due,run});
 const delayed=(ms,signal)=>new Promise(resolve=>{
  let done=false;const finish=()=>{if(done)return;done=true;signal.removeEventListener('abort',finish);resolve();};
  signal.addEventListener('abort',finish,{once:true});addJob(now+ms,finish);if(signal.aborted)finish();
 });
 function arriveExtra(id,level,token){
  if(closed||generation.get(id)!==token)return;
  const active=extras[level],prior=active.get(id),entry=prior??{token:0};entry.token++;active.set(id,entry);
  void transitions.target(id,level,true);
 }
 function leaveExtra(id,level){
  const active=extras[level],entry=active.get(id);if(!entry)return;
  const token=++entry.token;
  void transitions.target(id,level,false).then(()=>{
   if(active.get(id)!==entry||entry.token!==token)return;
   if(!closed&&transitions.value(id,level)!==0){hazards++;throw Error('Supplement disposed with positive coverage');}
   active.delete(id);
  });
 }
 const stream=manageDriveCells(manifest.cells,async(cell,signal)=>{
  const token=++serial;generation.set(cell.id,token);
  const recorded=latencies.perCellMedianMs[cell.id]??latencies.fallbackMs;
  const delay=stress?Math.min(1800,recorded*1.7+450):recorded;
  if(supplement){
   if(facadeIds.has(cell.id))addJob(now+delay*(token%2?.3:1.8),()=>arriveExtra(cell.id,'facade',token));
   if(treeIds.has(cell.id))addJob(now+delay*(token%3?.55:2.2)+90,()=>arriveExtra(cell.id,'tree',token));
  }
  await delayed(delay,signal);
  const resource={id:cell.id,attached:false,disposed:false,
   attach(){assert.equal(resource.disposed,false);assert.equal(full.has(cell.id),false);resource.attached=true;full.set(cell.id,resource);void transitions.target(cell.id,'detail',true);},
   retire(){retireCalls++;return transitions.target(cell.id,'detail',false);},
   restore(){restores++;void transitions.target(cell.id,'detail',true);},
   dispose(){assert.equal(resource.disposed,false,'Double disposal');resource.disposed=true;disposals++;
    if(resource.attached){if(!closed&&transitions.value(cell.id,'detail')!==0){hazards++;throw Error('Full resource disposed with positive coverage');}assert.equal(full.get(cell.id),resource);full.delete(cell.id);}
    if(generation.get(cell.id)===token){generation.delete(cell.id);leaveExtra(cell.id,'facade');leaveExtra(cell.id,'tree');}
   }};
  resources.push(resource);return resource;
 },()=>now);
 const family=[['facade',[drawTier.detail,drawTier.facade,drawTier.baseFacade]],['roof',[drawTier.detail,drawTier.baseRoof]],['tree',[drawTier.tree,drawTier.detailTree,drawTier.baseTree]]];
 function verify(){
  const status=stream.status();maxActive=Math.max(maxActive,status.active.length);maxRetiring=Math.max(maxRetiring,status.retiring.length);
  assert.ok(status.active.length<=3,'Physical trajectory exceeded existing three-slot diagonal bound');
  assert.equal(full.size,status.active.length,'Fading residents must stay counted');
  maxResidentBytes=Math.max(maxResidentBytes,manifest.cells.filter(c=>status.active.includes(c.id)).reduce((sum,c)=>sum+c.uncompressedBytes,0));
  maxFacades=Math.max(maxFacades,extras.facade.size);maxTrees=Math.max(maxTrees,extras.tree.size);
  assert.ok(extras.facade.size<=16&&extras.tree.size<=32,'Simulated supplemental pool exceeded production cell caps');
  for(const id of allIds){
   const values=['detail','facade','tree'].map(level=>transitions.value(id,level));
   const old=previous.get(id)??[0,0,0];
   for(let i=0;i<3;i++){const delta=Math.abs(values[i]-old[i]);maxDelta=Math.max(maxDelta,delta);assert.ok(delta<=.1,'Instantaneous coverage jump beyond 280ms ramp slope');}
   previous.set(id,values);
   for(const [kind,tiers] of family){
    const ranges=[];
    for(const tier of tiers){
     const available=tier===drawTier.detail||tier===drawTier.detailTree?full.has(id):tier===drawTier.facade?extras.facade.has(id):tier===drawTier.tree?extras.tree.has(id):true;
     const range=coverageRange(tier,...values,supplement&&treeIds.has(id));
     if(available)ranges.push(range);else assert.ok(Math.abs(range[1]-range[0])<1e-12,'Missing resource owns nonzero screen coverage');
    }
    assert.ok(Math.abs(ranges.reduce((n,[a,b])=>n+b-a,0)-1)<1e-12,'Blank/overlapping '+kind+' coverage');
    // Complementary intervals must cover each sampled rank exactly once.
    for(const rank of [0,.123,.499,.877,.999999])assert.equal(ranges.filter(([a,b])=>rank>=a&&rank<b).length,1);
    checks++;
   }
  }
 }
 async function step(p,speed){
  for(let i=jobs.length-1;i>=0;i--)if(jobs[i].due<=now){const [job]=jobs.splice(i,1);job.run();}
  await turn();transitions.advance(now);await turn();
  const fading=stream.status().retiring;
  if(reentry&&!injectedReentries&&fading.length&&turns.some(i=>Math.abs(stations[i]-mileage)<50)){
   reentryOwner=fading[0];reentryUntil=now+120;injectedReentries++;
  }
  const sight=now<reentryUntil?{cameraStation:mileage,lookStation:mileage,visible:()=>true,frontageScore:cell=>cell.id===reentryOwner?0:Infinity}:undefined;
  stream.update(mileage,...p,speed,true,sight);await turn();transitions.advance(now);await turn();verify();frame++;
 }
 await step(way[0].p,0);
 while(mileage<plannedM){
  now+=1000/60;const speed=way[Math.min(cursor+1,way.length-1)].corner?5.3:18;mileage=Math.min(plannedM,mileage+speed/60);
  while(cursor+1<stations.length&&stations[cursor+1]<mileage)cursor++;
  const next=Math.min(cursor+1,way.length-1),t=(mileage-stations[cursor])/Math.max(1e-12,stations[next]-stations[cursor]);
  const p=way[cursor].p.map((v,i)=>v+t*(way[next].p[i]-v));await step(p,speed);
 }
 // Let the last delayed attachment/fades finish under the same simulated clock.
 for(let i=0;i<180;i++){now+=1000/60;await step(way.at(-1).p,0);}
 const snapshot=stream.status(),crossings=snapshot.events.filter(e=>e.type==='crossing');
 assert.deepEqual(crossings.map(e=>e.id),fixture.expectedCrossings);
 const result={name,frames:frame,simulatedMs:now,routeM:plannedM,completedTurns:turns.filter(i=>stations[i]<=mileage).length,crossings:crossings.length,readyCrossings:crossings.filter(e=>e.ready).length,retireCalls,restores,injectedReentries,disposals,coverageFamilyChecks:checks,blankCoverage:0,overlappingCoverage:0,unsafeDisposals:hazards,maxCoverageDeltaPer60HzFrame:maxDelta,maxActiveFullCells:maxActive,maxRetiringFullCells:maxRetiring,maxResidentFullBytes:maxResidentBytes,maxFacadeCells:maxFacades,maxTreeCells:maxTrees,plannerErrors:snapshot.events.filter(e=>e.type==='error')};
 if(reentry){assert.equal(injectedReentries,1);assert.ok(restores>0,'Injected turn demand must cancel a pending retirement');}
 assert.equal(result.completedTurns,4);assert.ok(retireCalls>0);assert.equal(result.plannerErrors.length,0);
 closed=true;stream.dispose();transitions.dispose();await turn();
 // Resolve all abandoned delayed work so disposal is audited rather than hidden at shutdown.
 for(const job of jobs.splice(0))job.run();await turn();
 assert.equal(stream.status().active.length,0);assert.equal(full.size,0);assert.ok(resources.every(r=>r.disposed));
 result.shutdownDisposedResources=resources.length;result.shutdownLeakedResources=0;
 return result;
}
const wall=performance.now(),cases=[];
for(const config of [['recorded-delays',false,true],['stressed-delays',true,true],['supplement-unavailable',true,false],['turn-reentry',false,true,true]])cases.push(await replay(...config));
const report={kind:'CPU-only deterministic simulation; not rendered/GPU/FPS validation.',sourceWorldHash:manifest.sourceWorldHash,cells:manifest.cells.length,routeFixture:'full-map-1080/route.json',routeConstruction:'Current nominal 960m circuit fixture reconstructs to 964.862m with its existing 11m quadratic corners; synthetic 18m/s straights and 5.3m/s corners at 60Hz. Actual car physics and camera/frustum/occlusion are not simulated.',planner:'Current manageDriveCells',coverage:'Current createStreamTransitions and coverageRange; base representation assumed complete. Independent facade/tree resources are simulated from matching real manifests, arriving before/after full detail with deterministic delays. This does not test their production controllers, visual appearance, shaders or actual geometry overlap.',reentryCase:'One 120ms synthetic camera-demand return to a fading actual tile within 50m of a circuit corner; exercises restore and stale fade completion while moving.',latencyFixture:'arrival-latency-fixture.json; stress = min(1800ms,1.7×recorded+450ms)',frameJumpLimit:.1,wallMs:performance.now()-wall,cases};
await writeFile(path.join(destination,'graceful-route-replay.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
