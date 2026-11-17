import type {SceneManifest} from './scene-package';

type DrivePackage = SceneManifest & {gpu?: SceneManifest & {profile:string;portableHash:string}};
export type DriveCell = DrivePackage & {
  id: string; parentId?: string; startM: number; endM: number; boundsEN: number[]; tileEN?:number[];
  stats?: {buildingIds?:string[];residency?:{geometryBytes:number;portableMipBytes:number;gpuMipBytes:number}};
};
export type DriveStream = {
  scope: 'drive-stream'|'neighborhood-stream'; version: 1; sourceWorldHash: string; world: string; compactTerrain?:{world:string;ground:string}; distantExteriors?:string; facadeMiddle?:string;
  base: DrivePackage; driveFromRouteStart?:boolean; boundsEN?:number[]; routeEN: number[][]; cells: DriveCell[]; pieces?: DriveCell[]; cars?: string; overhead?: string; windingCorrections?:string;
};

export function driveDisplayPackage(source:DrivePackage,s3tc:boolean) {
  const gpu=source.gpu;
  return s3tc && gpu?.profile==='s3tc-v1' && gpu.portableHash===source.contentHash ? gpu : source;
}

function validPackage(p: SceneManifest) {
  return p && p.format === 'gzip-chunks' && /^[a-f0-9]{12}$/.test(p.contentHash || '') &&
    Number.isSafeInteger(p.uncompressedBytes) && p.uncompressedBytes! > 0 && p.compressedBytes > 0 &&
    Array.isArray(p.chunks) && p.chunks.length === Math.ceil(p.uncompressedBytes! / (8*1024**2)) &&
    p.chunks.every((url,i) => url === '/world-'+p.contentHash+'-'+String(i).padStart(2,'0')+'.bin');
}
export function validateDriveStream(value: DriveStream) {
  if(value?.facadeMiddle!==undefined && (typeof value.facadeMiddle!=='string' || !/^\/compare\/[\w.-]+\.json$/.test(value.facadeMiddle)))throw Error('Invalid middle facade asset');
  if(value?.scope==='neighborhood-stream'){
    const box=(b?:number[])=>b?.length===4&&b.every(Number.isFinite)&&b[0]<b[2]&&b[1]<b[3];
    if(value.version!==1||!/^\/compare\/[\w.-]+\.json$/.test(value.world)||!validPackage(value.base)||(value.boundsEN!==undefined&&!box(value.boundsEN))||
      !Array.isArray(value.routeEN)||value.routeEN.length<2||!value.routeEN.every(p=>Array.isArray(p)&&p.length>=2&&p.slice(0,2).every(Number.isFinite))||
      !Array.isArray(value.cells)||!value.cells.length||new Set(value.cells.map(c=>c.id)).size!==value.cells.length||
      !value.cells.every(c=>!!c.id&&validPackage(c)&&box(c.boundsEN)&&box(c.tileEN)))throw Error('Incomplete neighborhood tiles');
    return value;
  }
  const coherent = (cells:DriveCell[]) => Array.isArray(cells) && cells.length>0 &&
    new Set(cells.map(c=>c.id)).size===cells.length && cells.every((c,i)=>typeof c.id==='string' && validPackage(c) &&
      (!c.gpu||validPackage(c.gpu)) && c.boundsEN?.length===4 && c.boundsEN.every(Number.isFinite) &&
      Number.isFinite(c.endM) && c.endM>c.startM && Math.abs(c.startM-(i?cells[i-1].endM:0))<.05);
  if (value?.scope !== 'drive-stream' || value.version !== 1 || !/^\/compare\/[\w.-]+\.json$/.test(value.world) ||
      (!validPackage(value.base) || (value.base.gpu && !validPackage(value.base.gpu))) || value.routeEN.length < 2 || !value.routeEN.every(p => p.length >= 2 && p.slice(0,2).every(Number.isFinite)) ||
      !coherent(value.cells)) throw Error('Incomplete driving corridor');
  const length = value.routeEN.reduce((sum,p,i) => i ? sum + Math.hypot(p[0]-value.routeEN[i-1][0],p[1]-value.routeEN[i-1][1]) : 0, 0);
  if (Math.abs(length-value.cells.at(-1)!.endM) > 1) throw Error('Driving corridor and street length disagree');
  if(value.pieces && (!coherent(value.pieces) || Math.abs(length-value.pieces.at(-1)!.endM)>1 ||
    value.pieces.some(p=>!value.cells.some(c=>c.id===p.parentId && p.startM>=c.startM-.05 && p.endM<=c.endM+.05))))throw Error('Incomplete driving detail');
  return value;
}

export type CellResources = {attach(): void; dispose(): void;
  /** Resolve only once the resident resource is safe to release. */
  retire?(): Promise<void>;
  /** Cancel a pending retirement and synchronously restore its visible state. */
  restore?(): void;
};
export type DriveSight = {cameraStation:number; lookStation:number; visible:(cell:DriveCell)=>boolean;
  /** Stable camera/clip revision permits exact parked-plan reuse. */
  revision?:number;
  /** Lower scores identify nearby visible building frontages; Infinity means none. */
  frontageScore?:(cell:DriveCell,east:number,north:number)=>number;
  /** Distance to a visible exterior divided by its street-detail distance. */
  nearFrontage?:(cell:DriveCell)=>number;
  residentCost?:(cell:DriveCell)=>number;
  hasWindowGeometry?:(cell:DriveCell)=>boolean;
  predictedFrontage?:(cell:DriveCell)=>number};
type EventType = 'prepare'|'attached'|'evicted'|'error'|'crossing';
type StreamEvent = {type:EventType; id:string; ms:number; positionEN:number[]; ready:boolean; message?:string; piece?:string};

// Distance along the car's actual direction to a tile, including a short tile
// crossed at an intersection before the tile containing the look-ahead point.
function tileEntry(box:number[],e:number,n:number,dx:number,dy:number){
  let enter=0,exit=Infinity;
  for(let axis=0;axis<2;axis++){
    const origin=axis?n:e,dir=axis?dy:dx,lo=box[axis],hi=box[axis+2];
    if(Math.abs(dir)<1e-8){if(origin<lo||origin>hi)return Infinity;continue;}
    const a=(lo-origin)/dir,b=(hi-origin)/dir;
    enter=Math.max(enter,Math.min(a,b));exit=Math.min(exit,Math.max(a,b));
  }
  return exit>enter+.02?enter:Infinity;
}

// Keep the current block and one neighbor. Use travel direction (including reverse)
// and speed to predict the next block. The rear margin keeps the chase view intact.
export function manageDriveCells(cells: DriveCell[], load: (cell:DriveCell, signal:AbortSignal)=>Promise<CellResources>,
  clock:()=>number=()=>performance.now()) {
  const spatial=!!cells[0]?.tileEN;
  let lastE=NaN,lastN=NaN,headingE=0,headingN=0;
  const resident = new Map<number,CellResources>(), events:StreamEvent[]=[];
  const retiring = new Map<number,{resource:CellResources;done:Promise<void>}>();
  const retirementRetries = new Map<number,number>();
  const changed = new Set<()=>void>();
  const notify = () => {for(const wake of [...changed])wake();};
  const available = (i:number) => resident.has(i) && !retiring.has(i);
  function waitForChange(signal:AbortSignal) {
    return new Promise<void>(resolve=>{
      const wake=()=>{changed.delete(wake);signal.removeEventListener('abort',wake);resolve();};
      changed.add(wake);signal.addEventListener('abort',wake,{once:true});
      if(closed||signal.aborted)wake();
    });
  }
  let current = -1, direction = 1, station = 0, directionStation = 0, positionEN = [0,0], wanted:number[]=[];
  let work: {index:number; controller:AbortController; done:Promise<void>} | null=null;
  let closed = false, moving = false, initialStation:number|undefined;
  let displayedPrimary=-1,spatialLimit=1;
  let arrivals:number[]=[];
  let urgentArrival=false,arrivalLeadSeconds=1.8;
  const retries = new Map<number,number>();
  const lastClose = new Float64Array(cells.length).fill(-Infinity);
  let nextVisibilityExpiry=Infinity;
  let resourceRevision=0;
  let parkedPlan:{east:number;north:number;speed:number;prefetch:boolean;sight?:DriveSight;viewRevision?:number;resources:number}|undefined;
  function note(type:EventType, index:number, message?:string) {
    const piece=cells[index];
    events.push({type,id:type==='crossing'?(piece.parentId??piece.id):piece.id,piece:piece.id,ms:clock(),positionEN:[...positionEN],ready:resident.has(index),message});
    if (events.length > 256) events.shift();
  }
  function release(index:number,resource:CellResources) {
    if(resident.get(index)!==resource)return;
    resident.delete(index);retiring.delete(index);retirementRetries.delete(index);resourceRevision++;
    resource.dispose();note('evicted',index);notify();
  }
  function discard(index:number,force=false) {
    const resource=resident.get(index);
    if(!resource)return;
    if(force||!resource.retire){release(index,resource);return;}
    if(retiring.has(index)||(retirementRetries.get(index)??0)>clock())return;
    const task={resource,done:Promise.resolve()};retiring.set(index,task);resourceRevision++;
    task.done=Promise.resolve().then(()=>{
      if(retiring.get(index)===task)return resource.retire!();
    }).then(()=>{
      if(retiring.get(index)!==task)return;
      release(index,resource);
    },error=>{
      if(retiring.get(index)!==task)return;
      // A failed fade has not promised that the object is invisible. Retain it
      // and retry on a later update rather than freeing visible GPU resources.
      retiring.delete(index);resourceRevision++;retirementRetries.set(index,clock()+3000);
      resource.restore?.();note('error',index,'Retirement failed: '+String(error));
    }).finally(()=>{notify();schedule();});
  }
  function restoreWanted() {
    for(const index of wanted){
      const task=retiring.get(index);
      if(!task?.resource.restore)continue;
      retiring.delete(index);retirementRetries.delete(index);resourceRevision++;
      task.resource.restore();notify();
    }
  }
  function schedule() {
    if (closed || work) return;
    const order=spatial?[...arrivals,...wanted.filter(i=>!arrivals.includes(i))]:wanted;
    const index=order.find(i=>!resident.has(i) && (retries.get(i)??0) <= clock());
    if (index===undefined) return;
    const controller=new AbortController();
    const task={index,controller,done:Promise.resolve()},started=clock();
    work=task; note('prepare',index);
    task.done=Promise.resolve().then(()=>load(cells[index],controller.signal)).then(async resource=>{
      if (closed || controller.signal.aborted || !wanted.includes(index)) {resource.dispose();return;}
      const limit=()=>spatial?spatialLimit:2;
      while(resident.size>=limit()){
        // Fading resources still occupy real memory and attachment slots.
        // Wait for them; a new desired object never evicts a visible fade.
        const obsolete=[...resident.keys()].filter(i=>!wanted.includes(i)&&!retiring.has(i));
        const victim=obsolete.find(i=>i!==displayedPrimary)??obsolete[0];
        if(victim!==undefined)discard(victim);
        if(resident.size<limit())break;
        await waitForChange(controller.signal);
        if(closed||controller.signal.aborted||!wanted.includes(index)){resource.dispose();return;}
      }
      try {resource.attach();} catch(error) {resource.dispose();throw error;}
      resident.set(index,resource);resourceRevision++; retries.delete(index); note('attached',index);
      if(spatial){
        if(moving)arrivalLeadSeconds=Math.min(3.5,Math.max(1.8,arrivalLeadSeconds*.9,2*(clock()-started)/1000+.25));
        if(index===wanted[0]||!resident.has(displayedPrimary))displayedPrimary=index;
        pruneSpatial();
      }
    }).catch(error=>{
      if (!closed && !controller.signal.aborted) {
        retries.set(index,clock()+3000); note('error',index,String(error));
      }
    }).finally(()=>{if(work===task)work=null; schedule();});
  }
  function pruneSpatial(){
    if(available(wanted[0])){
      displayedPrimary=wanted[0];
      for(const index of resident.keys())if(!wanted.includes(index))discard(index);
    }else while(resident.size-retiring.size>spatialLimit){
      const victim=[...resident.keys()].find(i=>!wanted.includes(i)&&i!==displayedPrimary&&!retiring.has(i))
        ??[...resident.keys()].find(i=>!wanted.includes(i)&&!retiring.has(i));
      if(victim===undefined)break;
      const before=resident.size-retiring.size;discard(victim);
      if(resident.size-retiring.size===before)break;
    }
  }
  function planTiles(e:number,n:number,speed:number,prefetch:boolean,sight?:DriveSight){
    arrivals=[];spatialLimit=1;urgentArrival=false;
    const previous=current;
    if(!Number.isFinite(lastE)){lastE=e;lastN=n;}
    const dx=e-lastE,dy=n-lastN,travel=Math.hypot(dx,dy);
    if(travel>.6){moving=true;headingE=dx/travel;headingN=dy/travel;lastE=e;lastN=n;}
    // A tile ahead of the car at a low-speed bend can be several seconds
    // beyond the turn. Limit speculative uploads to reachable distance;
    // the visible frontage still gets its own detail slot independently.
    const lookAhead=Math.min(160,Math.max(16,Math.abs(speed)*3.5));
    let nearest=Infinity;
    current=-1;
    const distanceSq=(box:number[],x:number,y:number)=>{
      const dx=Math.max(box[0]-x,0,x-box[2]),dy=Math.max(box[1]-y,0,y-box[3]);
      return dx*dx+dy*dy;
    };
    for(let i=0;i<cells.length;i++){
      const d=distanceSq(cells[i].tileEN!,e,n);
      if(d<nearest || (d===nearest&&i===previous)){nearest=d;current=i;}
    }
    let primary=current,frontage=Infinity;
    const close:{index:number;priority:number}[]=[];
    const predicted:{index:number;priority:number}[]=[];
    const now=clock();
    nextVisibilityExpiry=Infinity;
    if(sight?.frontageScore)for(let i=0;i<cells.length;i++){
      const score=sight.frontageScore(cells[i],e,n)-(i===displayedPrimary?14:resident.has(i)||work?.index===i?5:0);
      if(score<frontage){frontage=score;primary=i;}
      if(sight.nearFrontage){
        const priority=sight.nearFrontage(cells[i]);
        const held=resident.has(i)||work?.index===i;
        if(priority<=(held?1.2:1)){
          lastClose[i]=now;
          close.push({index:i,priority:priority-(held ? .08 : 0)});
        }else if(held&&now-lastClose[i]<750){
          close.push({index:i,priority:2});
          nextVisibilityExpiry=Math.min(nextVisibilityExpiry,lastClose[i]+750);
        }
        if(moving&&priority>1&&sight.predictedFrontage){
          const next=sight.predictedFrontage(cells[i]);
          if(next<=1)predicted.push({index:i,priority:next-(held ? .08 : 0)});
        }
      }
    }
    // Detail ownership follows the visible facade, while current remains the
    // physical tile for crossing readiness and the arrival ray below. Empty
    // park/road tiles must not consume the only parked-start detail slot.
    wanted=[primary];
    if(previous>=0&&previous!==current)note('crossing',current);
    if(!prefetch)return;
    const protectView=()=>{
      // Encoded bytes describe download cost poorly. Bound extra *decoded*
      // geometry admitted alongside the normal physical arrival and primary.
      // The closest visible walls get the budget first, on both sides.
      let used=0;
      const ranked=close.sort((a,b)=>a.priority-b.priority);
      // At most one extra owner just around a bend. Real frontages and
      // physical crossings get first claim on the same existing byte budget.
      const next=predicted.sort((a,b)=>a.priority-b.priority).find(p=>!wanted.includes(p.index)&&!ranked.some(r=>r.index===p.index));
      if(next)ranked.push(next);
      for(const {index,priority} of ranked){
        if(wanted.includes(index))continue;
        const cell=cells[index];
        // Where native window/trim geometry already covers a moderate-size
        // frontage, leave the detailed material slot for the next close wall.
        // The primary owner and physical arrivals have already been reserved.
        if(sight?.hasWindowGeometry?.(cell)&&priority>.75)continue;
        const bytes=sight?.residentCost?.(cell)??cell.uncompressedBytes??0;
        // One very close opposite frontage may itself exceed the optional
        // budget. Let it take that slot alone, instead of permanently denying
        // detail on a large wall next to the car.
        const soleCloseFront=used===0&&priority<=.42;
        if((used+bytes>32*1024*1024&&!soleCloseFront)||wanted.length>=8)continue;
        wanted.push(index);used+=bytes;
      }
      spatialLimit=Math.max(spatialLimit,wanted.length);
    };
    if(!moving){protectView();return;}
    spatialLimit=2;
    let candidate=-1,best=Infinity,first=-1,second=-1,entry1=Infinity,entry2=Infinity;
    const searchRadius=lookAhead+4,radiusSq=searchRadius*searchRadius;
    const aheadE=e+lookAhead*headingE,aheadN=n+lookAhead*headingN;
    for(let i=0;i<cells.length;i++){
      if(i===current)continue;
      const tile=cells[i].tileEN!;
      const nearSq=distanceSq(tile,e,n);
      if(nearSq>radiusSq)continue;
      const near=Math.sqrt(nearSq),ahead=Math.sqrt(distanceSq(tile,aheadE,aheadN));
      const entry=tileEntry(tile,e,n,headingE,headingN);
      if(entry<entry1){entry2=entry1;second=first;entry1=entry;first=i;}
      else if(entry<entry2){entry2=entry;second=i;}
      const score=ahead+.40*near+(sight&&!sight.visible(cells[i])?22:0)-(resident.has(i)||work?.index===i?5:0);
      if(score<best){candidate=i;best=score;}
    }
    const keepLead=first>=0&&(resident.has(first)||work?.index===first)?4:0;
    if(first>=0 && entry1<=lookAhead+keepLead){
      // A distant crossing can wait for an already useful upload to finish.
      // Reserve two measured preparation times before preempting it, so slow
      // devices still get the crossing ready before the car arrives.
      urgentArrival=entry1<=Math.max(4,Math.abs(speed)*arrivalLeadSeconds);
      arrivals.push(first);
      if(!wanted.includes(first))wanted.push(first);
      // On a fast diagonal, two boundaries can be less than one frame apart.
      // Warm both destinations early, then release the corner slot immediately.
      if(second>=0 && entry2<=lookAhead && entry2-entry1<Math.min(14,Math.abs(speed)*.6)){
        arrivals.push(second);
        if(!wanted.includes(second))wanted.push(second);
        spatialLimit=Math.max(2,wanted.length);
      }
    }else if(first<0&&candidate>=0&&distanceSq(cells[current].tileEN!,aheadE,aheadN)>0&&
      distanceSq(cells[candidate].tileEN!,aheadE,aheadN)<=16&&
      !wanted.includes(candidate))wanted.push(candidate);
    if(wanted.length===1){
      // Slowing down can shorten the prediction before reaching an already
      // prepared tile. Keep that neighbor in its existing slot while it is
      // still on our heading; a new imminent arrival always replaces it.
      const keep=[...resident.keys(),...(work?[work.index]:[])].find(i=>i!==primary&&i!==current&&
        tileEntry(cells[i].tileEN!,e,n,headingE,headingN)<=160);
      if(keep!==undefined)wanted.push(keep);
    }
    protectView();
  }
  function update(distance:number, east:number, north:number, speedMps=0, prefetch=true, sight?:DriveSight) {
    if (closed) return;
    station=distance;positionEN[0]=east;positionEN[1]=north;
    if(spatial){
      // Camera changes invalidate the exact cache. Moving cars retain every
      // physical/corner planning update; only identical parked inputs reuse it.
      const reusable=!moving&&clock()<nextVisibilityExpiry&&(!sight?.frontageScore||sight.revision!==undefined)&&parkedPlan&&
        parkedPlan.east===east&&parkedPlan.north===north&&parkedPlan.speed===speedMps&&parkedPlan.prefetch===prefetch&&
        parkedPlan.sight===sight&&parkedPlan.viewRevision===sight?.revision&&parkedPlan.resources===resourceRevision;
      if(!reusable){
        planTiles(east,north,speedMps,prefetch,sight);
        parkedPlan=moving?undefined:{east,north,speed:speedMps,prefetch,sight,viewRevision:sight?.revision,resources:resourceRevision};
      }
    }
    else planCorridor(speedMps,prefetch,sight);
    restoreWanted();
    if(spatial)pruneSpatial();else for(const index of resident.keys())if(!wanted.includes(index))discard(index);
    notify();
    const missingArrival=arrivals.find(i=>!resident.has(i));
    if(work && (!wanted.includes(work.index)||spatial&&(urgentArrival||!sight?.nearFrontage)&&missingArrival!==undefined&&!arrivals.includes(work.index)))work.controller.abort();
    schedule();
  }
  function planCorridor(speedMps:number,prefetch:boolean,sight?:DriveSight){
    initialStation??=station;
    // Do not overlap a second section's uploads with the first ready frame of a parked car.
    // Once travel begins, retain the normal lead through stops and direction changes.
    moving ||= Math.abs(speedMps)>.35 || Math.abs(station-initialStation)>.5;
    let lo=0,hi=cells.length-1;
    while(lo<hi){const m=(lo+hi)>>1;if(station>=cells[m].endM)lo=m+1;else hi=m;}
    if (current<0) directionStation=station;
    else if(Math.abs(station-directionStation)>1){direction=station>directionStation?1:-1;directionStation=station;}
    const previous=current;current=lo;
    if(previous>=0 && (cells[current].parentId??cells[current].id)!==(cells[previous].parentId??cells[previous].id))note('crossing',current);
    const cell=cells[current], ahead=current+direction, behind=current-direction;
    const toBoundary=direction>0?cell.endM-station:station-cell.startM;
    const sinceBoundary=direction>0?station-cell.startM:cell.endM-station;
    const lead=Math.min(135,Math.max(95,70+Math.abs(speedMps)*2.5));
    wanted=[current];
    const tailMargin=Math.min(20,(cell.endM-cell.startM)*.25);
    const keepTail=sinceBoundary<tailMargin && resident.has(behind);
    if(prefetch && moving && sight){
      const front=cells[ahead], rear=cells[behind];
      const lookingBack=(sight.lookStation-sight.cameraStation)*direction < -2;
      const cameraBehind=(sight.cameraStation-(direction>0?cell.startM:cell.endM))*direction < 1;
      const approaching=front && toBoundary<=Math.max(8,Math.abs(speedMps)*1.25);
      // Reserve the arrival section when it is due. Else spend the second slot on what
      // the driver can actually see, including a backward look or a trailing chase camera.
      if(rear && sight.visible(rear) && !approaching &&
         (lookingBack || (resident.has(behind) && (cameraBehind || keepTail))))wanted.push(behind);
      else if(front && toBoundary<=lead && (sight.visible(front) || approaching || !lookingBack))wanted.push(ahead);
    } else if(keepTail) wanted.push(behind);
    else if(prefetch && moving && cells[ahead] && toBoundary<=lead) wanted.push(ahead);
  }
  return {
    update,
    async waitForCurrent() {
      while(!available(wanted[0]??current)){
        if(work)await work.done;
        else {const task=retiring.get(wanted[0]??current);if(task)await task.done;else break;}
      }
      if(!available(wanted[0]??current))throw Error(events.findLast(e=>e.type==='error')?.message || 'Could not prepare the starting block');
    },
    status:()=>({active:[...resident.keys()].map(i=>cells[i].id), retiring:[...retiring.keys()].map(i=>cells[i].id), loading:work?cells[work.index].id:null,
      current:cells[current]?.parentId??cells[current]?.id,piece:cells[spatial&&resident.has(displayedPrimary)?displayedPrimary:wanted[0]??current]?.id,station,
      ready:available(wanted[0]??current)||spatial&&available(displayedPrimary),events:events.slice()}),
    dispose(){closed=true;work?.controller.abort();for(const i of resident.keys())discard(i,true);retries.clear();retirementRetries.clear();notify();},
  };
}
