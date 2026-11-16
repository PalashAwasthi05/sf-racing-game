import assert from 'node:assert/strict';
import test from 'node:test';
import {manageDriveCells,driveDisplayPackage,validateDriveStream,type DriveCell,type DriveStream} from './drive-stream.ts';

const cells=Array.from({length:4},(_,i)=>({id:String(i),startM:i*146,endM:(i+1)*146,boundsEN:[i*146,-20,(i+1)*146,20],chunks:[],compressedBytes:10}));
const settle=()=>new Promise<void>(r=>setTimeout(r,0));
test('a parked start prepares one short section and begins prefetch as soon as the car moves',async()=>{
  const pieces=cells.map((c,i)=>({...c,startM:i*48,endM:(i+1)*48}));
  const drive=manageDriveCells(pieces,async()=>({attach(){},dispose(){}}));
  for(let frame=0;frame<4;frame++){drive.update(0,0,0,0);await settle();}
  assert.deepEqual(drive.status().active,['0']);
  drive.update(.1,.1,0,.5);await settle();
  drive.update(.1,.1,0,0);await settle();
  assert.deepEqual(drive.status().active,['0','1']);
  drive.dispose();
});
test('prepares ahead, evicts behind, and loads again on return with at most two resident details',async()=>{
  const live=new Set<string>(),loads:string[]=[];
  const drive=manageDriveCells(cells,async c=>({attach(){live.add(c.id);loads.push(c.id);assert.ok(live.size<=2);},dispose(){live.delete(c.id);}}));
  for(const m of [0,80,149,173,243,294,317,370,445,464,520,475,448,416,355,290,265,209,144,121,60]){
    drive.update(m,m,0,16);await settle();
  }
  assert.deepEqual(loads,['0','1','2','3','2','1','0']);
  const crosses=drive.status().events.filter(e=>e.type==='crossing');
  assert.equal(crosses.length,6);assert.ok(crosses.every(e=>e.ready));
  drive.dispose();assert.equal(live.size,0);
});
test('a reversed or closed drive releases an obsolete in-flight cell before starting another',async()=>{
  let resolve=()=>{},released=0;
  const drive=manageDriveCells(cells,async c=>{if(c.id==='1')await new Promise<void>(r=>resolve=r);
    return {attach(){},dispose(){released++;}};});
  drive.update(0,0,0);await drive.waitForCurrent();
  drive.update(80,80,0);await settle();drive.update(5,5,0);
  resolve();await settle();assert.deepEqual(drive.status().active,['0']);assert.equal(released,1);
  drive.dispose();assert.equal(released,2);
});
test('a backward view retains the visible neighbor until the next arrival needs the slot',async()=>{
  const pieces=cells.map((c,i)=>({...c,startM:i*50,endM:(i+1)*50}));
  const drive=manageDriveCells(pieces,async()=>({attach(){},dispose(){}}));
  const look=(cameraStation:number,lookStation:number)=>({cameraStation,lookStation,visible:()=>true});
  for(const m of [0,20,51,70]){
    drive.update(m,m,0,10,true,look(m-4,m<50?m+30:m-40));await settle();
  }
  assert.deepEqual(new Set(drive.status().active),new Set(['1','0']));
  drive.update(73,73,0,10,true,look(69,105));await settle();
  assert.deepEqual(new Set(drive.status().active),new Set(['1','2']));
  drive.update(93,93,0,10,true,look(89,50));await settle();
  assert.deepEqual(new Set(drive.status().active),new Set(['1','2']));
  drive.update(101,101,0,10,true,look(97,135));await settle();
  assert.ok(drive.status().ready);drive.dispose();
});
test('compressed cell must belong to the current portable encoding',()=>{
  const source={...cells[0],contentHash:'a',gpu:{profile:'s3tc-v1',portableHash:'a',chunks:[],compressedBytes:5}};
  assert.equal(driveDisplayPackage(source,true),source.gpu);assert.equal(driveDisplayPackage(source,false),source);
  source.gpu.portableHash='b';assert.equal(driveDisplayPackage(source,true),source);
});
test('only reports the mapped street boundary when detail sections change',async()=>{
  const pieces=Array.from({length:6},(_,i)=>({...cells[0],id:'part'+i,parentId:'street'+Math.floor(i/3),startM:i*50,endM:(i+1)*50}));
  const drive=manageDriveCells(pieces,async()=>({attach(){},dispose(){}}));
  for(const m of [0,20,51,66,101,117,151,167,201]){drive.update(m,m,0);await settle();}
  const events=drive.status().events.filter(e=>e.type==='crossing');
  assert.equal(events.length,1);assert.deepEqual([events[0].id,events[0].piece,events[0].ready],['street1','part3',true]);
  drive.dispose();
});

test('a turn off the benchmark road streams spatial neighbors and releases the previous block',async()=>{
  const grid=[[0,0],[1,0],[0,1],[0,2]].map(([x,y],i)=>{
    const tileEN=[x*55,y*55,(x+1)*55,(y+1)*55];
    return {...cells[i],tileEN,boundsEN:tileEN};
  });
  const live=new Set<string>();
  const drive=manageDriveCells(grid,async c=>({attach(){live.add(c.id);assert.ok(live.size<=2);},dispose(){live.delete(c.id);}}));
  const step=async(n:number)=>{drive.update(0,22,n,10,true,{cameraStation:0,lookStation:0,visible:()=>true});await settle();};
  await step(15);assert.deepEqual([...live],['0']);
  await step(43);assert.deepEqual([...live],['0','2']);
  await step(62);assert.equal(drive.status().piece,'2');assert.ok(drive.status().ready);
  await step(104);assert.deepEqual(new Set(live),new Set(['2','3']));
  await step(118);assert.equal(drive.status().piece,'3');
  await step(115);await step(108);assert.equal(drive.status().piece,'2');
  drive.dispose();assert.equal(live.size,0);
});

test('a diagonal approach prepares the narrow corner tile and its immediate exit',async()=>{
  const tiles=[[0,0],[1,0],[0,1],[1,1]].map(([x,y],i)=>{
    const tileEN=[55*x,55*y,55*(x+1),55*(y+1)];return {...cells[i],tileEN,boundsEN:tileEN};
  });
  const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
  for(const [e,n] of [[37,35],[40,38],[55.5,53.5],[58,56]]){
    stream.update(0,e,n,28);await settle();
  }
  const crossings=stream.status().events.filter(e=>e.type==='crossing');
  assert.deepEqual(crossings.map(e=>[e.id,e.ready]),[['1',true],['3',true]]);
  stream.dispose();
});

test('a slow bend warms the upcoming exit without uploading a distant tile on the old heading',async()=>{
 const tiles=[[0,0],[0,1],[0,2],[1,1]].map(([x,y],i)=>{
   const tileEN=[55*x,55*y,55*(x+1),55*(y+1)];return {...cells[i],tileEN,boundsEN:tileEN};
 });
 const loads:string[]=[];
 const stream=manageDriveCells(tiles,async c=>{loads.push(c.id);return {attach(){},dispose(){}};});
 for(const [e,n] of [[34,39],[34,50],[34,56],[35,59],[38,62],[43,63],[52,64],[56,64]]){
   stream.update(0,e,n,5.5);await settle();
 }
 assert.deepEqual(loads,['0','1','3']);
 assert.deepEqual(stream.status().events.filter(e=>e.type==='crossing').map(e=>[e.id,e.ready]),[['1',true],['3',true]]);
 stream.dispose();
});

test('slowing partway to a prepared tile retains it until the crossing',async()=>{
 const tiles=[0,1].map((x,i)=>({...cells[i],tileEN:[55*x,0,55*(x+1),55]}));
 const loads:string[]=[];
 const stream=manageDriveCells(tiles,async c=>{loads.push(c.id);return {attach(){},dispose(){}};});
 for(const [east,speed] of [[15,0],[17,16],[20,4],[36,4],[56,10]]){
   stream.update(0,east,20,speed);await settle();
 }
 assert.deepEqual(loads,['0','1']);
 assert.ok(stream.status().events.find(e=>e.type==='crossing')?.ready);
 stream.dispose();
});

test('a parked park tile spends its only detail slot on the visible opposite frontage',async()=>{
 const tiles=[[0,0],[0,-1],[1,0]].map(([x,y],i)=>{
  const tileEN=[55*x,55*y,55*(x+1),55*(y+1)];return {...cells[i],tileEN,boundsEN:tileEN};
 });
 const live=new Set<string>();
 const stream=manageDriveCells(tiles,async c=>({attach(){live.add(c.id);assert.ok(live.size<=2);},dispose(){live.delete(c.id);}}));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(cell:DriveCell)=>cell.id==='1'?8:Infinity};
 stream.update(0,22,2,0,false,sight);await stream.waitForCurrent();
 assert.deepEqual([...live],['1']);assert.equal(stream.status().piece,'1');assert.equal(stream.status().current,'0');
 for(let i=0;i<3;i++){stream.update(0,22,2,0,true,sight);await settle();}
 assert.deepEqual([...live],['1']);
 stream.update(0,24,2,10,true,sight);await settle();
 assert.deepEqual(new Set(live),new Set(['1','2']));
 stream.update(0,56,2,10,true,sight);await settle();
 assert.deepEqual(stream.status().events.filter(e=>e.type==='crossing').map(e=>[e.id,e.ready]),[['2',true]]);
 stream.dispose();
});

test('both visible street fronts keep detail while a separate arrival warms, including when parked',async()=>{
 const tiles=[[0,0],[1,0],[0,-1],[0,1]].map(([x,y],i)=>({...cells[i],uncompressedBytes:5*1024**2,tileEN:[55*x,55*y,55*(x+1),55*(y+1)]}));
 let now=0,revision=0;
 const close=new Map([['2',.2],['3',.3]]);
 const sight={cameraStation:0,lookStation:0,visible:()=>true,get revision(){return revision;},
   frontageScore:(c:DriveCell)=>c.id==='2'?1:20,nearFrontage:(c:DriveCell)=>close.get(c.id)??Infinity};
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}),()=>now);
 stream.update(0,20,20,0,false,sight);await settle();assert.deepEqual(stream.status().active,['2']);
 stream.update(0,20,20,0,true,sight);await settle();assert.deepEqual(new Set(stream.status().active),new Set(['2','3']));
 close.delete('3');revision++;now=50;stream.update(0,20,20,0,true,sight);await settle();
 assert.ok(stream.status().active.includes('3'));
 now=900;stream.update(0,20,20,0,true,sight);await settle();assert.deepEqual(stream.status().active,['2']);
 close.set('3',.25);revision++;stream.update(0,25,20,12,true,sight);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['1','2','3']));
 stream.dispose();
});

test('visible frontage admissions prioritize nearest walls within the decoded-byte budget',async()=>{
 const tiles=Array.from({length:7},(_,i)=>({...cells[0],id:String(i),uncompressedBytes:12*1024**2,tileEN:[55*i,0,55*(i+1),55]}));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>Number(c.id),
   nearFrontage:(c:DriveCell)=>1-Number(c.id)/10};
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 stream.update(0,5,5,0,true,sight);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['0','6','5']));
 stream.dispose();
});

test('warms one predicted frontage after motion while retaining both current street fronts',async()=>{
 const tiles=Array.from({length:5},(_,i)=>({...cells[0],id:String(i),uncompressedBytes:7*1024**2,tileEN:[i*100,0,(i+1)*100,100]}));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>Number(c.id),
   nearFrontage:(c:DriveCell)=>Number(c.id)<2 ? .4 : Infinity,predictedFrontage:(c:DriveCell)=>Number(c.id)/10};
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 stream.update(0,12,12,0,true,sight);await settle();assert.deepEqual(new Set(stream.status().active),new Set(['0','1']));
 stream.update(1,13,12,8,true,sight);await settle();assert.deepEqual(new Set(stream.status().active),new Set(['0','1','2']));
 stream.dispose();
});

test('native window geometry can cover a wider frontage while close detail uses decoded residency',async()=>{
 const tiles=Array.from({length:5},(_,i)=>({...cells[0],id:String(i),uncompressedBytes:1024,tileEN:[i*100,0,(i+1)*100,100]}));
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 const priority=[.1,.35,.6,.9,.99],cost=[0,18,12,1,20];
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>Number(c.id),
   nearFrontage:(c:DriveCell)=>priority[Number(c.id)],hasWindowGeometry:(c:DriveCell)=>c.id==='3',residentCost:(c:DriveCell)=>cost[Number(c.id)]*1024**2};
 stream.update(0,5,10,0,true,sight);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['0','1','2']));
 stream.dispose();
 const oversized=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 cost[1]=42;
 oversized.update(0,5,10,0,true,sight);await settle();
 assert.deepEqual(new Set(oversized.status().active),new Set(['0','1']));
 oversized.dispose();
});

test('a distant crossing leaves a visible frontage loading until the crossing deadline',async()=>{
 const tiles=[[0,0],[1,0],[0,1]].map(([x,y],i)=>({...cells[i],tileEN:[100*x,100*y,100*x+100,100*y+100]}));
 let now=0,frontageStarted=false,aborts=0;
 const stream=manageDriveCells(tiles,async(c,signal)=>{
   if(c.id==='2'&&!frontageStarted){
     frontageStarted=true;
     await new Promise<void>(resolve=>signal.addEventListener('abort',()=>{aborts++;resolve();},{once:true}));
   }
   return {attach(){},dispose(){}};
 },()=>now);
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id==='0'?1:10,
   nearFrontage:(c:DriveCell)=>c.id==='2' ? .5 : Infinity};
 stream.update(0,50,50,0,true,sight);await settle();assert.ok(frontageStarted);
 now=100;stream.update(18,68,50,10,true,sight);await settle();assert.equal(aborts,0);
 now=1500;stream.update(38,88,50,10,true,sight);await settle();assert.equal(aborts,1);assert.ok(stream.status().active.includes('1'));
 stream.dispose();
});

test('turning the view changes frontage detail without replacing the imminent crossing slot',async()=>{
 const tiles=[[0,0],[0,-1],[1,0],[0,1]].map(([x,y],i)=>{
  const tileEN=[55*x,55*y,55*(x+1),55*(y+1)];return {...cells[i],tileEN,boundsEN:tileEN};
 });
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 let visibleOwner='1';
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(cell:DriveCell)=>cell.id===visibleOwner?8:Infinity};
 stream.update(0,22,20,0,false,sight);await stream.waitForCurrent();
 stream.update(0,24,20,15,true,sight);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['1','2']));
 visibleOwner='3';stream.update(0,26,20,15,true,sight);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['3','2']));
 stream.update(0,56,20,15,true,sight);await settle();
 assert.ok(stream.status().events.filter(e=>e.type==='crossing').every(e=>e.ready));
 stream.dispose();
});

test('view-owned detail preserves both ready diagonal arrivals in the rare third slot',async()=>{
 const tiles=[[0,0],[1,0],[0,1],[1,1]].map(([x,y],i)=>{
  const tileEN=[55*x,55*y,55*(x+1),55*(y+1)];return {...cells[i],tileEN,boundsEN:tileEN};
 });
 const live=new Set<string>();let maximum=0;
 const stream=manageDriveCells(tiles,async c=>({attach(){live.add(c.id);maximum=Math.max(maximum,live.size);assert.ok(live.size<=3);},dispose(){live.delete(c.id);}}));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(cell:DriveCell)=>cell.id==='0'?8:Infinity};
 for(const [e,n] of [[37,35],[40,38],[55.5,53.5],[58,56]]){
  stream.update(0,e,n,28,true,sight);await settle();
 }
 assert.equal(maximum,3);
 assert.deepEqual(stream.status().events.filter(e=>e.type==='crossing').map(e=>[e.id,e.ready]),[['1',true],['3',true]]);
 assert.ok(stream.status().active.length<=2);
 stream.dispose();
});

test('a deferred parked frontage swap keeps the old owner visible until atomic attachment',async()=>{
 const tiles=[[0,0],[0,-1],[0,1]].map(([x,y],i)=>({...cells[i],tileEN:[55*x,55*y,55*(x+1),55*(y+1)]}));
 let visibleOwner='1',resolveSwap=()=>{};const live=new Set<string>();
 const stream=manageDriveCells(tiles,async cell=>{
  if(cell.id==='2')await new Promise<void>(resolve=>resolveSwap=resolve);
  return {attach(){live.add(cell.id);assert.ok(live.size<=1);},dispose(){live.delete(cell.id);}};
 });
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id===visibleOwner?0:Infinity};
 stream.update(0,20,20,0,false,sight);await stream.waitForCurrent();
 visibleOwner='2';stream.update(0,20,20,0,true,sight);await settle();
 assert.deepEqual(stream.status().active,['1']);assert.equal(stream.status().piece,'1');assert.ok(stream.status().ready);
 resolveSwap();await settle();assert.deepEqual(stream.status().active,['2']);assert.equal(stream.status().piece,'2');
 stream.dispose();
});

test('an impending crossing preempts deferred optional frontage without removing the displayed fallback',async()=>{
 const tiles=[[0,0],[0,-1],[1,0],[0,1]].map(([x,y],i)=>({...cells[i],tileEN:[55*x,55*y,55*(x+1),55*(y+1)]}));
 const loads:string[]=[],live=new Set<string>();let visibleOwner='1',releaseArrival=()=>{},releaseFrontage=()=>{};
 const stream=manageDriveCells(tiles,async(cell,signal)=>{
  loads.push(cell.id);
  if(cell.id==='3')await new Promise<void>(resolve=>{releaseFrontage=resolve;signal.addEventListener('abort',()=>resolve(),{once:true});});
  if(cell.id==='2')await new Promise<void>(resolve=>releaseArrival=resolve);
  return {attach(){live.add(cell.id);assert.ok(live.size<=2);},dispose(){live.delete(cell.id);}};
 });
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id===visibleOwner?0:Infinity};
 stream.update(0,20,20,0,false,sight);await stream.waitForCurrent();
 visibleOwner='3';stream.update(0,20,20,0,true,sight);await settle();
 assert.deepEqual(loads,['1','3']);
 stream.update(0,22,20,30,true,sight);await settle();
 assert.deepEqual(loads,['1','3','2']);assert.deepEqual(stream.status().active,['1']);
 releaseArrival();await settle();assert.deepEqual(new Set(stream.status().active),new Set(['1','2']));
 assert.deepEqual(loads,['1','3','2','3']);
 releaseFrontage();await settle();assert.deepEqual(new Set(stream.status().active),new Set(['2','3']));
 stream.update(0,56,20,30,true,sight);await settle();
 assert.deepEqual(stream.status().events.filter(e=>e.type==='crossing').map(e=>[e.id,e.ready]),[['2',true]]);
 stream.dispose();
});

test('high speed reserves a sparse next tile farther than the old 65m lookahead',async()=>{
 const tiles=[{...cells[0],tileEN:[0,0,55,55]},{...cells[1],tileEN:[165,0,220,55]}];
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 stream.update(0,5,20,0);await stream.waitForCurrent();
 stream.update(0,10,20,45);await settle();
 assert.deepEqual(stream.status().active,['0','1']);stream.dispose();
});

test('deferred diagonal arrivals both load before a changing optional frontage',async()=>{
 const tiles=[[0,0],[1,0],[0,1],[1,1]].map(([x,y],i)=>({...cells[i],tileEN:[55*x,55*y,55*(x+1),55*(y+1)]}));
 const loads:string[]=[],live=new Set<string>(),release=new Map<string,()=>void>();let owner='0',maximum=0;
 const stream=manageDriveCells(tiles,async(cell,signal)=>{
  loads.push(cell.id);
  if(cell.id!=='0')await new Promise<void>(resolve=>{release.set(cell.id,resolve);signal.addEventListener('abort',()=>resolve(),{once:true});});
  return {attach(){live.add(cell.id);maximum=Math.max(maximum,live.size);assert.ok(live.size<=3);},dispose(){live.delete(cell.id);}};
 });
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id===owner?0:Infinity};
 stream.update(0,37,35,0,false,sight);await stream.waitForCurrent();
 owner='2';stream.update(0,40,38,28,true,sight);await settle();
 assert.deepEqual(loads,['0','1']);assert.deepEqual(stream.status().active,['0']);
 release.get('1')!();await settle();assert.deepEqual(loads,['0','1','3']);
 assert.deepEqual(new Set(stream.status().active),new Set(['0','1']));
 release.get('3')!();await settle();assert.deepEqual(loads,['0','1','3','2']);
 assert.deepEqual(new Set(stream.status().active),new Set(['0','1','3']));
 release.get('2')!();await settle();assert.deepEqual(new Set(stream.status().active),new Set(['1','2','3']));
 for(const [e,n] of [[55.5,53.5],[58,56]]){stream.update(0,e,n,28,true,sight);await settle();}
 assert.deepEqual(stream.status().events.filter(e=>e.type==='crossing').map(e=>[e.id,e.ready]),[['1',true],['3',true]]);
 assert.equal(maximum,3);assert.ok(stream.status().active.length<=2);stream.dispose();
});

test('identical parked inputs reuse frontage planning while camera changes and movement invalidate it',async()=>{
 const tiles=Array.from({length:313},(_,i)=>({...cells[0],id:String(i),tileEN:[55*i,0,55*(i+1),55]}));
 let scores=0;const sight={cameraStation:0,lookStation:0,revision:1,visible:()=>true,frontageScore:(c:DriveCell)=>{scores++;return c.id==='0'?0:Infinity;}};
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){}}));
 stream.update(0,20,20,0,false,sight);await stream.waitForCurrent();
 stream.update(0,20,20,0,true,sight);scores=0;
 for(let frame=0;frame<300;frame++)stream.update(0,20,20,0,true,sight);
 assert.equal(scores,0);
 sight.revision++;stream.update(0,20,20,0,true,sight);assert.equal(scores,313);
 stream.update(0,21,20,10,true,sight);assert.equal(scores,626);
 stream.update(0,22,20,10,true,sight);assert.equal(scores,939);
 stream.dispose();
});


function parkedRetirementFixture(restore=true) {
 const tiles=cells.slice(0,2).map((c,i)=>({...c,tileEN:[i*55,0,(i+1)*55,55]}));
 let owner='0';const live=new Set<string>(),disposed:string[]=[],restored:string[]=[],fades:{id:string;finish:()=>void}[]=[];
 const stream=manageDriveCells(tiles,async cell=>({
  attach(){live.add(cell.id);assert.ok(live.size<=1);},dispose(){disposed.push(cell.id);live.delete(cell.id);},
  retire(){return new Promise<void>(resolve=>fades.push({id:cell.id,finish:resolve}));},
  ...(restore?{restore(){restored.push(cell.id);}}:{}),
 }));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id===owner?0:Infinity};
 const select=(id:string)=>{owner=id;stream.update(0,20,20,0,false,sight);};
 return {stream,live,disposed,restored,fades,select};
}

test('a fading resident stays counted until safe completion and blocks replacement attachment',async()=>{
 const f=parkedRetirementFixture();f.select('0');await f.stream.waitForCurrent();
 f.select('1');await settle();
 assert.deepEqual(f.stream.status().active,['0']);assert.deepEqual(f.stream.status().retiring,['0']);
 assert.deepEqual([...f.live],['0']);assert.deepEqual(f.disposed,[]);assert.equal(f.stream.status().loading,'1');
 f.fades[0].finish();await settle();
 assert.deepEqual(f.stream.status().active,['1']);assert.deepEqual(f.disposed,['0']);assert.deepEqual(f.stream.status().retiring,[]);
 f.stream.dispose();assert.deepEqual(f.disposed,['0','1']);
});

test('returning desire restores a resident and stale fade completion cannot dispose a newer fade',async()=>{
 const f=parkedRetirementFixture();f.select('0');await f.stream.waitForCurrent();
 f.select('1');await settle();const stale=f.fades[0];
 f.select('0');await settle();assert.deepEqual(f.restored,['0']);assert.deepEqual(f.stream.status().retiring,[]);
 assert.deepEqual(f.stream.status().active,['0']);assert.ok(f.stream.status().ready);
 f.select('1');await settle();assert.equal(f.fades.length,2);
 stale.finish();await settle();assert.deepEqual(f.stream.status().active,['0']);assert.deepEqual(f.stream.status().retiring,['0']);
 assert.equal(f.disposed.filter(id=>id==='0').length,0);
 f.fades[1].finish();await settle();assert.deepEqual(f.stream.status().active,['1']);
 f.stream.dispose();
});

test('disposal force-cleans retiring and waiting resources exactly once',async()=>{
 const f=parkedRetirementFixture();f.select('0');await f.stream.waitForCurrent();f.select('1');await settle();
 f.stream.dispose();await settle();assert.deepEqual(f.stream.status().active,[]);assert.deepEqual(f.stream.status().retiring,[]);
 assert.deepEqual(f.disposed.sort(),['0','1']);
 f.fades[0].finish();await settle();assert.deepEqual(f.disposed.sort(),['0','1']);
});

test('a resource without restore finishes retirement before being safely loaded again',async()=>{
 const f=parkedRetirementFixture(false);f.select('0');await f.stream.waitForCurrent();f.select('1');await settle();
 f.select('0');await settle();assert.equal(f.stream.status().ready,false);
 f.fades[0].finish();await f.stream.waitForCurrent();
 assert.deepEqual(f.stream.status().active,['0']);assert.equal(f.disposed.filter(id=>id==='0').length,1);f.stream.dispose();
});

test('corridor retirement retains the two-slot cap while the third resource waits',async()=>{
 const live=new Set<string>(),fades=new Map<string,()=>void>();
 const stream=manageDriveCells(cells,async cell=>({attach(){live.add(cell.id);assert.ok(live.size<=2);},dispose(){live.delete(cell.id);},
  retire(){return new Promise<void>(resolve=>fades.set(cell.id,resolve));},restore(){}}));
 stream.update(0,0,0,0);await stream.waitForCurrent();stream.update(80,80,0,16);await settle();
 assert.deepEqual(new Set(stream.status().active),new Set(['0','1']));
 stream.update(190,190,0,16);await settle();assert.deepEqual(new Set(stream.status().active),new Set(['0','1']));
 assert.deepEqual(stream.status().retiring,['0']);assert.equal(stream.status().loading,'2');
 fades.get('0')!();await settle();assert.deepEqual(new Set(stream.status().active),new Set(['1','2']));stream.dispose();
});

test('a failed retirement preserves visible resources without retrying in a microtask loop',async()=>{
 const tiles=cells.slice(0,2).map((c,i)=>({...c,tileEN:[55*i,0,55*(i+1),55]}));
 let owner='0',attempts=0,disposals=0;
 const stream=manageDriveCells(tiles,async()=>({attach(){},dispose(){disposals++;},retire:async()=>{attempts++;throw Error('fade failed');},restore(){}}));
 const sight={cameraStation:0,lookStation:0,visible:()=>true,frontageScore:(c:DriveCell)=>c.id===owner?0:Infinity};
 stream.update(0,20,20,0,false,sight);await stream.waitForCurrent();owner='1';stream.update(0,20,20,0,false,sight);await settle();
 assert.equal(attempts,1);assert.equal(disposals,0);assert.deepEqual(stream.status().active,['0']);
 assert.match(stream.status().events.findLast(e=>e.type==='error')!.message!,/Retirement failed/);
 stream.dispose();await settle();assert.equal(disposals,2);
});

test('optional facadeMiddle accepts only local comparison JSON assets',()=>{
 const pkg={contentHash:'0123456789ab',format:'gzip-chunks' as const,chunks:['/world-0123456789ab-00.bin'],compressedBytes:10,uncompressedBytes:10};
 const doc={scope:'neighborhood-stream',version:1,sourceWorldHash:'source',world:'/compare/world.json',base:pkg,routeEN:[[0,0],[55,0]],
  cells:[{...pkg,id:'0',tileEN:[0,0,55,55],boundsEN:[0,0,55,55]}]} as unknown as DriveStream;
 assert.equal(validateDriveStream({...doc,facadeMiddle:'/compare/facade-middle-v1.json'}).facadeMiddle,'/compare/facade-middle-v1.json');
 assert.equal(validateDriveStream(doc),doc);
 for(const facadeMiddle of ['https://example.com/middle.json','/compare/../middle.json','/other/middle.json','/compare/middle.bin',null,42])
  assert.throws(()=>validateDriveStream({...doc,facadeMiddle} as DriveStream),/Invalid middle facade/);
});
