// A repeatable native-1080p drive through the public viewer. Supply a local
// route fixture in east/north metres and run this command inside guard_memory.py.
import {geometryMemory,sampleGpu,startPresentation} from './circuit-profile.mjs';
import {spawn} from 'node:child_process';
import {mkdtemp,rm} from 'node:fs/promises';
import {setTimeout as delay} from 'node:timers/promises';
import {tmpdir} from 'node:os';
import {browser as cdp} from './stream-cdp.mjs';
import {readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';

const [origin, fixturePath, reportPath] = process.argv.slice(2);
if (!process.env.SF_MEMORY_GUARDED || !reportPath) throw Error('Run inside guard_memory.py with an origin, route and report path.');
const url = new URL(origin);
if (!['localhost','127.0.0.1','[::1]'].includes(url.hostname)) throw Error('Expected a local game server');
let fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
const namedManifest=url.searchParams.get('manifest');
if(namedManifest){
  url.searchParams.set('profileStream','1');
  if(fixture.spatialIndex){
    const indexUrl=fixture.spatialIndex;
    if(!/^\/compare\/[\w.-]+\.json$/.test(indexUrl))throw Error('Expected local immutable stream index');
    const diskIndex=JSON.parse(await readFile(path.join(path.dirname(fixturePath),path.basename(indexUrl)),'utf8'));
    if(diskIndex.sourceWorldHash!==fixture.contentHash)throw Error('On-disk archive and index disagree');
    fixture=diskIndex;
  }
  if(!/^\/compare\/[a-z0-9-]+\.json$/.test(namedManifest)||!fixture.driveFromRouteStart||fixture.scope!=='neighborhood-stream')throw Error('Expected the named full-area stream index as fixture');
  const route=fixture.routeEN;
  let junctions=[];
  try{junctions=JSON.parse(await readFile(reportPath.replace(/\.json$/,'')+'.junctions.json','utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  if(!junctions.length)throw Error('Named area requires mapped street junctions for crossing verification');
  fixture={junctions,sourceWorldHash:fixture.sourceWorldHash,spawn:route[0],centerlineEN:route,followPolyline:true,
    plannedDistanceM:route.slice(1).reduce((s,p,i)=>s+Math.hypot(p[0]-route[i][0],p[1]-route[i][1]),0)};
}
// This controller only supplies steering, pedals and brake. All motion, turns,
// camera work, collisions, tile selection and loading go through the game.
function beginDrive({centerlineEN, cornerRadiusM, followPolyline=false, junctions=[], visibilityAudit=false, captureTurns=false}) {
  const way = [], turnPoints=[], clamp = (x, lo, hi) => Math.max(lo, Math.min(x, hi));
  const distance = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
  function toward(a, b, m) {
    const t = m / distance(a,b);
    return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])];
  }
  function line(a, b, corner=false) {
    const d = distance(a,b), n = Math.ceil(d/1.4);
    for (let i=1; i<=n; i++) way.push({p:toward(a,b,d*i/n), corner});
  }
  let prev = centerlineEN[0];
  way.push({p:prev, corner:false});
  for (let i=1; i<centerlineEN.length-1; i++) {
    if(followPolyline){line(prev,centerlineEN[i]);prev=centerlineEN[i];continue;}
    const turn = centerlineEN[i], a=toward(turn, centerlineEN[i-1], cornerRadiusM), b=toward(turn, centerlineEN[i+1], cornerRadiusM);
    line(prev,a); const entryIndex=way.length-1; let last=a, apexIndex;
    for (let j=1; j<=14; j++) {
      const t=j/14, s=1-t, q=a.map((v,k)=>s*s*v+2*s*t*turn[k]+t*t*b[k]);
      line(last,q,true); last=q; if(j===7)apexIndex=way.length-1;
    }
    turnPoints.push({turn:i,index:apexIndex,entryIndex,exitIndex:way.length-1,positionEN:turn});
    prev=b;
  }
  line(prev,centerlineEN.at(-1));
  const stations=way.map((w,i)=>i?distance(w.p,way[i-1].p):0);
  for(let i=1;i<stations.length;i++)stations[i]+=stations[i-1];
  for(const t of turnPoints){t.startIndex=stations.findIndex(m=>m>=stations[t.entryIndex]-40);t.endIndex=stations.findIndex(m=>m>=stations[t.exitIndex]+10);if(t.endIndex<0)t.endIndex=way.length-1;}
  const view=window.__view, car=window.__car.car, update=car.update.bind(car);
  let idx=0, old=car.position(), mileage=0, maxError=0, done=false;
  const startMs=performance.mark('sf-circuit-start').startTime, timeOriginEpochMs=performance.timeOrigin;
  let prevFrame, worst=0, native=true,finishPosition;
  const junctionEvents=[];let nextJunction=0;
  const frames=[],turns=[],slowFrames=[],dimensions=new Map();
  const visibilitySamples=[],visibilitySpans=new Map();let visibilityAt=startMs;
  const stats=values=>{const a=values.slice().sort((a,b)=>a-b),sum=a.reduce((x,y)=>x+y,0);return {frames:a.length,fps:a.length?1000*a.length/sum:null,p50Ms:a[Math.ceil(a.length*.5)-1]??null,p95Ms:a[Math.ceil(a.length*.95)-1]??null,p99Ms:a[Math.ceil(a.length*.99)-1]??null,longestMs:a.at(-1)??null,over25:a.filter(x=>x>25).length};};
  const windowStats=ms=>stats(frames.filter(f=>f[0]>=ms-500&&f[0]<=ms+500).map(f=>f[1]));
  const rangeStats=(a,b)=>stats(frames.filter(f=>f[0]>=a&&f[0]<=b).map(f=>f[1]));
  const turnSnapshot=t=>({...t,complete:t.endMs!==undefined,frames:rangeStats(t.startMs-500,(t.endMs??performance.now())+500),actualTurnFrames:rangeStats(t.entryMs??Infinity,t.exitMs??performance.now()),brakingFrames:rangeStats(t.startMs,t.entryMs??performance.now()),egressFrames:rangeStats(t.exitMs??Infinity,t.endMs??performance.now()),apexFrames:t.ms===undefined?null:windowStats(t.ms)});
  const snapshot=()=>({profile:window.__circuitProfile?.(),complete:done,courseProgress:{waypoint:idx,totalWaypoints:way.length,completedTurns:turns.filter(t=>t.endMs!==undefined).length},mileage,maxError,native,startMs,endMs:performance.now(),timeOriginEpochMs,frameClock:"performance.now (navigation epoch), same as drive-stream events",frameWindowMs:500,slowFrames,streetCrossings:junctionEvents.map(e=>({...e,frames:windowStats(e.ms)})),graphics:{allocations:view.allocations?.(),programs:view.renderer.info.programs?.length,...view.renderer.info.memory,jsHeapMiB:performance.memory?performance.memory.usedJSHeapSize/(1024*1024):undefined},scenery:view.scenery?.(),preparation:view.streamTimings(),longFrames:window.__circuitLongFrames,...stats(intervals),dimensions:[...dimensions.values()],turns:turns.map(turnSnapshot),crossings:[...crossings.values()].map(e=>({...e,frames:windowStats(e.ms)})),visibility:visibilityAudit?{source:visibilitySamples.some(s=>s.unavailable)?'unavailable':'runtime frontageCoverage',samplePeriodMs:250,owners:visibilitySamples,spans:[...visibilitySpans.values()]}:undefined});
  const intervals=[], draws=[], crossings=new Map();
  car.update=(dt,input)=>{
    const [x,z,yaw]=car.position(), speed=car.debug().u;
    if (done) return update(dt,{...input,throttle:0,brake:speed>.1?1:0,steer:0});
    mileage+=Math.hypot(x-old[0],z-old[1]); old=[x,z,yaw];
    const dist = k => Math.hypot(way[k].p[0]-x,way[k].p[1]+z);
    let nearest=Infinity;
    const end=Math.min(way.length,idx+20);
    for (let k=idx; k<end; k++) {
      const d=dist(k);
      if (d<nearest) {nearest=d; idx=k;} else if (d>nearest+2) break;
    }
    while(nextJunction<junctions.length&&stations[idx]>=junctions[nextJunction].stationM)junctionEvents.push({...junctions[nextJunction++],ms:performance.now()});
    maxError=Math.max(maxError,nearest);
    for(const t of turnPoints){
      let record=turns.find(v=>v.turn===t.turn);
      if(!record&&idx>=t.startIndex){record={...t,startMs:performance.now(),startMileage:mileage};turns.push(record);}
      if(record)for(const [index,key] of [[t.entryIndex,'entryMs'],[t.index,'ms'],[t.exitIndex,'exitMs'],[t.endIndex,'endMs']])if(idx>=index&&record[key]===undefined){record[key]=performance.now();if(key==='endMs')record.endMileage=mileage;if(key==='ms'&&captureTurns)void window.__saveCircuitTurn(t.turn).catch(()=>{});}
    }
    if (idx>=way.length-2) {done=true;finishPosition=car.position().slice();return update(dt,{...input,throttle:0,brake:1,steer:0});}
    const look=clamp(2.6+speed*.30,3.3,10);
    let aim=idx;
    while (aim+1<way.length && dist(aim)<look) aim++;
    const [tx,tn]=way[aim].p, delta=Math.atan2(x-tx,tn+z)-yaw;
    const error=Math.atan2(Math.sin(delta),Math.cos(delta));
    const steer=clamp(Math.atan(2*2.48*Math.sin(error)/Math.max(2,dist(aim)))/(.57/(1+(speed/11)**2)),-1,1);
    let target=18;
    for (let k=idx;k<Math.min(idx+50,way.length);k++) if(way[k].corner){
      target=Math.min(target,Math.sqrt(5.3**2+2*4.3*Math.max(0,dist(k)-4))); break;
    }
    if (Math.abs(error)>.4) target=Math.min(target,5.3);
    return update(dt,{steer,handbrake:false,throttle:clamp((target-speed)*.33+.09,0,1),brake:clamp((speed-target-.4)*.22,0,1)});
  };
  return new Promise(resolve=>{
    function frame(now){
      if(prevFrame!==undefined){const interval=now-prevFrame;intervals.push(interval);frames.push([now,interval]);worst=Math.max(worst,interval);
        if(interval>25){
          const status=view.stream.status(),turn=turns.find(t=>t.startMs<=now&&(t.endMs??Infinity)>=now);
          const recentPreparation=view.streamTimings().filter(e=>e.startMs<=performance.now()&&e.startMs+e.durationMs>=prevFrame-40).slice(-20);
          slowFrames.push({recentPreparation,endMs:now,intervalMs:interval,stationM:mileage,tile:status.current,loading:status.loading,turn:turn?.turn??null});
          if(slowFrames.length>128){slowFrames.sort((a,b)=>b.intervalMs-a.intervalMs);slowFrames.pop();}
        }}
      prevFrame=now;
      if(visibilityAudit&&now-visibilityAt>=250){
        const elapsed=Math.min(1000,now-visibilityAt);visibilityAt=now;
        const coverage=typeof view.frontageCoverage==='function'?view.frontageCoverage():null;
        if(coverage?.owners){
          const owners=coverage.owners.map(o=>({id:o.id,cell:o.cell,nearM:o.nearM,resident:!!o.resident,detailVisible:!!o.detailVisible,middleVisible:!!o.middleVisible,exteriorVisible:!!o.exteriorVisible,windowCoverage:o.windowCoverage}));
          visibilitySamples.push({atMs:now,mileageM:mileage,owners,correctedExteriorFaces:coverage.correctedExteriorFaces??0});
          for(const o of owners){
            const span=visibilitySpans.get(o.id)??{id:o.id,cell:o.cell,coarseOnlyMs:0,missingMs:0,maxCoarseOnlyMs:0,maxMissingMs:0,coarseOnlyRunMs:0,missingRunMs:0,coarseOnlySamples:0,missingSamples:0};
            if(o.exteriorVisible&&!o.middleVisible&&!o.detailVisible){span.coarseOnlyMs+=elapsed;span.coarseOnlyRunMs+=elapsed;span.maxCoarseOnlyMs=Math.max(span.maxCoarseOnlyMs,span.coarseOnlyRunMs);span.coarseOnlySamples++;}else span.coarseOnlyRunMs=0;
            if(!o.exteriorVisible&&!o.middleVisible&&!o.detailVisible){span.missingMs+=elapsed;span.missingRunMs+=elapsed;span.maxMissingMs=Math.max(span.maxMissingMs,span.missingRunMs);span.missingSamples++;}else span.missingRunMs=0;
            span.cell=o.cell;span.lastSeenMs=now;visibilitySpans.set(o.id,span);
          }
          for(const span of visibilitySpans.values())if(now-span.lastSeenMs>750){span.coarseOnlyRunMs=0;span.missingRunMs=0;}
        }else visibilitySamples.push({atMs:now,mileageM:mileage,unavailable:true});
      }
      const scene=view.composer.passes[0].target, canvas=view.renderer.domElement;
      const dims={scene:[scene.width,scene.height],canvas:[canvas.width,canvas.height],css:[canvas.clientWidth,canvas.clientHeight],viewport:[innerWidth,innerHeight],dpr:devicePixelRatio};dimensions.set(JSON.stringify(dims),dims);
      native &&= scene.width===1920&&scene.height===1080&&canvas.width===1920&&canvas.height===1080;
      if (intervals.length%60===0 || done) {
        draws.push(view.info().calls);
        for(const ev of view.stream.status().events)if(ev.type==='crossing'&&ev.ms>=startMs)crossings.set(ev.ms+':'+ev.id,ev);
      }
      if(!done){if(intervals.length%120===0)window.__checkpoint=snapshot();requestAnimationFrame(frame);return;}
      performance.mark('sf-circuit-end');
      resolve({...snapshot(),medianDraws:draws.sort((a,b)=>a-b)[Math.floor(draws.length/2)],finishPosition,endpointErrorM:distance([finishPosition[0],-finishPosition[1]],centerlineEN.at(-1)),course:{plannedDistanceM:way.reduce((sum,w,i)=>i?sum+distance(w.p,way[i-1].p):0,0),centerlineDistanceM:centerlineEN.reduce((sum,p,i)=>i?sum+distance(p,centerlineEN[i-1]):0,0),plannedTurns:turnPoints.length,completedTurns:turns.filter(t=>t.endMs!==undefined).length,finalWaypoint:idx,totalWaypoints:way.length}});
    }
    requestAnimationFrame(frame);
  });
}
const dir=await mkdtemp(path.join(tmpdir(),'sf-drive-profile-'));
const launchEpochMs=Date.now();
const chrome=spawn(process.env.CHROME_EXECUTABLE||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-background-networking','--no-first-run','--disable-extensions','--disable-sync','--no-default-browser-check','--disable-component-update','--mute-audio','--use-angle=metal','--enable-gpu','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+dir,'about:blank'],{stdio:['ignore','ignore','pipe']});
let debugUrl;
chrome.stderr.on('data', chunk=>{
  const match=chunk.toString().match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/devtools\/browser\/[^\s]+)/);
  if(match)debugUrl=match[1];
});
let browser,checkpoint;
let metadata={};
try {
  let ws;for(let i=0;i<150&&!ws;i++){if(debugUrl){ws=debugUrl;break;}try{const [port,endpoint]=(await readFile(path.join(dir,'DevToolsActivePort'),'utf8')).trim().split('\n');ws='ws://127.0.0.1:'+port+endpoint;}catch{await delay(50);}}
  if(!ws)throw Error('The background Chrome process did not provide its debugging endpoint');
  browser=await cdp.connect(ws);const context=browser.contexts()[0],page=context.pages()[0];const session=await context.newCDPSession(page);
  await session.send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
  const {gpu}=await context.connection.send('SystemInfo.getInfo');
  const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  const refresh=await page.evaluate(()=>new Promise(resolve=>{let last;const a=[];function tick(now){if(last!==undefined)a.push(now-last);last=now;if(a.length<132)return requestAnimationFrame(tick);const v=a.slice(12).sort((x,y)=>x-y);resolve({samples:v.length,medianMs:v[Math.floor(v.length/2)],p95Ms:v[Math.ceil(v.length*.95)-1],fps:1000*v.length/v.reduce((x,y)=>x+y,0),visible:document.visibilityState==='visible'});}requestAnimationFrame(tick);}));
  metadata={browser:browser.version(),renderer:gpu.auxAttributes.glRenderer,refresh};
  if(!metadata.renderer||/swiftshader|llvmpipe|software rasterizer/i.test(metadata.renderer))throw Error('Hardware graphics unavailable for the full-map drive');
  if(!refresh.visible||refresh.fps<57||refresh.fps>63||refresh.p95Ms>19)throw Error('Browser did not verify 60 Hz before drive');
  await writeFile(reportPath+'.partial',JSON.stringify({...metadata,stage:'loading full map'}));
  await page.addInitScript(()=>{performance.setResourceTimingBufferSize(2048);const check=()=>{if(window.__car?.car?.group.visible&&!document.querySelector('.load-notice'))window.__circuitReadyMs=performance.now();else requestAnimationFrame(check);};requestAnimationFrame(check);});
  await page.addInitScript(()=>{
    const records=window.__circuitLongFrames=[];
    if(!PerformanceObserver.supportedEntryTypes.includes('long-animation-frame'))return;
    new PerformanceObserver(list=>{for(const frame of list.getEntries()){
      records.push({at:frame.startTime,ms:frame.duration,renderStart:frame.renderStart,blockingMs:frame.blockingDuration,
        scriptMs:frame.scripts.reduce((sum,s)=>sum+s.duration,0)});
      if(records.length>100)records.shift();
    }}).observe({type:'long-animation-frame'});
  });
  url.searchParams.set('drive','1');
  url.searchParams.set('measureStartup','1');
  if(url.searchParams.has('stream')&&url.searchParams.get('stream')!=='1'||url.pathname!=='/')throw Error('Expected the regular full-map drive entry');
  await page.goto(url.href);
  let pollingStartup=false;
  checkpoint=setInterval(async()=>{
    if(pollingStartup)return;
    pollingStartup=true;
    try{
      const startup=await page.evaluate(()=>({
        navigationMs:performance.now(), phases:performance.getEntriesByType('mark').filter(e=>e.name.startsWith('world.')).map(e=>({stage:e.name.slice(6),navigationMs:e.startTime})),
        heapMiB:performance.memory?.usedJSHeapSize/(1024*1024), graphics:window.__view?.renderer?.info?.memory,
        loading:document.querySelector('.load-notice')?.textContent, resources:performance.getEntriesByType('resource').slice(-10).map(e=>({name:new URL(e.name).pathname,bytes:e.transferSize}))}));
      await writeFile(reportPath+'.startup',JSON.stringify({...metadata,stage:'starting',startup,errors}));
    }catch{}finally{pollingStartup=false;}
  },1000);checkpoint.unref();
  await page.waitForFunction(()=>window.__car?.car?.group.visible&&window.__circuitReadyMs!==undefined&&!document.querySelector('.load-notice'),null,{timeout:60000});
  clearInterval(checkpoint);
  metadata.startup=await page.evaluate(()=>({readyMs:window.__circuitReadyMs,timeOriginEpochMs:performance.timeOrigin,
    phases:performance.getEntriesByType('mark').filter(e=>e.name.startsWith('world.')).map(e=>({stage:e.name.slice(6),navigationMs:e.startTime}))}));
  metadata.startup.launchToReadyMs=metadata.startup.timeOriginEpochMs+metadata.startup.readyMs-launchEpochMs;
  await writeFile(reportPath+'.partial',JSON.stringify({...metadata,stage:'car ready',errors}));
  metadata.qualityMode=process.env.SF_CIRCUIT_NATIVE_DEFAULT==='1'?'normal player settings':'locked native';
  if(metadata.qualityMode==='locked native')
    await page.evaluate(()=>{window.__view.lockQuality(true);window.__view.quality(1,true);});
  await delay(800);
  metadata.nativeBaseline=await page.evaluate(()=>{const v=window.__view,c=v.renderer.domElement,t=v.composer.passes[0].target;return {canvas:[c.width,c.height],scene:[t.width,t.height],viewport:[innerWidth,innerHeight],dpr:devicePixelRatio};});
  if([...metadata.nativeBaseline.canvas,...metadata.nativeBaseline.scene].some((v,i)=>v!==[1920,1080,1920,1080][i]))throw Error("Native 1080p lock failed before drive");
  const scope=await page.evaluate(async named=>{
    const assets=performance.getEntriesByType('resource').map(e=>new URL(e.name).pathname);
    const archive=named?await(await fetch(named)).json():null;
    const index=archive?.spatialIndex??'/compare/neighborhood-stream.json';
    const m=await(await fetch(index)).json();
    const worldResources=m.compactTerrain?[m.compactTerrain.world,m.compactTerrain.ground]:[m.world];
    const loadedWorld=worldResources.every(path=>typeof path==='string'&&assets.includes(path));
    if(named&&(!assets.includes(named)||!assets.includes(index)||!loadedWorld||archive.contentHash!==m.sourceWorldHash))throw Error('Named area archive/index/terrain proof failed');
    const active=window.__view.stream?.status().active??[]; const ids=new Set(m.cells.map(c=>c.id));
    return {scope:m.scope,index,worldResources,loadedWorld,hash:m.sourceWorldHash,cells:m.cells.length,buildings:m.base.stats.buildings,wallCorrections:m.windingCorrections??null,active,belongsToManifest:active.length>0&&active.every(id=>ids.has(id)),position:window.__car.car.position(),
      bundle:document.querySelector('script[type="module"][src]')?.getAttribute('src'),
      loadedBase:m.base.chunks.some(path=>assets.includes(path))||m.base.gpu?.chunks.some(path=>assets.includes(path)),
      parkedTextures:[...new Set(assets.filter(path=>path.includes('/vehicle/parked/atlas-')))]};
  },namedManifest);
  if(scope.scope!=='neighborhood-stream'||scope.hash!==fixture.sourceWorldHash||!scope.belongsToManifest||!scope.loadedBase)throw Error('The live scene did not load the expected full map');
  if(Math.hypot(scope.position[0]-fixture.spawn[0],scope.position[1]+fixture.spawn[1])>2)throw Error('Unexpected spawn');
  metadata.scope=scope;
  await writeFile(reportPath+'.partial',JSON.stringify({...metadata,stage:'verified full map',errors}));
  const observed=[];
  let saving=false;
  checkpoint=setInterval(async()=>{
    if(saving)return;
    saving=true;
    try{
      const s=await page.evaluate(offset=>{
        const point=window.__checkpoint;
        if(!point?.visibility)return point;
        return {...point,visibility:{...point.visibility,owners:point.visibility.owners.slice(offset)}};
      },observed.length);
      if(!s)return;
      if(s.visibility){observed.push(...s.visibility.owners);s.visibility.owners=observed;}
      await writeFile(reportPath+'.partial',JSON.stringify({...metadata,...s,errors}));
    }catch{/* The guard can close the page between checkpoints. */}
    finally{saving=false;}
  },2000);checkpoint.unref();
  if(process.env.SF_CIRCUIT_GPU==='1'){
    metadata.parkedGeometryMemory=await page.evaluate(geometryMemory);
    await page.evaluate(sampleGpu);
  }
  const visibilityAudit=process.env.SF_CIRCUIT_VISIBILITY==='1';
  const captureFrames=visibilityAudit&&process.env.SF_CIRCUIT_SCREENSHOTS!=='0';
  const saveScreenshot=async file=>{const shot=await session.send('Page.captureScreenshot',{format:'png',fromSurface:true,captureBeyondViewport:false});await writeFile(file,Buffer.from(shot.data,'base64'));};
  if(captureFrames){
    metadata.visibilityScreenshots={initial:reportPath+'.visibility-initial.png',postDrive:reportPath+'.visibility-post-drive.png'};
    await saveScreenshot(metadata.visibilityScreenshots.initial);
  }
  const captureTurns=process.env.SF_CIRCUIT_TURN_IMAGES==='1';
  if(captureTurns){
    metadata.turnImages=[];
    await page.exposeFunction('__saveCircuitTurn',async turn=>{
      if(!Number.isInteger(turn)||turn<1||turn>4)return;
      const file=reportPath+'.turn-'+turn+'.jpg';
      const shot=await session.send('Page.captureScreenshot',{format:'jpeg',quality:84,fromSurface:true,captureBeyondViewport:false});
      await writeFile(file,Buffer.from(shot.data,'base64'));metadata.turnImages.push({turn,file});
    });
  }
  const finishTrace=process.env.SF_CIRCUIT_PRESENT==='1'?await startPresentation(context.connection):namedManifest?await startPresentation(context.connection,{mode:'display'}):null;
  const [report]=await Promise.all([page.evaluate(beginDrive,{...fixture,visibilityAudit,captureTurns}),session.send('Input.dispatchKeyEvent',{type:'keyDown',key:'w',code:'KeyW',windowsVirtualKeyCode:87})]);
  clearInterval(checkpoint);
  if(finishTrace)report.compositor=await finishTrace(report);
  if(process.env.SF_CIRCUIT_GPU==='1')report.finalGeometryMemory=await page.evaluate(geometryMemory);
  await session.send('Input.dispatchKeyEvent',{type:'keyUp',key:'w',code:'KeyW',windowsVirtualKeyCode:87});
  if(captureFrames)await saveScreenshot(metadata.visibilityScreenshots.postDrive);
  if(url.searchParams.get('fps')==='1')report.meter=await page.locator('.drive-frame-meter').innerText();
  await writeFile(reportPath,JSON.stringify({...metadata,...report,errors},null,2));
  if(fixture.expectedCrossings&&JSON.stringify(report.crossings.map(e=>e.id))!==JSON.stringify(fixture.expectedCrossings))throw Error('Full circuit did not traverse the expected sequence of tiles');
  if(errors.length||!report.native||report.maxError>3||report.endpointErrorM>3||report.crossings.length<3||report.course.completedTurns!==(fixture.followPolyline?0:4)||report.course.plannedTurns!==(fixture.followPolyline?0:4)||report.turns.some(t=>!t.complete||!t.actualTurnFrames.frames||!t.brakingFrames.frames||!t.egressFrames.frames)||Math.abs(report.course.plannedDistanceM-(fixture.plannedDistanceM??960))>2||report.mileage<report.course.plannedDistanceM*.95||report.crossings.some(e=>!e.ready))throw Error('See native resolution, path, coverage and errors in report');
}catch(error){await writeFile(reportPath+'.failure',JSON.stringify({...metadata,error:String(error)},null,2));throw error;}finally{clearInterval(checkpoint);await browser?.close();chrome.kill('SIGTERM');await delay(300);await rm(dir,{recursive:true,force:true,maxRetries:3}).catch(()=>{});}
