// Isolated live multi-cell drive. Run only through guard_memory.py; never builds assets.
import {spawn} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {access, mkdir, mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {homedir, tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import path from 'node:path';
import net from 'node:net';
import {fileURLToPath} from 'node:url';
import {watchDriveFrames} from './stream-flash-review.mjs';

const [project, output] = process.argv.slice(2);
if (!project || !output || !process.env.SF_MEMORY_GUARDED) throw Error('Run inside the memory watchdog');
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || here, 'package.json'));
const connectBrowser=process.env.STREAM_DIRECT_CDP==='1'?(await import('./stream-cdp.mjs')).browser.connect:require('playwright').chromium.connectOverCDP.bind(require('playwright').chromium);
const neighborhood=process.env.STREAM_NEIGHBORHOOD==='1';
const workspace = path.resolve(project), manifestPath = neighborhood?'/compare/neighborhood-stream.json':'/compare/drive-stream.json';
const manifest = JSON.parse(await readFile(path.join(workspace, 'public', manifestPath)));
const point = p => Array.isArray(p) && p.length >= 2 && p.slice(0,2).every(Number.isFinite);
if (manifest.version !== 1 || manifest.scope !== (neighborhood?'neighborhood-stream':'drive-stream') || !Array.isArray(manifest.routeEN) ||
    manifest.routeEN.length < 2 || !manifest.routeEN.every(point) || !Array.isArray(manifest.cells) || manifest.cells.length < 4)
  throw Error('Invalid multi-cell drive package');
const allowed = new Set(), ids = new Set(), packages = new Map(), chunkPackages = new Map(), selectedProfiles = new Map();
function registerPackage(id,portable){
  const encodings=[{...portable,profile:'portable'}];
  if(portable.gpu){
    if(portable.gpu.profile!=='s3tc-v1'||portable.gpu.portableHash!==portable.contentHash)throw Error('Invalid GPU profile '+id);
    encodings.push(portable.gpu);
  }
  for(const encoding of encodings){
    if(!encoding.contentHash || !Array.isArray(encoding.chunks) || !encoding.chunks.length ||
       !Number.isFinite(encoding.compressedBytes) || encoding.compressedBytes<0 ||
       !Number.isFinite(encoding.uncompressedBytes) || encoding.uncompressedBytes<0 || encoding.uncompressedBytes>150*1024**2 ||
       !encoding.chunks.every(u=>typeof u==='string'&&u.startsWith('/world-'+encoding.contentHash+'-')&&!u.includes('..')&&!u.includes('?')))
      throw Error('Unsafe or oversized package '+id+' '+encoding.profile);
    for(const chunk of encoding.chunks){
      allowed.add(chunk);const owners=chunkPackages.get(chunk)||[];owners.push({id,profile:encoding.profile});chunkPackages.set(chunk,owners);
    }
  }
  packages.set(id,encodings);
}
if(!manifest.base)throw Error('Missing shared base package');registerPackage('$base',manifest.base);
for(const cell of manifest.cells){
  if(typeof cell.id!=='string'||cell.id==='$base'||ids.has(cell.id)||!cell.boundsEN)throw Error('Invalid stream cell '+cell.id);
  ids.add(cell.id);
}
const detailedPackages=manifest.pieces??manifest.cells, pieceParents=new Map();
if(!Array.isArray(detailedPackages)||!detailedPackages.length)throw Error('Missing detailed stream packages');
for(const detail of detailedPackages){
  if(typeof detail.id!=='string'||detail.id==='$base'||packages.has(detail.id)||!detail.boundsEN)throw Error('Invalid detail package '+detail.id);
  if(manifest.pieces&&!ids.has(detail.parentId))throw Error('Unknown piece parent '+detail.parentId);
  pieceParents.set(detail.id,manifest.pieces?detail.parentId:detail.id);registerPackage(detail.id,detail);
}
const slotLimit=neighborhood?3:2;
const maxActive=Number(process.env.STREAM_MAX_ACTIVE || slotLimit);
if(!Number.isInteger(maxActive)||maxActive<1||maxActive>slotLimit)throw Error('Invalid active cell limit for this drive');
const width=Number(process.env.STREAM_WIDTH || 1920),height=width*9/16;
if(![1920,1280].includes(width))throw Error('STREAM_WIDTH must be 1920 or diagnostic 1280');
const seconds = Number(process.env.STREAM_SECONDS || 95), manual = process.env.STREAM_MANUAL === '1';
const visualOnly=process.env.STREAM_VISUAL_ONLY==='1', visualDir=process.env.STREAM_VISUAL_DIR?path.resolve(process.env.STREAM_VISUAL_DIR):null;
if (!(seconds >= 10 && seconds <= 125)) throw Error('STREAM_SECONDS must be 10–125');
if(visualOnly&&manual)throw Error('STREAM_VISUAL_ONLY requires automatic steering; use STREAM_MANUAL for immediate control');
if(visualOnly&&!visualDir)throw Error('STREAM_VISUAL_ONLY requires STREAM_VISUAL_DIR');
const verify60=process.env.STREAM_VERIFY_60==='1',headed=manual;
async function browserExecutable(){
  if(process.env.CHROME_EXECUTABLE)return process.env.CHROME_EXECUTABLE;
  if(!manual&&process.platform==='darwin'){
    const cache=process.env.PLAYWRIGHT_BROWSERS_PATH||path.join(homedir(),'Library','Caches','ms-playwright');
    const builds=(await readdir(cache).catch(()=>[])).filter(name=>/^chromium_headless_shell-\d+$/.test(name))
      .sort((a,b)=>Number(b.split('-').at(-1))-Number(a.split('-').at(-1)));
    for(const build of builds){
      const bin=path.join(cache,build,'chrome-headless-shell-mac-'+(process.arch==='arm64'?'arm64':'x64'),'chrome-headless-shell');
      if(await access(bin).then(()=>true,()=>false))return bin;
    }
  }
  return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
}
const executable=await browserExecutable();
const [start] = manifest.routeEN;
const probe = net.createServer();
await new Promise(resolve => probe.listen(0,'127.0.0.1',resolve));
const port = probe.address().port; await new Promise(resolve=>probe.close(resolve));
const origin = 'http://127.0.0.1:'+port, dir = await mkdtemp(path.join(tmpdir(),'sf-stream-browser-'));
let server, chrome, browser, page, serverTail = '', report;
const failures=[], consoleErrors=[], gpuWarnings=[], received=new Set(), requests=[], requestedChunks=[];
const launchEpochMs = Date.now();
async function until(fn, ms) {
  const end = Date.now()+ms; let last;
  while (Date.now()<end) { try { const result=await fn(); if(result)return result; } catch(e){last=e;} await delay(100); }
  throw Error('Startup timeout '+(last?.message || serverTail.slice(-750)));
}
try {
  server=spawn('python3',[path.join(here,'../king-street/local-viewer/serve.py'),'--bundle',process.env.BUNDLE_DIR || path.join(workspace,'.local-viewer'),'--public',path.join(workspace,'public'),'--port',String(port)],
    {cwd:workspace,stdio:['ignore','pipe','pipe'],env:{...process.env,NODE_OPTIONS:'--max-old-space-size=512'}});
  for(const stream of [server.stdout,server.stderr])stream.on('data',data=>{serverTail=(serverTail+data.toString()).slice(-2048);});
  await until(async()=>{if(server.exitCode!==null)throw Error(serverTail);return (await fetch(origin+manifestPath)).ok;},20000);
  const shell=await fetch(origin+'/');if(!shell.ok)throw Error('Could not prepare page');await shell.text();
  chrome=spawn(executable,
    [headed?'--new-window':'--headless=new','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+dir,
    '--no-first-run','--no-default-browser-check','--disable-extensions','--disable-sync','--disable-component-update',
    '--disable-background-networking','--enable-precise-memory-info','--disable-breakpad','--mute-audio','--use-angle=metal','--enable-gpu','about:blank'],{stdio:'ignore'});
  const debug=await until(async()=>{const [p,uri]=(await readFile(path.join(dir,'DevToolsActivePort'),'utf8')).split('\n');return uri && 'ws://127.0.0.1:'+p+uri;},10000);
  browser=await connectBrowser(debug);
  const context=browser.contexts()[0];page=context.pages()[0];
  const cdp=await context.newCDPSession(page);
  await cdp.send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
  if(manual)await page.bringToFront();
  // An animated tab requests the display's active refresh rate on adaptive monitors.
  // Measure the actual browser animation clock, without overriding vsync or driving its clock.
  const refresh=await page.evaluate(async()=>{
    const strip=document.createElement('div');
    strip.style.cssText='height:12px;width:70px;background:#678';document.body.append(strip);
    let last=0;const intervals=[];
    await new Promise(resolve=>{
      const sample=now=>{strip.style.transform='translateX('+Math.round(now%250)+'px)';
        if(last)intervals.push(now-last);last=now;
        if(intervals.length>=132)resolve();else requestAnimationFrame(sample);};requestAnimationFrame(sample);
    });
    strip.remove();const stable=intervals.slice(12).sort((a,b)=>a-b);
    return {samples:stable.length,medianMs:stable[Math.floor(stable.length/2)],p95Ms:stable[Math.ceil(.95*stable.length)-1],
      fps:1000*stable.length/stable.reduce((a,b)=>a+b,0),visible:document.visibilityState==='visible'};
  });
  const idleDisplayIntervalMs=refresh.medianMs;refresh.browser=browser.version();refresh.executable=path.basename(executable);
  console.log(JSON.stringify({refresh,headed,width}));
  if(verify60&&(!refresh.visible||refresh.fps<57||refresh.p95Ms>19)){
    report={scope:'browser-refresh',headed,width,refresh,failures:['Browser did not sustain a 60 Hz animation cadence before navigation']};
    await writeFile(output,JSON.stringify(report));throw Error(report.failures[0]);
  }
  await page.route('**/*',async route=>{
    const url=new URL(route.request().url());
    if(url.origin!==origin && ['http:','https:'].includes(url.protocol)){failures.push('external asset '+url.origin+url.pathname);return route.abort();}
    if((url.pathname.startsWith('/world-')&&!allowed.has(url.pathname)) || /^\/block-trial\/(photo-mesh-|block-splats-)/.test(url.pathname)){
      failures.push('outside stream allowlist '+url.pathname);return route.abort();
    }
    for(const owner of chunkPackages.get(url.pathname)||[]){const prior=selectedProfiles.get(owner.id);if(prior&&prior!==owner.profile){failures.push('Mixed encodings for '+owner.id+': '+prior+' and '+owner.profile);return route.abort();}selectedProfiles.set(owner.id,owner.profile);}
    return route.continue();
  });
  page.on('pageerror', e=>{if(failures.length<4)console.error(e.message);failures.push(e.message);});
  page.on('console',m=>{if(m.type()==='error'){if(consoleErrors.length<3)console.error(m.text().slice(-650));consoleErrors.push(m.text().slice(-450));}});
  cdp.on('Log.entryAdded',({entry})=>{
    if(/(?:GL_INVALID|feedback loop|framebuffer.*incomplete|context lost|shader error)/i.test(entry.text)&&gpuWarnings.length<12)gpuWarnings.push(entry.text.slice(0,450));
  });
  await cdp.send('Log.enable');
  page.on('requestfailed',r=>failures.push('request failed '+new URL(r.url()).pathname+' '+r.failure()?.errorText));
  page.on('request',r=>{const p=new URL(r.url()).pathname;if(allowed.has(p))requestedChunks.push({path:p,epochMs:Date.now()});});
  page.on('response',r=>{const p=new URL(r.url()).pathname;if(allowed.has(p)){if(r.ok())received.add(p);requests.push({path:p,status:r.status(),epochMs:Date.now()});}else if(r.status()>=400)failures.push('HTTP '+r.status()+' '+p);});
  const url=new URL('/',origin);url.search=new URLSearchParams({stream:neighborhood?'neighborhood':'corridor',drive:'1'});
  if(neighborhood&&!manual){
    const next=manifest.routeEN[1],heading=Math.atan2(next[0]-start[0],next[1]-start[1])*180/Math.PI;
    url.searchParams.set('driveFrom',[...start.slice(0,2),heading].join(','));
  }
  if(width===1280)url.searchParams.set('display','720');
  await page.addInitScript(()=>{const wait=()=>{if(window.__view?.stream?.status()?.ready && window.__car?.car?.group.visible && !document.querySelector('.load-notice'))window.__streamDriveReadyMs=performance.now();else requestAnimationFrame(wait);};requestAnimationFrame(wait);});
  await page.goto(url.href,{waitUntil:'domcontentloaded',timeout:20000});
  try{await page.waitForFunction(()=>window.__streamDriveReadyMs,null,{timeout:35000});}
  catch(error){throw Error('Drive never became ready: '+(await page.locator('body').innerText()).slice(-500)+'; '+error);}
  console.log(JSON.stringify({readyMs:await page.evaluate(()=>Math.round(window.__streamDriveReadyMs))}));
  const gpu=await page.evaluate(()=>{const gl=window.__view.renderer.getContext(),e=gl.getExtension('WEBGL_debug_renderer_info');return e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);});
  const adaptive=process.env.STREAM_ADAPTIVE==='1';
  const initial=await page.evaluate(adaptive=>{if(!adaptive)window.__view.quality(1,true);return window.__car.car.position();},adaptive);
  const flash=process.env.STREAM_FLASH_REVIEW==='1'?await watchDriveFrames(page,cdp,require('jpeg-js').decode,output):null;
  if(flash){
    await flash.phase('resize');
    // A preview panel can change layout after the game draws. Exercise that ordering
    // without touching the canvas, the renderer settings or the game clock.
    await page.evaluate(async()=>{
      const host=window.__view.renderer.domElement.parentElement,prior=host.style.right;
      try{
        for(const inset of [8,0,8,0]){
          await new Promise(resolve=>requestAnimationFrame(()=>{host.style.right=inset+'px';resolve();}));
          await new Promise(resolve=>setTimeout(resolve,240));
        }
      }finally{host.style.right=prior;}
    });
    await flash.phase('parked');
    const resizeReview=await flash.snapshot();console.log(JSON.stringify({resizeReview}));
    await writeFile(output.replace(/\.json$/,'-resize.json'),JSON.stringify({refresh,resizeReview}));
  }
  if(flash&&process.env.STREAM_RESIZE_ONLY==='1'){
    report={scope:'resize-flash-check',refresh,flashReview:await flash.finish(),failures,consoleErrors};
    await writeFile(output,JSON.stringify(report));
  }else{
  if(!(neighborhood&&manual) && Math.hypot(initial[0]-start[0],initial[1]+start[1])>8)throw Error('Car did not spawn on stream route '+JSON.stringify(initial));
  const spawnPiece=await page.evaluate(()=>window.__view.stream.status().piece);
  let idleCheck=null;
  if(!manual){
    const readyEpochMs=Date.now();
    await delay(1000); // Let the chase camera reach its stationary pose before both pictures.
    if(visualDir){await mkdir(visualDir,{recursive:true});await page.screenshot({path:path.join(visualDir,'idle-start.png'),type:'png',fullPage:false});}
    const idle=await page.evaluate(async()=>{
      const view=window.__view,car=window.__car.car,idleStartMs=performance.now(),startStatus=view.stream.status(),pose=()=>{const p=car.position(),c=view.camera;return {car:[p[0],-p[1]],camera:c.position.toArray(),quaternion:c.quaternion.toArray(),status:view.stream.status()};};
      const start=pose(),samples=[];const end=performance.now()+4000;
      await new Promise(resolve=>{const sample=()=>{samples.push(pose());if(performance.now()>=end)resolve();else requestAnimationFrame(sample);};requestAnimationFrame(sample);});
      const maxVector=(key,n)=>Math.max(0,...samples.map(s=>Math.hypot(...s[key].slice(0,n).map((v,i)=>v-start[key][i]))));
      const qdot=(a,b)=>Math.abs(a.reduce((sum,v,i)=>sum+v*b[i],0));
      const maxQuaternionDegrees=Math.max(0,...samples.map(s=>2*Math.acos(Math.min(1,qdot(start.quaternion,s.quaternion)))*180/Math.PI));
      return {settleMs:1000,observeMs:4000,sampleCount:samples.length,idleStartMs,startStatus,endStatus:samples.at(-1)?.status??start.status,
        maxCarPositionJitterM:maxVector('car',2),maxCameraPositionJitterM:maxVector('camera',3),maxCameraQuaternionJitterDegrees:maxQuaternionDegrees,
        startCarEN:start.car,endCarEN:samples.at(-1)?.car??start.car};
    });
    if(visualDir)await page.screenshot({path:path.join(visualDir,'idle-end.png'),type:'png',fullPage:false});
    const idleEndEpochMs=Date.now();
    const idlePrepared=idle.endStatus.events.filter(e=>e.type==='prepare'&&e.id!==spawnPiece);
    const idleOtherRequests=requestedChunks.filter(r=>r.epochMs<=idleEndEpochMs).flatMap(r=>{
      const owners=chunkPackages.get(r.path)||[];if(owners.some(owner=>owner.id===spawnPiece))return [];
      return owners.filter(owner=>owner.id!=='$base').map(owner=>({...r,owner:owner.id}));
    });
    idleCheck={readyEpochMs,spawnPiece,allowedPackages:['$base',spawnPiece],settleMs:idle.settleMs,observeMs:idle.observeMs,sampleCount:idle.sampleCount,
      maxCarPositionJitterM:idle.maxCarPositionJitterM,maxCameraPositionJitterM:idle.maxCameraPositionJitterM,maxCameraQuaternionJitterDegrees:idle.maxCameraQuaternionJitterDegrees,
      startCarEN:idle.startCarEN,endCarEN:idle.endCarEN,idlePrepared,idleOtherRequests,visualPair:visualDir?[path.join(visualDir,'idle-start.png'),path.join(visualDir,'idle-end.png')]:null};
    if(idle.maxCarPositionJitterM>.08)failures.push('Car moved while parked during idle check: '+idle.maxCarPositionJitterM.toFixed(3)+' m');
    if(idle.maxCameraPositionJitterM>.15)failures.push('Camera position drifted while parked during idle check: '+idle.maxCameraPositionJitterM.toFixed(3)+' m');
    if(idle.maxCameraQuaternionJitterDegrees>.5)failures.push('Camera rotation drifted while parked during idle check: '+idle.maxCameraQuaternionJitterDegrees.toFixed(3)+' degrees');
    if(idle.startStatus.active.length!==1||idle.endStatus.active.length!==1||idle.endStatus.active[0]!==spawnPiece||idle.endStatus.loading)failures.push('Parked car did not keep exactly one resident detail section');
    if(idlePrepared.length)failures.push('Neighbor piece prepared while parked: '+idlePrepared.map(e=>e.id).join(','));
    if(idleOtherRequests.length)failures.push('Neighbor piece requested while parked: '+idleOtherRequests.map(e=>e.owner).join(','));
  }
  // The pilot dispatches the same key events as manual driving. It never changes poses or advances physics itself.
  let checkpoint={headed,refresh,gpu,frames:[],samples:[],events:[],idleCheck,readyMs:await page.evaluate(()=>Math.round(window.__streamDriveReadyMs))};
  await page.exposeFunction('__streamBenchProgress',partial=>{
    checkpoint={...checkpoint,...partial,frames:checkpoint.frames.concat(partial.frames),samples:checkpoint.samples.concat(partial.samples),events:checkpoint.events.concat(partial.events)};
    writeFileSync(output,JSON.stringify({scope:'drive-stream',partial:true,launchEpochMs,guardStartedEpochMs:Number(process.env.SF_GUARD_STARTED_EPOCH_MS)||null,packageSelections:Object.fromEntries(selectedProfiles),...checkpoint,presented:flash?.presented,failures,consoleErrors,gpuWarnings,requests}));
  });
  let visualCaptures=[];
  if(visualOnly){await page.exposeFunction('__streamVisualCapture',async meta=>{
    const filename='drive-'+String(meta.crossings).padStart(2,'0')+'-'+String(meta.elapsedSeconds).padStart(3,'0')+'s.png',filepath=path.join(visualDir,filename);
    await page.screenshot({path:filepath,type:'png',fullPage:false});visualCaptures.push({...meta,file:filepath});
  });}
  // The visual pass keeps a small rolling picture so a swap has both sides,
  // including the last render before detail was attached or evicted.
  if(visualOnly)await page.exposeFunction('__streamSwapPair',async ({id,before,after,ms})=>{
    const prefix='swap-'+String(Math.round(ms))+'-'+id;
    await Promise.all([['before',before],['after',after]].map(([side,uri])=>
      writeFile(path.join(visualDir,prefix+'-'+side+'.jpg'),Buffer.from(uri.split(',')[1],'base64'))));
  });
  if(flash)await flash.phase('drive');
  const run=await page.evaluate(async({routeEN,seconds,manual,visualOnly,coverageExpected,measureRide})=>{
    const car=window.__car.car, view=window.__view, route=routeEN.map(([e,n])=>[e,-n]);
    const rideFrames=[],deck=car.group.position.clone(),cameraDirection=deck.clone();
    const initialStatus=view.stream.status(),frames=[], samples=[], held=new Set();let idx=0, direction=1, turnarounds=0, last=performance.now(), sampleAt=-Infinity, distanceM=0;
    let prev=car.position(), startMs=performance.now(), lastProgressMs=startMs, furthest=0, checkpointMs=-Infinity;
    const benchmarkEvents=[];let turning=false,checkpointFrames=0,checkpointSamples=0,checkpointEvents=0,lastVisualCapture=-Infinity,lastVisualCrossings=0,visualCapturePending=false,visualCapturePromise=null;
    const preview=visualOnly?document.createElement('canvas'):null;
    if(preview){preview.width=640;preview.height=360;}
    const previewContext=preview?.getContext('2d',{alpha:false});
    let lastPreviewMs=-Infinity,previousPreview='',swapPairs=0,observedEvent=initialStatus.events.at(-1)?.ms??0;
    const pairWrites=[];
    const coverageSummary={enabled:coverageExpected.length>0,expectedCount:coverageExpected.length,frames:0,minRepresented:Infinity,collisionHullFrames:0,violations:[],evidence:[]};
    const key=(code,on)=>{window.dispatchEvent(new KeyboardEvent(on?'keydown':'keyup',{code,key:{KeyA:'a',KeyD:'d',KeyW:'w',KeyS:'s'}[code],bubbles:true}));};
    const controls=keys=>{for(const code of ['KeyA','KeyD','KeyW','KeyS']){const want=keys.includes(code);if(want!==held.has(code)){key(code,want);if(want)held.add(code);else held.delete(code);}}};
    const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
    let finishReason='duration';
    try {
      await new Promise(done=>{
        const tick=now=>{
          const pose=car.position(), [x,z]=pose, speed=car.telemetry().speedKmh, status=view.stream.status();
          if(measureRide){
            const state=car.debug(),h=state.wheelH;
            deck.set(-1.4,.85,0).applyMatrix4(car.group.matrixWorld).project(view.camera);
            view.camera.getWorldDirection(cameraDirection);
            rideFrames.push([now-startMs,speed,x,z,pose[2],state.pitch,state.roll,
              (1+deck.x)*view.renderer.domElement.width/2,(1-deck.y)*view.renderer.domElement.height/2,
              Math.atan2(-cameraDirection.x,-cameraDirection.z),view.camera.fov,
              (h[0]+h[1]-h[2]-h[3])/2,(h[1]+h[3]-h[0]-h[2])/2,state.sax,state.ay,state.delta,
              state.y-(h[0]+h[1]+h[2]+h[3])/4]);
          }
          distanceM+=Math.hypot(x-prev[0],z-prev[1]);prev=pose;
          if(!visualOnly&&now>startMs)frames.push([+now.toFixed(2),+(now-last).toFixed(2),view.renderer.info.render.calls,view.renderer.info.render.triangles]);last=now;
          // Advance only to nearby route vertices, then steer for a 10 m lookahead. Bounce at endpoints when feasible.
          const near=(i)=>Math.hypot(route[i][0]-x,route[i][1]-z);
          while(idx+direction>=0 && idx+direction<route.length && near(idx+direction)<8)idx+=direction;
          const terminal=direction>0?route.length-1:0;
          if(!manual && idx===terminal && near(terminal)<9)turning=true;
          if(turning && speed<.8){car.placeAt(x,z,(pose[2]??car.group.rotation.y)+Math.PI);direction=-direction;turnarounds++;turning=false;furthest=0;lastProgressMs=now;benchmarkEvents.push({type:'stopped-turnaround',ms:now,positionEN:[x,-z],speedKmh:speed});}
          const progress=direction>0?idx:route.length-1-idx;
          if(progress>furthest){furthest=progress;lastProgressMs=now;}
          let segment=idx, target=route[idx], look=10;
          if(segment+direction>=0 && segment+direction<route.length){
            const a=route[segment], b=route[segment+direction], dx=b[0]-a[0], dz=b[1]-a[1], length=Math.hypot(dx,dz);
            const t=length?Math.max(0,Math.min(1,((x-a[0])*dx+(z-a[1])*dz)/(length*length))):1;
            target=[a[0]+dx*t,a[1]+dz*t];
            while(look>0 && segment+direction>=0 && segment+direction<route.length){
              const end=route[segment+direction], remaining=Math.hypot(end[0]-target[0],end[1]-target[1]);
              if(remaining>=look){target=[target[0]+(end[0]-target[0])*look/remaining,target[1]+(end[1]-target[1])*look/remaining];break;}
              look-=remaining;target=end;segment+=direction;
            }
          }
          const yaw=typeof car.heading==='function'?car.heading():pose[2]??car.group.rotation.y;
          const err=wrap(Math.atan2(-(target[0]-x),-(target[1]-z))-yaw);
          if(!manual && turning)controls(['KeyS']);
          else if(!manual){const keys=[];if(err>.04)keys.push('KeyA');else if(err<-.04)keys.push('KeyD');
            if(speed<54-40*Math.min(1,Math.abs(err)))keys.push('KeyW');else if(speed>60)keys.push('KeyS');controls(keys);}
          if(now-sampleAt>=250){sampleAt=now;
            if(!visualOnly)samples.push({ms:+now.toFixed(2),epochMs:Math.round(performance.timeOrigin+now),positionEN:[+x.toFixed(2),+(-z).toFixed(2)],yaw,speedKmh:+speed.toFixed(2),active:[...status.active],current:status.current??null,piece:status.piece??null,loading:status.loading,ready:status.ready,
              heapBytes:performance.memory?.usedJSHeapSize??null,textures:view.renderer.info.memory.textures,geometries:view.renderer.info.memory.geometries,programs:view.renderer.info.programs?.length??null,imagePool:view.imagePool?.(),routeIndex:idx});
            if(coverageExpected.length){const coverage=view.coverage(),active=new Set(coverage.active),distant=new Set(coverage.shownDistant),missing=[],duplicate=[];
              for(const id of coverageExpected){const count=Number(active.has(id))+Number(distant.has(id));if(!count)missing.push(id);else if(count>1)duplicate.push(id);}
              const represented=coverageExpected.length-missing.length;coverageSummary.frames++;coverageSummary.minRepresented=Math.min(coverageSummary.minRepresented,represented);
              if(coverage.visibleCollisionHulls)coverageSummary.collisionHullFrames++;
              if(missing.length||duplicate.length||coverage.visibleCollisionHulls){if(coverageSummary.violations.length<20)coverageSummary.violations.push({ms:+now.toFixed(0),missing:missing.slice(0,12),duplicate:duplicate.slice(0,12),visibleCollisionHulls:coverage.visibleCollisionHulls});}
              if(coverageSummary.evidence.length<6||missing.length||duplicate.length||coverage.visibleCollisionHulls)if(coverageSummary.evidence.length<10)coverageSummary.evidence.push({ms:+now.toFixed(0),represented,expected:coverageExpected.length,active:coverage.active.slice(0,16),shownDistant:coverage.shownDistant.slice(0,16),visibleCollisionHulls:coverage.visibleCollisionHulls});
            }
          }
          const crossings=status.events.filter(e=>e.type==='crossing').length;
          if(previewContext&&swapPairs<8){
            const changed=status.events.find(e=>e.ms>observedEvent&&(e.type==='attached'||e.type==='evicted'));
            if(changed||now-lastPreviewMs>=100){
              previewContext.drawImage(view.renderer.domElement,0,0,preview.width,preview.height);
              const picture=preview.toDataURL('image/jpeg',.88);
              if(changed&&previousPreview){
                pairWrites.push(window.__streamSwapPair({id:changed.type+'-'+changed.id,before:previousPreview,after:picture,ms:now-startMs}));
                swapPairs++;
              }
              previousPreview=picture;lastPreviewMs=now;observedEvent=status.events.at(-1)?.ms??observedEvent;
            }
          }
          if(visualOnly&&!visualCapturePending&&(now-lastVisualCapture>=8000||(crossings>=4&&lastVisualCrossings<4))){visualCapturePending=true;lastVisualCapture=now;
            visualCapturePromise=window.__streamVisualCapture({elapsedSeconds:Math.round((now-startMs)/1000),crossings,positionEN:[+x.toFixed(2),+(-z).toFixed(2)],current:status.current??null,piece:status.piece??null,active:[...status.active]})
              .then(()=>{lastVisualCrossings=crossings;visualCapturePending=false;}).catch(()=>{visualCapturePending=false;});}
          if(!visualOnly&&now-checkpointMs>=1000){checkpointMs=now;const {events,...snapshotStatus}=status;window.__streamBenchProgress({startMs,endMs:now,timeOriginEpochMs:performance.timeOrigin,initialStatus,canvasReview:window.__flashStats,frames:frames.slice(checkpointFrames),samples:samples.slice(checkpointSamples),events:events.slice(checkpointEvents),status:snapshotStatus,benchmarkEvents,turnarounds,distanceM}).catch(()=>{});checkpointFrames=frames.length;checkpointSamples=samples.length;checkpointEvents=events.length;}
          // Drive through both reversals on the visual pass as well as the timed pass.
          if(now-startMs>=seconds*1000)return done();
          if(!manual && now-lastProgressMs>25000 && speed<2){finishReason='stalled';return done();}
          requestAnimationFrame(tick);
        };requestAnimationFrame(tick);
      });
    }finally{if(!manual)controls([]);}
    if(visualCapturePromise)await visualCapturePromise;
    await Promise.all(pairWrites);
    const status=view.stream.status();
    coverageSummary.minRepresented=coverageSummary.frames?coverageSummary.minRepresented:null;
    return {readyMs:Math.round(window.__streamDriveReadyMs),startMs,endMs:performance.now(),timeOriginEpochMs:performance.timeOrigin,finishReason,manual,initialStatus,distanceM:+distanceM.toFixed(2),turnarounds,benchmarkEvents,frames,samples,events:status.events,status,coverageSummary,
      ...(measureRide?{rideFrameColumns:['ms','kmh','x','z','yaw','pitch','roll','rearDeckXpx','rearDeckYpx','cameraYaw','fov','frontMinusRearM','leftMinusRightM','ax','ay','steer','heaveAboveRoadM'],rideFrames}:{}),
      renderer:typeof view.info==='function'?view.info():null,resolution:[view.renderer.domElement.width,view.renderer.domElement.height],devicePixelRatio:window.devicePixelRatio};
  },{routeEN:manifest.routeEN,seconds,manual,visualOnly,coverageExpected:manifest.distantExteriors&&!manual?detailedPackages.map(p=>p.id):[],measureRide:process.env.STREAM_RIDE_REVIEW==='1'});
  const flashReview=flash?await flash.finish():null;
  if(!manual && flashReview?.presented.vehicle.longestFrames>=3 && flashReview.canvas.vehicle.longestFrames>=3){
    failures.push('Vehicle body turned black in the daylight chase view');
  }
  const firstNeighbor=run.events.find(e=>e.type==='prepare'&&e.id!==spawnPiece);
  const firstNeighborFromStartM=firstNeighbor?.positionEN?.length>=2?Math.hypot(firstNeighbor.positionEN[0]-initial[0],firstNeighbor.positionEN[1]+initial[1]):null;
  if(!manual && !(firstNeighborFromStartM>0.0001))failures.push('First neighbor preparation did not follow real car motion');
  if(run.coverageSummary.enabled&&run.coverageSummary.violations.length)failures.push('Distant representation coverage failed: '+run.coverageSummary.violations.length+' sampled violations');
  if(run.coverageSummary.enabled&&run.coverageSummary.frames&&run.coverageSummary.minRepresented!==run.coverageSummary.expectedCount)failures.push('Distant representation coverage fell below all expected pieces');
  if(run.coverageSummary.enabled&&run.coverageSummary.collisionHullFrames)failures.push('Visible collision hulls appeared in street-level view');
  if(visualOnly){
    if(run.events.filter(e=>e.type==='crossing').length<4)failures.push('Visual-only route captured fewer than four crossings');
    const visualReport={scope:'drive-stream-visual-only',refresh,headed,idleDisplayIntervalMs,launchEpochMs,guardStartedEpochMs:Number(process.env.SF_GUARD_STARTED_EPOCH_MS)||null,idleCheck,finishReason:run.finishReason,
      elapsedMs:Math.round(run.endMs-run.startMs),distanceM:run.distanceM,turnarounds:run.turnarounds,firstNeighbor,firstNeighborFromStartM,coverageSummary:run.coverageSummary,visualCaptures,events:run.events,status:run.status,failures,consoleErrors};
    await writeFile(output,JSON.stringify(visualReport));console.log(JSON.stringify({output,scope:visualReport.scope,captures:visualCaptures.length,crossings:run.events.filter(e=>e.type==='crossing').length,firstNeighborFromStartM,coverageSummary:run.coverageSummary,failures,consoleErrors}));
    if(failures.length||consoleErrors.length)throw Error('Stream visual-only validation failed; see '+output);
  }
  if(!visualOnly){
  if(process.env.STREAM_CAPTURE_FINAL)try{await page.screenshot({path:process.env.STREAM_CAPTURE_FINAL,type:'jpeg',quality:75,fullPage:false});}catch(error){failures.push('Final screenshot failed: '+String(error));}
  const crossingEvents=run.events.filter(e=>e.type==='crossing'&&e.ms>=run.startMs);
  let previousParent=run.initialStatus.current??null;
  const crossings=crossingEvents.filter(e=>{if(!ids.has(e.id)){failures.push('Unknown crossing parent '+e.id);return false;}if(e.id===previousParent){failures.push('Crossing did not change semantic block '+e.id);return false;}previousParent=e.id;return true;});
  const stats=frames=>{const values=frames.map(f=>f[1]).sort((a,b)=>a-b),q=p=>values.length?values[Math.min(values.length-1,Math.ceil(values.length*p)-1)]:null;
    return {frames:values.length,fps:values.length?+(1000*values.length/values.reduce((a,b)=>a+b,0)).toFixed(2):null,p50Ms:q(.5),p95Ms:q(.95),p99Ms:q(.99),maxMs:values.at(-1)??null,over33ms:values.filter(v=>v>33.33).length};};
  const near=f=>crossings.some(e=>Math.abs(f[0]-e.ms)<=1000);
  const crossingChecks=crossings.map(e=>{const before=run.samples.filter(s=>s.ms<=e.ms).at(-1),after=run.samples.find(s=>s.ms>=e.ms);
    const resident=new Set();for(const event of run.events.slice(0,run.events.indexOf(e)+1)){if(event.type==='attached')resident.add(event.id);else if(event.type==='evicted')resident.delete(event.id);}
    const entryPiece=manifest.pieces?e.piece:e.id,entryPieceReady=resident.has(entryPiece)&&pieceParents.get(entryPiece)===e.id;
    return {...e,entryPiece,entryPieceReady,safe:e.ready===true&&entryPieceReady,previousSample:before??null,nextSample:after??null,frames:stats(run.frames.filter(f=>Math.abs(f[0]-e.ms)<=1000))};});
  const evictions=run.events.filter(e=>e.type==='evicted');
  const initialPiece=run.initialStatus.piece??run.initialStatus.current;
  const initialEntryCheck={parent:run.initialStatus.current??null,piece:initialPiece??null,
    safe:!!run.initialStatus.ready&&run.initialStatus.active.includes(initialPiece)&&
      (neighborhood?ids.has(initialPiece):pieceParents.get(initialPiece)===run.initialStatus.current)};
  const traversedParents=[...new Set([run.initialStatus.current,...run.samples.map(s=>s.current)].filter(Boolean))];
  for(const parent of traversedParents)if(!(initialEntryCheck.parent===parent&&initialEntryCheck.safe)&&!crossingChecks.some(e=>e.id===parent&&e.safe))failures.push('Traversed block lacks ready entry-piece evidence '+parent);
  if(manifest.pieces&&!initialEntryCheck.safe)failures.push('Initial semantic block entry piece was not resident and ready');
  if(!manual&&!evictions.length)failures.push('No eviction observed');
  let peakEventActive=0;const eventResident=new Set();for(const event of run.events){if(event.type==='attached'&&pieceParents.has(event.id))eventResident.add(event.id);else if(event.type==='evicted')eventResident.delete(event.id);peakEventActive=Math.max(peakEventActive,eventResident.size);}
  const peakActive=Math.max(peakEventActive,0,...run.samples.map(s=>s.active.length));
  if(peakActive>maxActive)failures.push('Active residency exceeded '+maxActive+': '+peakActive);

  if(!manual&&crossings.length<3)failures.push('Fewer than three crossings: '+crossings.length);
  if(!manual&&new Set(crossings.map(e=>e.id)).size<3)failures.push('Fewer than three distinct destination cells');
  if(crossingChecks.some(e=>!e.safe))failures.push('Crossing without destination ready=true');
  for(const e of run.events)if(e.type==='error')failures.push('stream error '+JSON.stringify(e));
  if(!adaptive&&(run.resolution[0]!==width||run.resolution[1]!==height))failures.push('Renderer resolution differs from requested native '+width+'×'+height);
  const attachedIds=[...new Set(run.events.filter(e=>e.type==='attached').map(e=>e.id))];
  const packageEncodings=[];
  for(const id of ['$base',...attachedIds]){
    const encodings=packages.get(id), selected=selectedProfiles.get(id), encoding=encodings?.find(e=>e.profile===selected);
    if(!encoding){failures.push('No known encoding loaded for '+id);continue;}
    for(const chunk of encoding.chunks)if(!received.has(chunk))failures.push('Missing '+selected+' chunk for '+id+': '+chunk);
    const receivedProfiles=encodings.filter(e=>e.chunks.some(c=>received.has(c))).map(e=>e.profile);
    if(receivedProfiles.length!==1)failures.push('Expected exactly one encoding for '+id+': '+receivedProfiles.join(','));
    packageEncodings.push({id,profile:selected,contentHash:encoding.contentHash,chunks:encoding.chunks,receivedProfiles,
      attachmentCount:run.events.filter(e=>e.type==='attached'&&e.id===id).length});
  }
  report={scope:manifest.scope,refresh,headed,gpu,gpuWarnings,adaptive,flashReview,idleDisplayIntervalMs,world:manifest.world,diagnostic:width!==1920,requestedResolution:[width,height],finalCapture:process.env.STREAM_CAPTURE_FINAL||null,detailedPackageKind:manifest.pieces?'piece':'cell',launchEpochMs,guardStartedEpochMs:Number(process.env.SF_GUARD_STARTED_EPOCH_MS)||null,idleCheck,
    firstNeighbor,firstNeighborFromStartM,coverageSummary:run.coverageSummary,
    ...run,frameColumns:['performanceMs','intervalMs','drawCalls','triangles'],crossingWindowMs:1000,allFrames:stats(run.frames),nearCrossings:stats(run.frames.filter(near)),steady:stats(run.frames.filter(f=>!near(f))),
    crossingChecks,initialEntryCheck,traversedParents,realInterblockCrossings:crossings.length,evictions,peakActive,maxActive,preparationWindows:run.events.filter(e=>e.type==='prepare').map(e=>{const attached=run.events.find(a=>a.type==='attached'&&a.id===e.id&&a.ms>=e.ms),endMs=attached?.ms??run.endMs;return {...e,endMs,attached:!!attached,frames:stats(run.frames.filter(f=>f[0]>=e.ms&&f[0]<=endMs))};}),attachedIds,packageEncodings,packageSelections:Object.fromEntries(selectedProfiles),assetAllowlist:[...allowed],receivedChunks:[...received],requests,failures,consoleErrors};
  await writeFile(output,JSON.stringify(report));
  console.log(JSON.stringify({output,crossings:crossings.length,distanceM:run.distanceM,turnarounds:run.turnarounds,idleCheck,nearCrossings:report.nearCrossings,steady:report.steady,failures,consoleErrors}));
  if(failures.length||consoleErrors.length)throw Error('Stream drive validation failed; see '+output);
  }
  }
} catch(error) {
  if(!report){let partial={};try{partial=JSON.parse(await readFile(output,'utf8'));}catch{}await writeFile(output,JSON.stringify({...partial,scope:'drive-stream',launchEpochMs,guardStartedEpochMs:Number(process.env.SF_GUARD_STARTED_EPOCH_MS)||null,error:String(error),failures,consoleErrors,requests,serverTail}));}
  throw error;
} finally {
  if(browser)await browser.close().catch(()=>{});
  chrome?.kill('SIGTERM');server?.kill('SIGTERM');
  await rm(dir,{recursive:true,force:true,maxRetries:2}).catch(()=>{});
}
