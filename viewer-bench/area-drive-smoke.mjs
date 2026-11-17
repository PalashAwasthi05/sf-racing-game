// Background-only named-area smoke. Run via guard_memory.py; never starts a server.
import {spawn,execFileSync} from 'node:child_process';
import {mkdir,mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {browser as cdp} from './stream-cdp.mjs';
const [origin,areaArg,outArg]=process.argv.slice(2),bench=path.dirname(fileURLToPath(import.meta.url));
if(!process.env.SF_MEMORY_GUARDED||!outArg)throw Error('Required: guard_memory.py ... node area-drive-smoke.mjs ORIGIN AREA OUTPUT');
const area=path.resolve(areaArg),out=path.resolve(outArg),url=new URL(origin);
if(!['localhost','127.0.0.1','[::1]'].includes(url.hostname)||url.pathname!=='/'||url.search)throw Error('Supply bare local viewer origin');
const areaId=path.basename(area);if(!/^[a-z0-9][a-z0-9-]*$/.test(areaId))throw Error('Invalid area name');
const fixture=JSON.parse(execFileSync(path.join(bench,'../geoenv/bin/python'),['-c',`import json,sys\nfrom pyproj import Transformer\nfrom shapely.geometry import shape\nfrom shapely.ops import transform\np=sys.argv[1];c=json.load(open(p+'/config.json'))['region']['center'];crs=json.load(open(p+'/area-source-job.json'))['projectedCrs'];t=Transformer.from_crs(4326,crs,always_xy=True);x,y=t.transform(c['lon'],c['lat']);g=transform(t.transform,shape(json.load(open(p+'/route-drive.geojson'))));pts=[g.interpolate(g.length*i/16) for i in range(17)];print(json.dumps({'routeEN':[[q.x-x,q.y-y] for q in pts],'lengthM':g.length,'center':c}))`,area],{encoding:'utf8'}));
const manifestPath='/compare/'+areaId+'.json';url.searchParams.set('manifest',manifestPath);url.searchParams.set('drive','1');
const report={url:url.href,manifestPath,area,fixture,method:'3-second real keyboard physics probe, then 17 projected debug car/camera placements along real route; not a full simulated drive',samples:[],errors:[],complete:false};
await mkdir(out,{recursive:true});await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
let chrome,browser;const dir=await mkdtemp(path.join(tmpdir(),'sf-area-smoke-'));
try{
 chrome=spawn(process.env.CHROME_EXECUTABLE||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-background-networking','--no-first-run','--disable-extensions','--disable-sync','--no-default-browser-check','--disable-component-update','--mute-audio','--use-angle=metal','--enable-gpu','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+dir,'about:blank'],{stdio:['ignore','ignore','pipe']});
 let endpoint;chrome.stderr.on('data',b=>{const m=String(b).match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/devtools\/browser\/\S+)/);if(m)endpoint=m[1];});
 for(let i=0;i<150&&!endpoint;i++){try{const [port,suffix]=(await readFile(path.join(dir,'DevToolsActivePort'),'utf8')).trim().split('\n');endpoint=`ws://127.0.0.1:${port}${suffix}`;}catch{}await delay(50);}
 if(!endpoint)throw Error('Headless browser failed to start');
 browser=await cdp.connect(endpoint);const context=browser.contexts()[0],page=context.pages()[0],session=await context.newCDPSession(page);
 await session.send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
 page.on('pageerror',e=>report.errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
 await session.send('Page.addScriptToEvaluateOnNewDocument',{source:'performance.setResourceTimingBufferSize(4000)'});await page.goto(url.href);await page.waitForFunction(()=>window.__car?.car?.group.visible&&window.__view?.stream?.status().ready&&!document.querySelector('.load-notice'),null,{timeout:60000});
 report.scope=await page.evaluate(async mp=>{
  // Snapshot before inspection fetches so our own requests cannot manufacture proof.
  const assets=performance.getEntriesByType('resource').map(e=>new URL(e.name).pathname);
  const archive=await(await fetch(mp)).json();if(!archive.spatialIndex||!archive.world)throw Error('Named archive lacks immutable spatialIndex/world');
  const indexPath=new URL(archive.spatialIndex,location.href).pathname;
  const m=await(await fetch(indexPath)).json(),worldPath=new URL(m.world,location.href).pathname,s=__view.stream.status(),ids=new Set(m.cells.map(c=>c.id));
  const chunkPaths=p=>!p||typeof p!=='object'?[]:[...(p.chunks??[]),...Object.entries(p).filter(([k,v])=>k!=='chunks'&&v&&typeof v==='object').flatMap(([,v])=>Array.isArray(v)?[]:chunkPaths(v))];
  return {archiveHash:archive.contentHash,manifest:{routeEN:m.routeEN,cellCount:m.cells.length},indexPath,worldPath,hashMatches:m.sourceWorldHash===archive.contentHash,active:s.active,activeBelongs:s.active.length>0&&s.active.every(id=>ids.has(id)),baseLoaded:chunkPaths(m.base).some(p=>assets.includes(p)),archiveLoaded:assets.includes(mp),indexLoaded:assets.includes(indexPath),worldLoaded:assets.includes(worldPath),loadedManifests:assets.filter(p=>p.endsWith('.json')&&p.includes('/compare/')),spawnYaw:__car.car.position()[2],spawnEN:[__car.car.position()[0],-__car.car.position()[1]]};
 },manifestPath);
 if(!report.scope.hashMatches||!report.scope.activeBelongs||!report.scope.baseLoaded||!report.scope.archiveLoaded||!report.scope.indexLoaded||!report.scope.worldLoaded)throw Error('Named archive/hash/index/world/base/active-cell scope proof failed');
 const m=report.scope.manifest;if(!Array.isArray(m.routeEN)||m.routeEN.length<2)throw Error('Named manifest lacks routeEN');
 const endError=Math.max(...[0,1].map(i=>{const p=i?m.routeEN.at(-1):m.routeEN[0],q=i?fixture.routeEN.at(-1):fixture.routeEN[0];return Math.hypot(p[0]-q[0],p[1]-q[1]);}));report.manifestRouteEndpointErrorM=endError;if(endError>15)throw Error('Named manifest route differs from configured real route');
 report.spawnErrorM=Math.hypot(...report.scope.spawnEN.map((v,i)=>v-fixture.routeEN[0][i]));if(report.spawnErrorM>20)throw Error('Driving spawn is not near the configured carriageway');
 await page.evaluate(()=>{__view.lockQuality(true);__view.quality(1,true);__car.car.mute(true);});await delay(700);
 report.physics=await page.evaluate(async()=>{const c=__car.car,before=c.position();window.dispatchEvent(new KeyboardEvent('keydown',{code:'KeyW',key:'w',bubbles:true}));try{await new Promise(r=>setTimeout(r,3000));}finally{window.dispatchEvent(new KeyboardEvent('keyup',{code:'KeyW',key:'w',bubbles:true}));}const after=c.position();return {before,after,advanceM:Math.hypot(after[0]-before[0],after[1]-before[1]),speedKmh:c.telemetry().speedKmh};});
 const tangent=fixture.routeEN[1].map((v,i)=>v-fixture.routeEN[0][i]),tangentLength=Math.hypot(...tangent);report.physics.forwardAlongRouteM=((report.physics.after[0]-report.physics.before[0])*tangent[0]-(report.physics.after[1]-report.physics.before[1])*tangent[1])/tangentLength;
 if(report.physics.advanceM<.5||report.physics.forwardAlongRouteM<.2)throw Error('Real car physics did not advance along the configured route from its start');
 for(let i=0;i<fixture.routeEN.length;i++){
  const p=fixture.routeEN[i],next=fixture.routeEN[Math.min(i+1,16)],prior=fixture.routeEN[Math.max(0,i-1)],d=i===16?[p[0]-prior[0],p[1]-prior[1]]:[next[0]-p[0],next[1]-p[1]],len=Math.hypot(...d),look=[p[0]+d[0]/len*25,p[1]+d[1]/len*25];
  await page.evaluate(({p,look})=>{const car=__car.car;car.placeAt(p[0],-p[1],Math.atan2(-(look[0]-p[0]),look[1]-p[1]));},{p,look});
  await delay(250);await page.waitForFunction(()=>{const s=__view.stream.status();return s.ready&&!s.loading;},null,{timeout:20000});await delay(i%8===0?1100:180);
  const sample=await page.evaluate(()=>{const v=__view,t=v.composer.passes[0].target,c=v.renderer.domElement;return {carEN:[__car.car.position()[0],-__car.car.position()[1]],camera:v.camera.position.toArray(),canvas:[c.width,c.height],scene:[t.width,t.height],stream:v.stream.status(),info:v.info(),frontage:typeof v.frontageCoverage==='function'?v.frontageCoverage():null,sceneryErrors:v.scenery?.().errors};});
  if([...sample.canvas,...sample.scene].some((n,j)=>n!==[1920,1080,1920,1080][j]))throw Error('Native 1080p failed');if(sample.info.calls<1||sample.info.triangles<1)throw Error('Empty render');
  sample.fraction=i/16;report.samples.push(sample);if(i%8===0){sample.screenshot=path.join(out,['start','','','','','','','','midpoint','','','','','','','','endpoint'][i]+'.png');await page.screenshot({path:sample.screenshot});}
  await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
 }
 report.projectedAdvancementM=fixture.lengthM;report.complete=report.errors.length===0;if(!report.complete)throw Error('Page errors recorded');
}catch(e){report.failure=String(e);process.exitCode=1;}finally{await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));await browser?.close().catch(()=>{});chrome?.kill('SIGTERM');await delay(250);await rm(dir,{recursive:true,force:true}).catch(()=>{});}
