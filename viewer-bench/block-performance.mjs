// Call through guard_memory.py. Only a spatially compiled block may be opened.
import {spawn} from 'node:child_process';
import {createRequire} from 'node:module';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {setTimeout as delay} from 'node:timers/promises';
import path from 'node:path';
import net from 'node:net';
import {fileURLToPath} from 'node:url';

const [project, output] = process.argv.slice(2);
if (!project || !output || !process.env.SF_MEMORY_GUARDED) throw Error('Run inside the memory watchdog');
const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PLAYWRIGHT_FROM || here, 'package.json'));
const {chromium} = require('playwright');
const workspace = path.resolve(project);
const manifestPath = '/compare/route-block.json';
const manifest = JSON.parse(await readFile(path.join(workspace, 'public', manifestPath)));
if (manifest.scope !== 'route-block' || manifest.format !== 'gzip-chunks' || manifest.uncompressedBytes > 150 * 1024**2 ||
    !(manifest.routeEN.length >= 2) || !manifest.chunks.every(url => url.startsWith('/world-'+manifest.contentHash+'-')))
  throw Error('Unsafe or oversized block package');
const comparison=process.env.BLOCK_ENTRY==='comparison';
const [start, next] = manifest.routeEN;
const heading = (Math.atan2(next[0]-start[0], next[1]-start[1])*180/Math.PI+360)%360;
const allowed = new Set(manifest.chunks);
const portProbe=net.createServer();
await new Promise(resolve=>portProbe.listen(0,'127.0.0.1',resolve));
const port=portProbe.address().port; await new Promise(resolve=>portProbe.close(resolve));
const origin='http://127.0.0.1:'+port;
const dir=await mkdtemp(path.join(tmpdir(),'sf-block-browser-'));
let server,chrome,browser;
const failures=[], consoleErrors=[], received=new Set();
let serverTail='';
async function until(fn, ms){
  const end=Date.now()+ms; let last;
  while(Date.now()<end){
    try{const result=await fn(); if(result)return result;}catch(e){last=e;}
    await delay(100);
  }
  throw Error('Startup timeout '+(last?.message || serverTail.slice(-750)));
}
try {
  server=spawn('python3',[path.join(here,'../king-street/local-viewer/serve.py'),'--bundle',path.join(workspace,'.local-viewer'),'--public',path.join(workspace,'public'),'--port',String(port)],
    {cwd:workspace,stdio:['ignore','pipe','pipe'],env:{...process.env,NODE_OPTIONS:'--max-old-space-size=512'}});
  for(const stream of [server.stdout,server.stderr])stream.on('data',data=>{serverTail=(serverTail+data.toString()).slice(-2048);});
  await until(async()=>{if(server.exitCode!==null)throw Error(serverTail);return (await fetch(origin+manifestPath)).ok;},25000);
  // Verify the viewer shell before opening the isolated browser.
  const warmShell=await fetch(origin+'/'); if(!warmShell.ok)throw Error('Could not prepare page');await warmShell.text();
  chrome=spawn(process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    ['--headless=new','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+dir,
    '--no-first-run','--no-default-browser-check','--disable-extensions','--disable-sync','--disable-component-update',
    '--disable-background-networking','--disable-breakpad','--mute-audio','--use-angle=metal','--enable-gpu','about:blank'],
    {stdio:'ignore'});
  const debug=await until(async()=>{const [port,uri]=(await readFile(path.join(dir,'DevToolsActivePort'),'utf8')).split('\n');return uri && 'ws://127.0.0.1:'+port+uri;},10000);
  browser=await chromium.connectOverCDP(debug);
  const context=browser.contexts()[0], page=context.pages()[0];
  const width=Number(process.env.BLOCK_WIDTH || 1280);
  if(![1280,1920].includes(width))throw Error('Unsupported test viewport');
  await page.setViewportSize({width,height:width*9/16});
  await page.route('**/*',async route=>{
    const url=new URL(route.request().url());
    if(url.origin!==origin && ['http:','https:'].includes(url.protocol))return route.abort();
    if(comparison && /^\/block-trial\/(photo-mesh-|block-splats-)/.test(url.pathname)){failures.push('unused overlay '+url.pathname);return route.abort();}
    if(url.pathname.startsWith('/world-')&&!allowed.has(url.pathname)){failures.push('outside block '+url.pathname);return route.abort();}
    return route.continue();
  });
  page.on('pageerror', e=>failures.push(e.message));
  page.on('console',m=>{if(m.type()==='error'&&/shader|webgl|context lost/i.test(m.text()))consoleErrors.push(m.text().slice(-450));});
  page.on('response',r=>{const p=new URL(r.url()).pathname;if(allowed.has(p)&&r.ok())received.add(p);});
  const url=new URL(comparison?'/block-drive':'/',origin);url.search=new URLSearchParams({manifest:manifestPath,drive:'1',measureStartup:'1',driveFrom:[...start,heading].join(',')});
  await page.addInitScript(()=>{const check=()=>{if(window.__view&&window.__car?.car?.group.visible&&!document.querySelector('.load-notice'))window.__readyAt=performance.now();else requestAnimationFrame(check);};requestAnimationFrame(check);});
  await page.goto(url.href,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.__readyAt || document.querySelector('.load-notice')?.textContent.includes('could not open'),null,{timeout:55000});
  const ready=await page.evaluate(()=>window.__readyAt||document.querySelector('.load-notice')?.textContent);
  if(typeof ready!=='number')throw Error(ready);
  const initial=await page.evaluate(()=>window.__car.car.position());
  if(Math.hypot(initial[0]-start[0],initial[1]+start[1])>6)throw Error('Car did not spawn on the scoped route '+JSON.stringify(initial));
  await page.evaluate(()=>{
    const view=window.__view, gl=view.renderer.getContext(), composer=view.composer;
    const ext=gl.getExtension('EXT_disjoint_timer_query_webgl2'), submit=[], gpu=[], calls=[], triangles=[], pending=[];
    const render=composer.render.bind(composer);
    let frame=0;
    composer.render=(...args)=>{
      if(ext && gl.getParameter(ext.GPU_DISJOINT_EXT)){for(const q of pending)gl.deleteQuery(q);pending.length=0;}
      while(pending.length && gl.getQueryParameter(pending[0],gl.QUERY_RESULT_AVAILABLE)){
        const q=pending.shift();gpu.push(gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6);gl.deleteQuery(q);
      }
      const q=ext && frame++%16===0 && pending.length<3 ? gl.createQuery():null;
      if(q)gl.beginQuery(ext.TIME_ELAPSED_EXT,q);
      const t=performance.now();render(...args);submit.push(performance.now()-t);
      if(q){gl.endQuery(ext.TIME_ELAPSED_EXT);pending.push(q);}
      calls.push(view.renderer.info.render.calls);triangles.push(view.renderer.info.render.triangles);
    };
    window.__costs=()=>{const median=a=>a.length?+a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)].toFixed(2):null;
      composer.render=render;for(const q of pending)gl.deleteQuery(q);
      return {medianSubmitMs:median(submit),medianGpuMs:median(gpu),gpuSamples:gpu.length,
        medianDraws:median(calls),medianTriangles:median(triangles)};};
  });
  await page.keyboard.down('w');
  const run=await page.evaluate(async()=>{
    const intervals=[], start=performance.now(), car=window.__car.car, pose=car.position();
    let last=await new Promise(requestAnimationFrame);
    while(performance.now()-start<12000){
      const now=await new Promise(requestAnimationFrame);
      intervals.push(now-last);last=now;
      if(Math.hypot(car.position()[0]-pose[0],car.position()[1]-pose[1])>=100)break;
    }
    const elapsed=intervals.reduce((a,b)=>a+b,0), sorted=[...intervals].sort((a,b)=>a-b);
    const end=car.position();
    return {frames:intervals.length, driveMs:Math.round(elapsed),fps:Math.round(1000*intervals.length/elapsed*10)/10,
      p95Ms:+sorted[Math.ceil(sorted.length*.95)-1].toFixed(2),meters:Math.round(Math.hypot(end[0]-pose[0],end[1]-pose[1])),
      p99Ms:+sorted[Math.ceil(sorted.length*.99)-1].toFixed(2), textures:window.__view.renderer.info.memory.textures,
      stages:performance.getEntriesByType('mark').filter(x=>x.name.startsWith('world.')).map(x=>[x.name,Math.round(x.startTime)]),
      ...window.__view.info(),...window.__costs(),resolution:[window.__view.renderer.domElement.width,window.__view.renderer.domElement.height]};
  });
  await page.keyboard.up('w');
  const report={world:manifest.contentHash,readyMs:Math.round(ready),...run,chunks:received.size,expectedChunks:allowed.size,failures,consoleErrors};
  console.log(JSON.stringify(report));await writeFile(output,JSON.stringify(report,null,2));
  if(received.size!==allowed.size||failures.length||consoleErrors.length||run.meters<80)throw Error('Drive validation failed');
} finally {
  if(browser)await browser.close().catch(()=>{});
  chrome?.kill('SIGTERM');server?.kill('SIGTERM');
  // The watchdog also clears all remaining isolated helpers after this command.
  await rm(dir,{recursive:true,force:true,maxRetries:2}).catch(()=>{});
}
