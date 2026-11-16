// Guarded full-map stationary QA. Captures the regular live render pipeline, never view.capture().
// node sharpness-parked.mjs http://127.0.0.1:3001/ route.json out-dir [prior-report.json]
import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {browser as cdp} from './stream-cdp.mjs';
const [origin,fixturePath,out,priorFile]=process.argv.slice(2);
if(!process.env.SF_MEMORY_GUARDED||!out)throw Error('Run through guard_memory.py: origin route.json out-dir [prior-report.json]');
const url=new URL(origin);
if(!['127.0.0.1','localhost','[::1]'].includes(url.hostname)||url.pathname!=='/'||url.searchParams.has('manifest')||url.searchParams.has('stream'))throw Error('Expected regular local full-map entry');
const fixture=JSON.parse(await readFile(fixturePath,'utf8'));
const prior=priorFile?JSON.parse(await readFile(priorFile,'utf8')):null;
url.searchParams.set('drive','1');url.searchParams.set('driveFrom',fixture.spawn.join(','));
await mkdir(out,{recursive:true});
const dir=await mkdtemp(path.join(tmpdir(),'sf-sharpness-'));
const chrome=spawn(process.env.CHROME_EXECUTABLE||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-background-networking','--no-first-run','--disable-extensions','--disable-sync','--no-default-browser-check','--disable-component-update','--mute-audio','--use-angle=metal','--enable-gpu','--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','--user-data-dir='+dir,'about:blank'],{stdio:['ignore','ignore','pipe']});
let debugUrl,browser;
chrome.stderr.on('data',chunk=>{const m=chunk.toString().match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/devtools\/browser\/[^\s]+)/);if(m)debugUrl=m[1];});
const report={url:url.href,fixture:fixturePath,priorFile:priorFile??null,samples:[],errors:[]};
try{
  for(let i=0;i<150&&!debugUrl;i++){try{const [port,endpoint]=(await readFile(path.join(dir,'DevToolsActivePort'),'utf8')).trim().split('\n');debugUrl='ws://127.0.0.1:'+port+endpoint;}catch{await delay(50);}}
  if(!debugUrl)throw Error('No Chrome endpoint');
  browser=await cdp.connect(debugUrl);const context=browser.contexts()[0],page=context.pages()[0],session=await context.newCDPSession(page);
  await session.send('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
  report.browser=browser.version();const {gpu}=await context.connection.send('SystemInfo.getInfo');report.renderer=gpu.auxAttributes.glRenderer;
  if(!report.renderer||/swiftshader|llvmpipe|software rasterizer/i.test(report.renderer))throw Error('Hardware GPU required');
  page.on('pageerror',e=>report.errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')report.errors.push(m.text());});
  await page.goto(url.href);
  await page.waitForFunction(()=>window.__car?.car?.group.visible&&!document.querySelector('.load-notice'),null,{timeout:60000});
  await page.evaluate(()=>{__view.lockQuality(true);__view.quality(1,true);});
  // Give the stationary chase camera, temporal history and full-map tile attachments time to settle.
  await delay(1800);
  await page.waitForFunction(()=>{const s=__view.stream.status();return s.ready&&!s.loading;},null,{timeout:20000});
  report.scope=await page.evaluate(async()=>{const m=await(await fetch('/compare/neighborhood-stream.json')).json();return {scope:m.scope,hash:m.sourceWorldHash,cells:m.cells.length,buildings:m.base.stats.buildings,active:__view.stream.status().active,bundle:document.querySelector('script[type="module"][src]')?.getAttribute('src')};});
  if(report.scope.hash!==fixture.sourceWorldHash||report.scope.scope!=='neighborhood-stream'||report.scope.cells<300)throw Error('Unexpected map scope');
  report.camera=await page.evaluate(saved=>{const v=__view,c=v.camera;const pose=saved??{position:c.position.toArray(),quaternion:c.quaternion.toArray(),fov:c.fov,near:c.near,far:c.far};
    // Restore before each normal composer render, keeping the regular live post-processing path.
    const render=v.composer.render.bind(v.composer);v.composer.render=(...args)=>{c.position.fromArray(pose.position);c.quaternion.fromArray(pose.quaternion);c.fov=pose.fov;c.aspect=1920/1080;c.updateProjectionMatrix();c.updateMatrixWorld();return render(...args);};return pose;},prior?.camera??null);
  if(await page.evaluate(()=>typeof __view.scenery==='function')){
    await page.waitForFunction(()=>{const s=__view.scenery();return !!s.errors.length || !!s.facades?.lastError || s.trees?.errors>0 ||
        s.facades&&!s.facades.loading&&s.facades.active.length>0&&s.trees?.readyCells>0&&!s.trees.loading;},null,{timeout:45000});
    await delay(400);report.scenery=await page.evaluate(()=>__view.scenery());
    if(report.scenery.errors.length||report.scenery.facades?.lastError||report.scenery.trees?.errors)throw Error('Scenery layer reported an error');
  }
  await delay(500);
  for(let i=0;i<3;i++){
    const sample=await page.evaluate(()=>{const v=__view,c=v.camera,t=v.composer.passes[0].target,r=v.renderer.domElement;return {scene:[t.width,t.height],canvas:[r.width,r.height],dpr:devicePixelRatio,viewport:[innerWidth,innerHeight],camera:{position:c.position.toArray(),quaternion:c.quaternion.toArray(),fov:c.fov},car:__car.car.position(),speedKmh:__car.car.telemetry().speedKmh,stream:v.stream.status(),info:v.info()};});
    if(JSON.stringify(sample.scene)!=='[1920,1080]'||JSON.stringify(sample.canvas)!=='[1920,1080]'||sample.dpr!==1)throw Error('Native 1080p check failed');
    if(Math.hypot(sample.car[0]-fixture.spawn[0],sample.car[1]+fixture.spawn[1])>.1||sample.speedKmh>.01)throw Error('Stationary pose drift');
    await page.screenshot({path:path.join(out,`sample-${i}.png`)});report.samples.push(sample);await delay(150);
  }
  report.cameraStable=report.samples.every(s=>JSON.stringify(s.camera)===JSON.stringify(report.samples[0].camera));
  if(!report.cameraStable||report.errors.length)throw Error('Camera unstable or page errors');
  report.complete=true;
}catch(e){report.failure=String(e);throw e;}finally{
  await writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));await browser?.close();chrome.kill('SIGTERM');await delay(300);await rm(dir,{recursive:true,force:true,maxRetries:3}).catch(()=>{});
}
