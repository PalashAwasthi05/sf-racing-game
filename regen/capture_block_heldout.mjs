import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync,copyFileSync} from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const require=createRequire('/Users/palash/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const {chromium}=require('playwright');
const manifestPath=process.argv[2],baseUrl=process.argv[3]||'http://127.0.0.1:3001';
const jobs=JSON.parse(readFileSync(manifestPath,'utf8')),out=path.dirname(manifestPath),captures=[];
mkdirSync(path.join(out,'sources'),{recursive:true});
for(const j of jobs.jobs)copyFileSync(j.sourceImagePath,path.join(out,'sources',`${j.sourceCameraId}.jpg`));
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--headless=new','--use-angle=metal','--enable-gpu','--ignore-gpu-blocklist']});
const page=await browser.newPage({viewport:{width:1100,height:740},deviceScaleFactor:1}),errors=[];
page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text().slice(0,1800));});
const audit={startedAt:new Date().toISOString(),jobsSha256:crypto.createHash('sha256').update(readFileSync(manifestPath)).digest('hex'),captures,errors};
try{
 await page.goto(baseUrl+'/block-drive?version=baseline',{waitUntil:'domcontentloaded',timeout:120000});
 await page.getByRole('button',{name:'Current mesh',exact:true}).waitFor({timeout:300000});
 await page.waitForFunction(()=>document.querySelector('.trial-switch button')?.disabled===false,null,{timeout:300000});
 audit.trial=await page.evaluate(()=>fetch('/block-trial/trial.json',{cache:'no-cache'}).then(r=>r.json()));
 audit.loadedFiles={};
 for(const key of ['manifest','photo','splats']){
   const response=await page.request.get(baseUrl+audit.trial[key]);const body=await response.body();
   audit.loadedFiles[key]={url:audit.trial[key],bytes:body.length,sha256:crypto.createHash('sha256').update(body).digest('hex')};
 }
 // Seeking once exits manual mode and hides the manual player vehicle.
 await page.getByRole('slider').click();
 await page.waitForTimeout(300);
 for(const job of jobs.jobs){
   for(const [index,variant] of ['baseline','photo','splats'].entries()){
     await page.keyboard.press('Digit'+(index+1));
     await page.waitForFunction(v=>new URL(location.href).searchParams.get('version')===v,variant);
     await page.evaluate(j=>{const v=window.__view;v.quality(1,true);v.pose(j.fromEnu,j.atEnu);v.camera.fov=j.verticalFovDeg;v.camera.updateProjectionMatrix();},job);
     await page.waitForTimeout(400);
     const before=await page.evaluate(()=>({eye:window.__view.camera.position.toArray(),fov:window.__view.camera.fov,quality:window.__view.quality(),url:location.href,info:window.__view.info()}));
     const expected=[job.fromEnu[0],job.fromEnu[2],-job.fromEnu[1]];
     if(Math.max(...before.eye.map((v,i)=>Math.abs(v-expected[i])))>1e-4)throw Error('Capture eye moved from frozen pose: '+job.id);
     const data=await page.evaluate(async j=>{const blob=await window.__view.capture(j.width,j.height);return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(blob);});},job);
     const body=Buffer.from(data.split(',')[1],'base64');
     if(body.readUInt32BE(16)!==job.width||body.readUInt32BE(20)!==job.height)throw Error('Wrong capture resolution');
     const filename=`${job.id}-${variant}.png`;writeFileSync(path.join(out,filename),body);
     captures.push({jobId:job.id,variant,filename,sha256:crypto.createHash('sha256').update(body).digest('hex'),bytes:body.length,before});
     console.log(JSON.stringify({captured:filename,bytes:body.length}));
   }
 }
 audit.completedAt=new Date().toISOString();audit.status='complete';
}catch(e){audit.status='failed';audit.failure=String(e);throw e;}
finally{writeFileSync(path.join(out,'capture-audit.json'),JSON.stringify(audit,null,2));await browser.close();}
