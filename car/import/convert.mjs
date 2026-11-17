import {createRequire} from 'node:module';import {writeFileSync} from 'node:fs';
const require=createRequire('/Users/palash/Documents/reflex/code/soma/package.json');const {chromium}=require('playwright');
const [out,optJson='{}']=process.argv.slice(2);
const b=await chromium.launch({headless:true,args:['--headless=new','--use-angle=metal','--enable-gpu']});
const p=await (await b.newContext()).newPage();p.on('pageerror',e=>console.log('pageerror',e.message));p.on('console',m=>{if(m.type()==='error')console.log('console',m.text().slice(0,300));});
await p.goto('http://127.0.0.1:3011/fbx/index.html');await p.waitForFunction(()=>window.READY);
await p.evaluate(()=>window.X.load('boxster','FINAL_MODEL_S_17.fbx'));await p.waitForTimeout(1500);
const r=await p.evaluate(async o=>{const m=await import('./convert.js?'+Date.now());return m.convert(o);},JSON.parse(optJson));
writeFileSync(out,Buffer.from(r.b64,'base64'));console.log(JSON.stringify({report:r.report,meshes:r.meshes,bytes:r.bytes}));await b.close();
