import {createRequire} from 'node:module';import {mkdirSync,writeFileSync} from 'node:fs';
const require=createRequire('/Users/palash/Documents/reflex/code/soma/package.json');const {chromium}=require('playwright');
const [dir,file,out]=process.argv.slice(2);mkdirSync(out,{recursive:true});
const b=await chromium.launch({headless:true,args:['--headless=new','--use-angle=metal','--enable-gpu']});
const p=await (await b.newContext({viewport:{width:1600,height:1000}})).newPage();p.on('pageerror',e=>console.log('pageerror',e.message));p.on('console',m=>{if(m.type()==='error'||m.type()==='warning')console.log('console',m.text().slice(0,200));});
await p.goto('http://127.0.0.1:3011/fbx/index.html');await p.waitForFunction(()=>window.READY);
const box=await p.evaluate(([d,f])=>window.X.load(d,f),[dir,file]);console.log('bbox',JSON.stringify(box));
const tree=await p.evaluate(()=>window.X.tree());writeFileSync(out+'/tree.json',JSON.stringify(tree,null,1));
console.log('meshes',tree.length,'tris',tree.reduce((s,t)=>s+t.tris,0));
await p.waitForTimeout(1500);
const c=[(box.min[0]+box.max[0])/2,(box.min[1]+box.max[1])/2,(box.min[2]+box.max[2])/2],L=Math.max(box.max[0]-box.min[0],box.max[1]-box.min[1],box.max[2]-box.min[2]);
const views={a:[1,.35,1],b:[-1,.35,-1],c:[1,.3,-1],top:[.01,1,0]};
for(const [n,d] of Object.entries(views)){const k=L*2.6/Math.hypot(...d);const img=await p.evaluate(([f,a])=>window.X.shot(f,a),[[c[0]+d[0]*k,c[1]+d[1]*k,c[2]+d[2]*k],c]);writeFileSync(`${out}/${n}.jpg`,Buffer.from(img,'base64'));}
await b.close();
