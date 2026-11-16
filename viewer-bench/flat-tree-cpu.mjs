// Actual local full-map GLB headers and flat node assembly, without a browser/GPU/image decoding.
import {readFile,writeFile} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {performance} from 'node:perf_hooks';
import {setImmediate as yieldTurn} from 'node:timers/promises';
import path from 'node:path';
import {MeshStandardMaterial} from '../king-street/node_modules/three/build/three.module.js';
import {flatNodes,omitStaticTrees} from '../king-street/app/scene-flat.ts';
const root=path.resolve(import.meta.dirname,'../king-street/public');
const manifest=JSON.parse(await readFile(path.join(root,'compare/neighborhood-stream.json'),'utf8'));
const results=[];
for(const id of ['grid-0--3','grid-1--2','grid-2--1']){
 const cell=manifest.cells.find(c=>c.id===id);if(!cell)continue;
 const bytes=gunzipSync(Buffer.concat(await Promise.all(cell.chunks.map(c=>readFile(path.join(root,c))))));
 const length=bytes.readUInt32LE(12),source=JSON.parse(bytes.subarray(20,20+length).toString());
 const classes=new Set(source.nodes.filter(n=>n.name?.startsWith('vegetation___')).map(n=>n.name.slice('vegetation___'.length)).filter(c=>/^(leaf.*|frond|bark|wood|trunk)$/.test(c)));
 const originalNodes=source.scenes[0].nodes.length;
 for(const mode of ['original','omit','omit-sliced']){
  const doc=structuredClone(source),start=performance.now();
  const skipped=mode==='original'?new Set():omitStaticTrees(doc,classes);
  const pruningMs=performance.now()-start;
  const buffers=doc.bufferViews.map((v,i)=>skipped.has(i)?undefined:bytes.buffer.slice(bytes.byteOffset+28+length+(v.byteOffset??0),bytes.byteOffset+28+length+(v.byteOffset??0)+v.byteLength));
  const materials=doc.materials.map(()=>new MeshStandardMaterial());
  const parser={json:doc,createUniqueName:s=>s,getDependency:async(_kind,i)=>materials[i],assignFinalMaterial:()=>{},associations:new Map()};
  let yields=0,segmentStart=performance.now(),maxCpuSliceMs=0;
  const yieldCpu=mode==='omit-sliced'?async()=>{maxCpuSliceMs=Math.max(maxCpuSliceMs,performance.now()-segmentStart);yields++;await yieldTurn();segmentStart=performance.now();}:undefined;
  const plugin=flatNodes(parser,buffers,Promise.resolve(),yieldCpu);
  const assemblyStart=performance.now();segmentStart=assemblyStart;
  if(!plugin.loadNode)throw Error('Actual cell did not select static flat path');
  const nodes=await Promise.all(doc.scenes[0].nodes.map(i=>plugin.loadNode(i)));
  const assemblyMs=performance.now()-assemblyStart;maxCpuSliceMs=Math.max(maxCpuSliceMs,performance.now()-segmentStart);
  results.push({id,mode,originalNodes,assembledNodes:nodes.length,classes:[...classes],skippedViews:skipped.size,skippedBytes:[...skipped].reduce((sum,i)=>sum+doc.bufferViews[i].byteLength,0),retainedViewBytes:buffers.reduce((sum,b)=>sum+(b?.byteLength??0),0),pruningMs,assemblyMs,yields,maxCpuSliceMs});
  for(const g of new Set(nodes.map(n=>n.geometry)))g.dispose();materials.forEach(m=>m.dispose());plugin.afterRoot?.();
 }
}
const report={sourceWorldHash:manifest.sourceWorldHash,scope:'Three static flat geometry assembly from actual GLB headers and bytes; excludes material/image decoding, drawable finishes, GPU and browser scheduling. Single cheap sample per mode, not an FPS benchmark.',results};
const out=process.argv[2]||'/tmp/flat-tree-cpu.json';await writeFile(out,JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
