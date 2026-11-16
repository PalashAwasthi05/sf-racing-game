import assert from 'node:assert/strict';
import test from 'node:test';
import {MeshStandardMaterial} from 'three';
import {flatNodes,omitStaticTrees,sceneCpuSlice} from './scene-flat.ts';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';

function document() {
 const matrix=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
 const primitive=(position:number,material:number)=>({attributes:{POSITION:position},indices:1,material});
 return {asset:{version:'2.0'},scene:0,scenes:[{nodes:[0,1,2]}],nodes:[
  {name:'facades__id__paint',mesh:0,matrix},
  {name:'vegetation___leaf1',mesh:1,matrix,extras:{layer:'vegetation'}},
  {name:'vegetation___bark',mesh:2,matrix,extras:{layer:'vegetation'}},
 ],meshes:[{primitives:[primitive(0,0)]},{primitives:[primitive(2,1)]},{primitives:[primitive(0,2)]}],
 materials:[{},{pbrMetallicRoughness:{baseColorTexture:{index:0}}},{}],
 textures:[{source:0}],images:[{bufferView:3}],bufferViews:[{},{},{},{}],
 accessors:[{bufferView:0,componentType:5126,type:'VEC3',count:3,min:[0,0,0],max:[1,1,0]},
  {bufferView:1,componentType:5123,type:'SCALAR',count:3},
  {bufferView:2,componentType:5126,type:'VEC3',count:3,min:[0,0,0],max:[1,1,0]}]};
}

test('tree pruning removes only declared vegetation roots and exclusive geometry/image views',()=>{
 const doc=document(),skipped=omitStaticTrees(doc,new Set(['leaf1','bark']));
 assert.deepEqual(doc.scenes[0].nodes,[0]);assert.deepEqual([...skipped].sort(),[2,3]);
 // Source indices stay stable for parser associations and shared accessor references.
 assert.equal(doc.nodes.length,3);assert.equal(doc.meshes[0].primitives[0].attributes.POSITION,0);
});
test('uncovered classes, conflicting layers and shared texture views remain intact',()=>{
 const doc=document();Object.assign(doc.nodes[1].extras!,{layer:'facades'});
 assert.deepEqual([...omitStaticTrees(doc,new Set(['leaf1','bark']))],[]);
 assert.deepEqual(doc.scenes[0].nodes,[0,1]);
 const shared=document();shared.materials[0]=shared.materials[1];
 assert.deepEqual([...omitStaticTrees(shared,new Set(['leaf1']))],[2]);
 assert.deepEqual(shared.scenes[0].nodes,[0,2]);
});
test('hierarchy, animations and custom material extensions disable pruning',()=>{
 for(const mutate of [
  (d:ReturnType<typeof document>)=>Object.assign(d.nodes[0],{children:[1]}),
  (d:ReturnType<typeof document>)=>Object.assign(d,{animations:[{}]}),
  (d:ReturnType<typeof document>)=>Object.assign(d.materials[0],{extensions:{custom:{texture:0}}}),
 ]){const doc=document();mutate(doc);assert.deepEqual([...omitStaticTrees(doc,new Set(['leaf1']))],[]);assert.deepEqual(doc.scenes[0].nodes,[0,1,2]);}
});
test('flat assembly yields between expensive nodes only when live scheduling is supplied',async()=>{
 const doc=document();doc.materials=[{}, {}, {}];const buffers=[new Float32Array([0,0,0,1,0,0,0,1,0]).buffer,new Uint16Array([0,1,2]).buffer,new Float32Array(9).buffer];
 let yields=0;
 const parser={json:doc,createUniqueName:(s:string)=>s,getDependency:async()=>new MeshStandardMaterial(),
  assignFinalMaterial:()=>{const start=performance.now();while(performance.now()-start<5){};},associations:new Map()};
 const plugin=flatNodes(parser as unknown as GLTFParser,buffers,Promise.resolve(),async()=>{yields++;});
 const nodes=await Promise.all([0,1,2].map(i=>plugin.loadNode!(i)));
 assert.equal(nodes.length,3);assert.ok(yields>=2);
 let startupWaits=0;const startupSlice=sceneCpuSlice(undefined,0);assert.equal(startupSlice(),null);
 const liveSlice=sceneCpuSlice(async()=>{startupWaits++;},0);await liveSlice();assert.equal(startupWaits,1);
});
