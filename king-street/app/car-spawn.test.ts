import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {roadSpots,nearestRoadPointIndices} from './car-spawn.ts';

test('bounded road candidates match a full stable sort, including equal distances',()=>{
 let state=7;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/2**32;};
 const p={x:Float32Array.from({length:20000},()=>Math.round((random()-.5)*200)),z:Float32Array.from({length:20000},()=>Math.round((random()-.5)*200))};
 for(const [x,z,limit] of [[0,0,6000],[3.125,-8.25,17],[-200,310,1],[0,0,0],[10,10,25000]]){
  const reference=Array.from({length:p.x.length},(_,i)=>i).sort((a,b)=>Math.hypot(p.x[a]-x,p.z[a]-z)-Math.hypot(p.x[b]-x,p.z[b]-z)).slice(0,limit);
  assert.deepEqual(nearestRoadPointIndices(p,x,z,limit),reference);
 }
});

test('chunked road points match previous coordinates, ordering, and double-precision bucket boundaries',()=>{
 const coordinates:number[]=[];
 for(let i=0;i<20000;i++)coordinates.push((i%200)*.51,0,Math.floor(i/200)*.51);
 coordinates.push(8-1e-8,0,-8-1e-8,8+1e-8,0,-8+1e-8);
 const mesh=new THREE.Mesh(new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(new Float64Array(coordinates),3)));
 mesh.position.set(.1234567,0,-.8765432);mesh.updateMatrixWorld(true);
 const referenceX:number[]=[],referenceZ:number[]=[],seen=new Set<number>(),buckets=new Map<number,number[]>();
 const key=(i:number,j:number)=>(i+4096)*8192+j+4096;
 for(let i=0;i<coordinates.length;i+=3){
  const x=coordinates[i]+mesh.position.x,z=coordinates[i+2]+mesh.position.z,k=key(Math.round(x*2),Math.round(z*2));
  if(seen.has(k))continue;seen.add(k);
  const index=referenceX.length;referenceX.push(x);referenceZ.push(z);
  const b=key(Math.floor(x/8),Math.floor(z/8)),bucket=buckets.get(b);if(bucket)bucket.push(index);else buckets.set(b,[index]);
 }
 const actual=roadSpots([mesh],()=> 'roads');
 assert.deepEqual(actual.x,new Float32Array(referenceX));assert.deepEqual(actual.z,new Float32Array(referenceZ));
 assert.deepEqual(actual.buckets,buckets);assert.equal(actual.centre.x.length,0);
 mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();
});
