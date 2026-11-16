import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {loadOverhead} from './overhead.ts';

test('overhead batching keeps pole transforms, wire shapes, lamps and collision positions',async t=>{
  const data={poles:[{x:12,y:-40,z:4,h:6,arm:[1,0]},{x:-300,y:80,z:12,h:8,arm:null}],
    wires:[{p:[[12,-40,10],[-143,20,9],[-300,80,20]],r:.018}]};
  t.mock.method(globalThis,'fetch',async()=>Response.json(data));
  const ordinary=(await loadOverhead('/overhead',false))!,batched=(await loadOverhead('/overhead',true))!;
  assert.deepEqual(batched.circles,ordinary.circles);assert.deepEqual(batched.lamps,ordinary.lamps);
  const src=new THREE.Matrix4(),dst=new THREE.Matrix4();
  for (const original of ordinary.group.children as THREE.Mesh[]){
    const group=batched.group.getObjectByName(original.name) as THREE.BatchedMesh;
    assert.ok(group.isBatchedMesh&&group.perObjectFrustumCulled);assert.equal(group.castShadow,original.castShadow);
    if(original instanceof THREE.InstancedMesh){
      assert.equal(original.count,group.instanceCount);
      for(let i=0;i<original.count;i++){original.getMatrixAt(i,src);group.getMatrixAt(i,dst);assert.ok(src.equals(dst));}
    }else assert.deepEqual(group.geometry.attributes.position.array,original.geometry.attributes.position.array);
  }
});
