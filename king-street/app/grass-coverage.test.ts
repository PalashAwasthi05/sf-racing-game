import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {lawnCoverage} from './grass-coverage.ts';

test('lawn tiers stay off on paved blocks and cover transformed triangles and their edges',()=>{
 const vertices=[0,0,0,80,0,0,0,0,80];
 for(const coords of [vertices,[...vertices.slice(6),...vertices.slice(3,6),...vertices.slice(0,3)]]){
  const geo=new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(coords,3));
  const mesh=new THREE.Mesh(geo);mesh.position.set(100,12,220);const cover=lawnCoverage([mesh]);
  for(let x=0;x<=80;x+=1.5)for(let z=0;z<=80-x;z+=1.5)assert.ok(cover.touches(100+x,220+z,.01));
  assert.ok(cover.touches(99,219,2));assert.equal(cover.touches(177,297,5),false);
  assert.equal(cover.touches(15,16,36),false);
 }
});

test('aggregate bounds conservatively cover displaced instances',()=>{
 const mesh=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial(),1);
 mesh.setMatrixAt(0,new THREE.Matrix4().makeTranslation(-41,3,105));const cover=lawnCoverage([mesh]);
 assert.ok(cover.touches(-41,105,.1));assert.equal(cover.touches(0,0,36),false);
});
