import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {attachCarLod} from './car-lod.ts';

test('chase view builds only castable depth, first distant view builds reusable shaded geometry',()=>{
  const car=new THREE.Group(), body=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
  body.castShadow=true;body.name='body';car.add(body);
  const lamp=new THREE.Mesh(body.geometry,body.material);lamp.castShadow=false;car.add(lamp);
  const lod=attachCarLod(car,{enabled:()=>true}),camera=new THREE.PerspectiveCamera();
  lod.update(camera);
  assert.equal(lod.stats.farMeshes,0);assert.equal(lod.stats.depthMeshes,1);
  const shadow=car.children.find(c=>c.name==='car-lod-depth') as THREE.Mesh;
  assert.equal(shadow.geometry.getAttribute('normal'),undefined);
  camera.position.z=50;lod.update(camera);
  assert.equal(lod.stats.farMeshes,2);assert.equal(body.visible,false);
  const far=car.children.find(c=>c.name==='body__far') as THREE.Mesh;
  assert.ok(far.geometry.getAttribute('normal'));assert.ok(far.visible);
  lod.depthSwap.enter();assert.equal(far.visible,false);assert.ok(shadow.visible);
  lod.depthSwap.leave();assert.ok(far.visible);assert.equal(shadow.visible,false);
  camera.position.z=1;lod.update(camera);assert.ok(body.visible);assert.equal(far.visible,false);
  camera.position.z=55;lod.update(camera);assert.equal(lod.stats.farMeshes,2);
});
