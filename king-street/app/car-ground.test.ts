import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildCarGround,Surface} from './car-ground.ts';

test('collision surfaces retain only accepted triangles when most input capacity is unused', () => {
  const positions: number[] = [];
  // One usable road triangle and nine roofs above the drivable band.
  for (let i = 0; i < 10; i++) positions.push(0, i ? 10 : .1, 0, 2, i ? 10 : .1, 0, 0, i ? 10 : .1, 2);
  const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mesh = new THREE.Mesh(geometry);
  const ground = buildCarGround([mesh], () => 'roads', () => 0, 4);
  const triangles = (ground.surface as unknown as {tris: Float32Array}).tris;
  assert.equal(ground.stats.surfaceTriangles, 1);
  assert.equal(triangles.length, 9);
  assert.equal(triangles.buffer.byteLength, triangles.byteLength);
  assert.ok(Math.abs(ground.surface.at(.25, .25) - .1) < 1e-6);
  assert.equal(ground.surface.at(3, 3), 0);
});

test('hidden permanent collision proxies survive layer visibility and repeated object references', () => {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(.2, 3, .2));
  mesh.position.set(1, 1.5, 1);
  mesh.visible = false;
  mesh.userData = {class: 'static_collision_proxy', collisionOnly: true};
  const single = buildCarGround([mesh], () => 'not-a-display-layer', () => 0, 4);
  const repeated = buildCarGround([mesh, mesh], () => 'not-a-display-layer', () => 0, 4);
  assert.ok(single.obstacles.count > 0);
  assert.equal(repeated.obstacles.count, single.obstacles.count);
  assert.equal(repeated.stats.surfaceTriangles, 0);
  assert.equal(mesh.visible, false);
});

test('the alpha-masked tree silhouette adds no obstacle beside the real trunk', () => {
  const trunk = new THREE.Mesh(new THREE.BoxGeometry(.25, 3, .25));
  trunk.userData.collisionOnly = true;
  trunk.position.y = 1.5;
  const crown = new THREE.Mesh(new THREE.PlaneGeometry(12, 8));
  crown.userData.class = 'distant_exterior';
  crown.position.y = 4;
  const count = (objects: THREE.Mesh[]) => buildCarGround(objects, () => 'vegetation', () => 0, 10).obstacles.count;
  assert.equal(count([crown]), 0);
  assert.ok(count([trunk]) > 0);
  assert.equal(count([trunk, crown]), count([trunk]));
});

test('triangle-local transforms reproduce the previous full world-position raster exactly',()=>{
 const geometry=new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(new Float64Array([
  -2,.1,-2,2,.17,-2,2,.24,2,-2,.11,2]),3));
 geometry.setIndex([0,1,2,0,2,3]);
 const mesh=new THREE.Mesh(geometry);mesh.position.set(.12345678,.02,-.7654321);
 mesh.rotation.y=.37;mesh.scale.set(1.2,1,.8);mesh.updateWorldMatrix(true,false);
 const source=geometry.getAttribute('position'),e=mesh.matrixWorld.elements,world=new Float32Array(source.count*3);
 for(let i=0;i<source.count;i++){
  const x=source.getX(i),y=source.getY(i),z=source.getZ(i);
  world[i*3]=e[0]*x+e[4]*y+e[8]*z+e[12];
  world[i*3+1]=e[1]*x+e[5]*y+e[9]*z+e[13];
  world[i*3+2]=e[2]*x+e[6]*y+e[10]*z+e[14];
 }
 const triangles=Float32Array.from([0,1,2,0,2,3].flatMap(i=>Array.from(world.subarray(i*3,i*3+3))));
 const actual=buildCarGround([mesh],()=> 'roads',()=>0,8,.25);
 const reference=new Surface(8,.25,triangles,()=>0);
 assert.deepEqual((actual.surface as unknown as {tris:Float32Array}).tris,triangles);
 for(let z=-4;z<4;z+=.137)for(let x=-4;x<4;x+=.193)assert.equal(actual.surface.at(x,z),reference.at(x,z));
 mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();
});
