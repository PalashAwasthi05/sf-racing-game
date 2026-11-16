import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {batchStaticWorld} from './static-detail.ts';
import {buildDepthProxies} from './render-budget.ts';

function mesh(material: THREE.Material, x: number) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([x, 0, 0, x + 1, 0, 0, x, 1, 0], 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1], 2));
  geometry.setIndex([0, 1, 2]);
  const result = new THREE.Mesh(geometry, material);
  result.userData.featureId = String(x);
  result.userData.evidence = x ? 'visual-fit' : 'inferred';
  return result;
}
const layer = () => 'buildings';

test('batch and depth merges retain exact source picking, bounds, metadata and local indices using shared vertex storage', () => {
  const material = new THREE.MeshStandardMaterial();
  const sources = [mesh(material, 0), mesh(material, 2), mesh(new THREE.MeshStandardMaterial(), 4)];
  const old = sources.map(m => m.geometry.attributes.position.array.buffer);
  const expected = sources.map(m => Array.from(m.geometry.attributes.position.array));
  const root = new THREE.Group();
  root.add(...sources);
  root.updateMatrixWorld(true);
  const batches = batchStaticWorld(sources, layer, () => '', root);
  assert.equal(batches.length, 1);
  const batch = batches[0];
  for (const [i, source] of sources.slice(0, 2).entries()) {
    assert.equal(source.geometry.attributes.position.array.buffer, batch.geometry.attributes.position.array.buffer);
    assert.notEqual(source.geometry.attributes.position.array.buffer, old[i]);
    assert.deepEqual(Array.from(source.geometry.attributes.position.array), expected[i]);
    assert.deepEqual(Array.from(source.geometry.index!.array), [0, 1, 2]);
    assert.equal(source.userData.featureId, String(i * 2));
    assert.equal(source.geometry.attributes.position.count, 3);
    assert.equal(source.geometry.boundingBox!.min.x, i * 2);
  }
  const depth = buildDepthProxies([batch, sources[2]], layer);
  assert.equal(depth.meshes.length, 1);
  const proxy = depth.meshes[0];
  for (const [i, source] of sources.entries()) {
    assert.equal(source.geometry.attributes.position.array.buffer, proxy.geometry.attributes.position.array.buffer);
    assert.deepEqual(Array.from(source.geometry.attributes.position.array), expected[i]);
  }
  const ray = new THREE.Raycaster(new THREE.Vector3(2.2, .2, 1), new THREE.Vector3(0, 0, -1));
  assert.equal(ray.intersectObjects(sources, false)[0].object, sources[1]);
  assert.deepEqual(batch.userData.members.map((m: THREE.Mesh) => m.geometry.index!.count), [3, 3]);
});

test('singleton depth proxies share GPU attributes and indices; transformed static sources keep local coordinates', () => {
  const material = new THREE.MeshStandardMaterial();
  const a = mesh(material, 0), b = mesh(material, 0);
  b.position.x = 2;
  const root = new THREE.Group(); root.add(a, b); root.updateMatrixWorld(true);
  const original = a.geometry.attributes.position.array;
  const [batch] = batchStaticWorld([a, b], layer, () => '', root);
  assert.equal(a.geometry.attributes.position.array, original);
  assert.deepEqual(Array.from(a.geometry.attributes.position.array), [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const {meshes: [proxy]} = buildDepthProxies([batch], layer);
  assert.equal(proxy.geometry.attributes.position, batch.geometry.attributes.position);
  assert.equal(proxy.geometry.attributes.normal, batch.geometry.attributes.normal);
  assert.equal(proxy.geometry.index, batch.geometry.index);
});

test('shadow-only proxies retain caster positions and omit normals and non-casters',()=>{
 const material=new THREE.MeshStandardMaterial();
 const a=mesh(material,0),b=mesh(material,2),nonCaster=mesh(material,4);
 a.castShadow=b.castShadow=true;nonCaster.castShadow=false;
 const normals=a.geometry.getAttribute('normal').array;
 const depth=buildDepthProxies([a,b,nonCaster],layer,120,()=>false,true);
 assert.deepEqual(depth.skipped,[nonCaster]);assert.equal(depth.meshes.length,1);
 const geometry=depth.meshes[0].geometry;
 assert.equal(geometry.index?.count,6);assert.equal(geometry.getAttribute('position').count,6);
 assert.equal(geometry.getAttribute('normal'),undefined);
 assert.equal(a.geometry.getAttribute('normal').array,normals);
 depth.dispose();
});

test('live static and depth slices preserve exact arrays, bounds and source storage sharing',async()=>{
 const {batchStaticWorldAsync}=await import('./static-detail.ts');const {buildDepthProxiesAsync}=await import('./render-budget.ts');
 const fixture=()=>{
  const material=new THREE.MeshStandardMaterial(),root=new THREE.Group();
  const sources=[0,1,2].map((i)=>{const g=new THREE.PlaneGeometry(20,20,200,200);g.translate(i*22,0,0);const m=new THREE.Mesh(g,i<2?material:new THREE.MeshStandardMaterial());m.castShadow=true;root.add(m);return m;});
  root.updateMatrixWorld(true);return {root,sources};
 };
 const ordinary=fixture(),live=fixture();
 const expected=batchStaticWorld(ordinary.sources,layer,()=>'',ordinary.root);let batchYields=0,depthYields=0;
 const actual=await batchStaticWorldAsync(live.sources,layer,()=>'',live.root,async()=>{batchYields++;});
 assert.ok(batchYields>0);assert.equal(actual.length,expected.length);
 const same=(a:THREE.BufferGeometry,b:THREE.BufferGeometry)=>{
  assert.deepEqual(Object.keys(a.attributes),Object.keys(b.attributes));
  for(const name of Object.keys(a.attributes))assert.deepEqual(a.attributes[name].array,b.attributes[name].array);
  assert.deepEqual(a.index!.array,b.index!.array);assert.deepEqual(a.boundingBox,b.boundingBox);assert.deepEqual(a.boundingSphere,b.boundingSphere);
 };
 same(actual[0].geometry,expected[0].geometry);
 const before=live.sources.map(m=>m.geometry.attributes.position.array.slice());
 const depthExpected=buildDepthProxies([...expected,ordinary.sources[2]],layer,120,()=>false,true);
 const depthActual=await buildDepthProxiesAsync([...actual,live.sources[2]],layer,120,()=>false,true,async()=>{depthYields++;});
 assert.ok(depthYields>0);assert.equal(depthActual.meshes.length,depthExpected.meshes.length);
 for(let i=0;i<depthActual.meshes.length;i++)same(depthActual.meshes[i].geometry,depthExpected.meshes[i].geometry);
 for(const [i,m] of live.sources.entries()){
  assert.deepEqual(m.geometry.attributes.position.array,before[i]);
  assert.equal(m.geometry.attributes.position.array.buffer,depthActual.meshes[0].geometry.attributes.position.array.buffer);
 }
 depthExpected.dispose();depthActual.dispose();
});
