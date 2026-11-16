import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {compactStaticIndex, computeStaticBounds, ensureGeometryNormals,finishStaticSteps,packStaticNormals} from './static-detail-geometry.ts';

test('small static meshes narrow 32-bit indices exactly and retain malformed legacy indices', () => {
  const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0, 0, 1, 0], 3));
  geometry.setIndex(new THREE.Uint32BufferAttribute([0, 1, 2], 1));
  compactStaticIndex(geometry);
  assert.ok(geometry.index!.array instanceof Uint16Array);
  assert.deepEqual(Array.from(geometry.index!.array), [0, 1, 2]);
  geometry.setIndex(new THREE.Uint32BufferAttribute([0, 1, 65536], 1));
  compactStaticIndex(geometry);
  assert.ok(geometry.index!.array instanceof Uint32Array);
});

test('packed normal buffers remain shared and retain unit lighting directions',()=>{
 const data=new THREE.Float32BufferAttribute([1,0,0,Math.SQRT1_2,Math.SQRT1_2,0],3);
 const a=new THREE.BufferGeometry().setAttribute('normal',data),b=new THREE.BufferGeometry().setAttribute('normal',data);
 const other=new THREE.Float32BufferAttribute([2,0,0],3),c=new THREE.BufferGeometry().setAttribute('normal',other);
 finishStaticSteps(packStaticNormals([a,b,c]));
 const result=a.getAttribute('normal');assert.equal(result,b.getAttribute('normal'));assert.equal(result.array.byteLength,data.array.byteLength/2);
 assert.equal(result.getX(0),1);assert.ok(Math.abs(result.getX(1)-data.getX(1))<=.5/32767);assert.equal(c.getAttribute('normal'),other);
});

test('generated normals match Three for shared vertices, unindexed triangles and interleaved exports', () => {
  const data = new Float32Array([0, 0, 0, 4, .7, 0, 2, .35, 3, 2.53, 3, 1.23, 0, 0, 0, 0, 0, 0]);
  const packed = new THREE.BufferAttribute(data, 3);
  const interleaved = new THREE.InterleavedBufferAttribute(new THREE.InterleavedBuffer(data, 3), 3, 0);
  for (const [position, indices] of [[packed, [0, 1, 2, 2, 1, 3, 4, 4, 4]], [packed, null], [interleaved, null]] as const) {
    const geometry = new THREE.BufferGeometry().setAttribute('position', position);
    if (indices) geometry.setIndex([...indices]);
    const reference = geometry.clone();
    reference.computeVertexNormals();
    ensureGeometryNormals(geometry);
    assert.deepEqual(geometry.attributes.normal.array, reference.attributes.normal.array);
    const first = geometry.attributes.normal;
    ensureGeometryNormals(geometry);
    assert.equal(geometry.attributes.normal, first);
  }
});

test('static boxes and culling spheres agree with Three for dense and empty meshes', () => {
  for (const points of [[0, 3, 0, 4, .5, 3, 2, 0, 0, -1.75, 5.25, -3], []]) {
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const reference = geometry.clone();
    reference.computeBoundingBox();
    reference.computeBoundingSphere();
    computeStaticBounds(geometry);
    assert.deepEqual(geometry.boundingBox, reference.boundingBox);
    assert.deepEqual(geometry.boundingSphere, reference.boundingSphere);
  }
});

test('live normal slices preserve exact dense indexed normals and allow cancellation',async t=>{
 const {ensureGeometryNormalsAsync}=await import('./static-detail-geometry.ts');
 let clock=0;
 t.mock.method(performance,'now',()=>clock+=1);
 const reference=new THREE.PlaneGeometry(20,20,400,400);reference.deleteAttribute('normal');
 const live=reference.clone();ensureGeometryNormals(reference);let yields=0;
 await ensureGeometryNormalsAsync(live,async()=>{yields++;});
 assert.ok(yields>0);assert.deepEqual(live.attributes.normal.array,reference.attributes.normal.array);
 const cancelled=new THREE.PlaneGeometry(20,20,400,400);cancelled.deleteAttribute('normal');
 await assert.rejects(ensureGeometryNormalsAsync(cancelled,async()=>{throw Error('cancelled');}),/cancelled/);
 assert.equal(cancelled.hasAttribute('normal'),false);
 reference.dispose();live.dispose();cancelled.dispose();
});
