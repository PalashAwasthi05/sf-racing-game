import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {gunzipSync} from 'node:zlib';
import * as THREE from 'three';
import {loadParkedCars} from './parked-cars.ts';

void test('parked cars restore view detail after shadow and occlusion passes and when zooming in', async t => {
  const model = await readFile(new URL('../public/vehicle/parked/cars.bin.gz', import.meta.url));
  const data = {
    types: [{name: 'sedan', width: 2, length: 4, height: 1.5}],
    cars: [{type: 'sedan', x: 0, y: 0, z: 0, heading: 0, pitch: 0, paint: '#bca081'}],
  };
  t.mock.method(globalThis, 'fetch', async (url: string) => url.endsWith('.gz')
    ? new Response(Uint8Array.from(model)) : Response.json(data));
  t.mock.method(THREE.TextureLoader.prototype, 'loadAsync', async () => new THREE.Texture());
  const parked = await loadParkedCars('/cars.json');
  assert.ok(parked);
  assert.equal(parked.circles.length, 9);
  const group = parked.group;
  const [body, glass] = group.children as THREE.BatchedMesh[];
  // The BC7 colour companion gates RGB only: the material's alpha is unused.
  const paint = body.material as THREE.MeshPhysicalMaterial;
  assert.equal(paint.transparent, false);
  assert.equal(paint.alphaTest, 0);
  assert.equal(paint.alphaHash, false);
  assert.equal(paint.alphaToCoverage, false);
  assert.equal(paint.transmission, 0);
  assert.equal(paint.opacity, 1);
  const scene = new THREE.Scene(), view = new THREE.PerspectiveCamera(58, 1, .18, 1000);
  const light = new THREE.OrthographicCamera(-50, 50, 50, -50, .1, 400);
  const renderer = {} as THREE.WebGLRenderer, depth = new THREE.MeshNormalMaterial();
  scene.add(group);
  light.position.set(50, 100, 50); light.lookAt(0, 0, 0); light.updateMatrixWorld();
  const triangles = () => body.getGeometryRangeAt(body.getGeometryIdAt(0))!.count / 3;
  function render(distance: number) {
    view.position.set(0, 0, distance); view.updateMatrixWorld();
    for (const mesh of [glass, body]) {
      mesh.onBeforeRender(renderer, scene, view, mesh.geometry, scene.overrideMaterial ?? mesh.material, group);
    }
    return triangles();
  }
  const near = render(20);
  assert.ok(near > 5_000);
  body.onBeforeShadow(renderer, scene, view, light, body.geometry, depth, group);
  assert.ok(triangles() < near);
  assert.equal(render(20), near);
  scene.overrideMaterial = depth;
  assert.ok(render(20) < near);
  scene.overrideMaterial = null;
  assert.equal(render(20), near);
  const middle = render(75), far = render(180), distant = render(300);
  assert.ok(near > middle && middle > far && far > distant && distant < 150);
  view.zoom = 4; view.updateProjectionMatrix();
  assert.equal(render(300), middle);
});


void test('parked spatial culling shares visibility across materials and restores every camera', async t => {
  const model = await readFile(new URL('../public/vehicle/parked/cars.bin.gz', import.meta.url));
  const data = {
    types: [{name: 'sedan', width: 2, length: 4, height: 1.5}],
    cars: [-300, 0, 300].map(x => ({type: 'sedan', x, y: 0, z: 0, heading: .6, pitch: .1, paint: '#bca081'})),
  };
  t.mock.method(globalThis, 'fetch', async (url: string) => url.endsWith('.gz')
    ? new Response(Uint8Array.from(model)) : Response.json(data));
  t.mock.method(THREE.TextureLoader.prototype, 'loadAsync', async () => new THREE.Texture());
  const parked = (await loadParkedCars('/cars.json'))!;
  const [body, glass] = parked.group.children as THREE.BatchedMesh[];
  const scene = new THREE.Scene(), view = new THREE.PerspectiveCamera(58, 1, .18, 1000);
  const depth = new THREE.MeshNormalMaterial(), light = new THREE.OrthographicCamera(-500, 500, 500, -500, .1, 1000);
  let height = 1125;
  const renderer = {getCurrentViewport: (v: THREE.Vector4) => v.set(0, 0, height, height)} as THREE.WebGLRenderer;
  const draw = () => { for (const mesh of [body, glass]) mesh.onBeforeRender(renderer, scene, view, mesh.geometry, mesh.material, parked.group); };
  view.position.set(0, 3, 20); view.lookAt(0, 1, 0); view.updateMatrixWorld(); draw();
  assert.deepEqual([0, 1, 2].map(i => body.getVisibleAt(i)), [false, true, false]);
  assert.deepEqual([0, 1, 2].map(i => glass.getVisibleAt(i)), [false, true, false]);
  const passes = parked.group.userData.culling.passes;
  draw(); assert.equal(parked.group.userData.culling.passes, passes);
  const full = body.getGeometryIdAt(1);
  height = 128; draw(); assert.notEqual(body.getGeometryIdAt(1), full);
  height = 1125; draw(); assert.equal(body.getGeometryIdAt(1), full);
  light.position.set(0, 500, 0); light.lookAt(0, 0, 0); light.updateMatrixWorld();
  for (const mesh of [body, glass]) mesh.onBeforeShadow(renderer, scene, view, light, mesh.geometry, depth, parked.group);
  assert.deepEqual([0, 1, 2].map(i => body.getVisibleAt(i)), [true, true, true]);
  draw(); assert.deepEqual([0, 1, 2].map(i => body.getVisibleAt(i)), [false, true, false]);
  view.position.x = 300; view.lookAt(300, 1, 0); view.updateMatrixWorld(); draw();
  assert.deepEqual([0, 1, 2].map(i => body.getVisibleAt(i)), [false, false, true]);
});


void test('parked batches preserve every packed normal without float32 expansion', async t => {
  const model = await readFile(new URL('../public/vehicle/parked/cars.bin.gz', import.meta.url));
  const bytes = gunzipSync(model), length = bytes.readUInt32LE(0);
  const header = JSON.parse(bytes.subarray(4, 4 + length).toString()) as {meshes: {count: number; normal: number}[]};
  const start = 4 + length + (4 - length % 4) % 4;
  const expected: string[] = [];
  for (const mesh of header.meshes) {
    const packed = new Int8Array(bytes.buffer, bytes.byteOffset + start + mesh.normal, mesh.count * 4);
    for (let i = 0; i < mesh.count; i++) {
      const vector = [packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2]];
      assert.ok(vector.every(value => value >= -127));
      expected.push(vector.join(','));
    }
  }
  t.mock.method(globalThis, 'fetch', async (url: string) => url.endsWith('.gz')
    ? new Response(Uint8Array.from(model)) : Response.json({types: [{name: 'sedan', width: 2, length: 4, height: 1.5}], cars: [{type: 'sedan', x: 0, y: 0, z: 0, heading: 0, paint: '#bca081'}]}));
  t.mock.method(THREE.TextureLoader.prototype, 'loadAsync', async () => new THREE.Texture());
  const parked = (await loadParkedCars('/cars.json'))!;
  const actual: string[] = [];
  let normalBytes = 0;
  for (const mesh of parked.group.children as THREE.BatchedMesh[]) {
    const normal = mesh.geometry.getAttribute('normal') as THREE.BufferAttribute;
    assert.ok(normal.array instanceof Int8Array);
    assert.equal(normal.normalized, true);
    normalBytes += normal.array.byteLength;
    for (let i = 0; i < normal.count; i++) {
      const raw: number[] = Array.from(normal.array.slice(i * 3, i * 3 + 3));
      actual.push(raw.join(','));
      assert.equal(normal.getX(i), raw[0] / 127);
      assert.equal(normal.getY(i), raw[1] / 127);
      assert.equal(normal.getZ(i), raw[2] / 127);
    }
  }
  assert.deepEqual(actual.sort(), expected.sort());
  assert.equal(normalBytes, expected.length * 3);
  assert.equal(normalBytes / (expected.length * 12), .25);
});
