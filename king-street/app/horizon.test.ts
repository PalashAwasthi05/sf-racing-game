import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {polar, tileHorizonGeometry, compactHorizonAttributes, horizonTileIntersectsFrustum, createHorizonFrustumCache} from './horizon.ts';

void test('horizon tiles preserve exact terrain, attributes, seams and every triangle across cameras', () => {
  const angles = 128, rings = 33, radii = Float32Array.from({length: rings}, (_, i) => 500 * Math.exp(i / 7));
  const meta = {angles, rings} as Parameters<typeof polar>[0];
  const source = polar(meta, (i, j) => 40 * Math.sin(j / angles * Math.PI * 8) + i * 3,
    {radii, hzWater: new Float32Array(rings * angles), hzSlope: new Float32Array(rings * angles), hzNorth: new Float32Array(rings * angles)});
  const tiles = tileHorizonGeometry(source), material = new THREE.MeshBasicMaterial();
  const triangles = (geometries: THREE.BufferGeometry[]) => geometries.flatMap(g => Array.from({length: g.index!.count / 3}, (_, i) =>
    [g.index!.getX(i * 3), g.index!.getX(i * 3 + 1), g.index!.getX(i * 3 + 2)].join(','))).sort();
  assert.deepEqual(triangles(tiles), triangles([source]));
  for (const tile of tiles) {
    for (const [name, attribute] of Object.entries(source.attributes)) assert.equal(tile.getAttribute(name), attribute);
    const point = new THREE.Vector3();
    for (let i = 0; i < tile.index!.count; i++) {
      point.fromBufferAttribute(tile.attributes.position, tile.index!.getX(i));
      assert.ok(tile.boundingBox!.containsPoint(point));
      assert.ok(point.distanceTo(tile.boundingSphere!.center) <= tile.boundingSphere!.radius + 1e-6);
    }
  }
  const camera = new THREE.PerspectiveCamera(58, 16 / 9, .18, 60000), clip = new THREE.Matrix4(), frustum = new THREE.Frustum();
  const meshes = tiles.map(tile => new THREE.Mesh(tile, material));
  const prior = tileHorizonGeometry(source, 16, 2);
  // Reconstruct the previous box-derived sphere, independent of the new wedge bound.
  prior.forEach(tile => tile.boundingBox!.getBoundingSphere(tile.boundingSphere!));
  let before = 0, after = 0;
  for (const heading of [0, Math.PI / 3, Math.PI, 5 * Math.PI / 3]) {
    camera.position.set(0, 5, 0); camera.lookAt(Math.cos(heading) * 1000, 5, Math.sin(heading) * 1000); camera.updateMatrixWorld();
    frustum.setFromProjectionMatrix(clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    const submitted = meshes.filter(mesh => frustum.intersectsObject(mesh)).reduce((sum, mesh) => sum + mesh.geometry.index!.count, 0);
    assert.ok(submitted > 0 && submitted < source.index!.count * .5);
    camera.position.set(150, 5, -100); camera.lookAt(150 + Math.cos(heading) * 1000, 5, -100 + Math.sin(heading) * 1000); camera.updateMatrixWorld();
    frustum.setFromProjectionMatrix(clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    before += prior.filter(tile => frustum.intersectsSphere(tile.boundingSphere!)).reduce((n, tile) => n + tile.index!.count, 0);
    after += tiles.filter(tile => frustum.intersectsSphere(tile.boundingSphere!)).reduce((n, tile) => n + tile.index!.count, 0);
  }
  assert.ok(after < before * .95, 'tighter bands and measured wedge spheres reduce submitted terrain');
  source.dispose(); [...tiles, ...prior].forEach(tile => tile.dispose()); material.dispose();
});

void test('coarse water tiles retain curvature, source buffers and wraparound seams', () => {
  const angles = 64, rings = 17, radii = Float32Array.from({length: rings}, (_, i) => 500 * 1.4 ** i);
  const source = polar({angles, rings} as Parameters<typeof polar>[0], i => -(radii[i] ** 2) / 16000000,
    {radii, hzWater: new Float32Array(angles * rings).fill(1)}, 4);
  const tiles = tileHorizonGeometry(source, 8, 2);
  const triangleKeys = (geometries: THREE.BufferGeometry[]) => geometries.flatMap(g => Array.from({length: g.index!.count / 3}, (_, i) =>
    [g.index!.getX(i * 3), g.index!.getX(i * 3 + 1), g.index!.getX(i * 3 + 2)].join(','))).sort();
  assert.deepEqual(triangleKeys(tiles), triangleKeys([source]));
  for (const tile of tiles) {
    for (const [name, attribute] of Object.entries(source.attributes)) assert.equal(tile.getAttribute(name), attribute);
    const point = new THREE.Vector3();
    for (let i = 0; i < tile.index!.count; i++) {
      point.fromBufferAttribute(tile.attributes.position, tile.index!.getX(i));
      assert.ok(point.distanceTo(tile.boundingSphere!.center) <= tile.boundingSphere!.radius + 1e-6);
    }
  }
  const camera = new THREE.PerspectiveCamera(58, 16/9, .18, 60000), frustum = new THREE.Frustum();
  camera.position.set(0, 5, 0); camera.lookAt(1000, 5, 0); camera.updateMatrixWorld();
  frustum.setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
  const submitted = tiles.filter(tile => frustum.intersectsSphere(tile.boundingSphere!)).reduce((n,tile) => n + tile.index!.count, 0);
  assert.ok(submitted > 0 && submitted < source.index!.count * .65);
  source.dispose(); tiles.forEach(tile => tile.dispose());
});

void test('permanently masked terrain is removed while interpolated shoreline triangles remain',()=>{
  const flags=new Float32Array(24).fill(2);flags[2]=1;flags[3]=1.5;
  const geometry=polar({angles:8,rings:3} as Parameters<typeof polar>[0],()=>0,
    {radii:new Float32Array([10,20,30]),hzWater:flags});
  const indices=Array.from(geometry.index!.array);
  const source=[];
  for(let i=0;i<indices.length;i+=3){
    const tri=indices.slice(i,i+3);
    if(tri.some(index=>flags[index]<=1.5))source.push(tri.join(','));
  }
  const tiles=tileHorizonGeometry(geometry,4);
  const actual=tiles.flatMap(tile=>{
    const ids=Array.from(tile.index!.array),triangles:string[]=[];
    for(let i=0;i<ids.length;i+=3)triangles.push(ids.slice(i,i+3).join(','));
    return triangles;
  });
  assert.ok(actual.length>0&&actual.length<indices.length/3);
  assert.deepEqual(actual.sort(),source.sort());
  assert.equal(new Set(tiles.map(tile => tile.index!.array.buffer)).size, 1);
  assert.equal(tiles[0].index!.array.buffer.byteLength, actual.length * 3 * 4);
  geometry.dispose();tiles.forEach(tile=>tile.dispose());
});

void test('compact fields decode to identical shader float32 values and preserve arbitrary data', () => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('hzWater', new THREE.Float32BufferAttribute([0, 1, 2], 1));
  geometry.setAttribute('hzSlope', new THREE.Float32BufferAttribute([0, 45, 255], 1));
  geometry.setAttribute('hzNorth', new THREE.Float32BufferAttribute([-1, 17 / 127, 1], 1));
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  const position = geometry.attributes.position;
  const before = Object.fromEntries(['hzWater', 'hzSlope', 'hzNorth'].map(name => [name,
    Array.from({length: 3}, (_, i) => Math.fround(geometry.attributes[name].getX(i)))]));
  compactHorizonAttributes(geometry);
  assert.equal(geometry.attributes.position, position);
  for (const name of Object.keys(before)) {
    assert.equal(geometry.attributes[name].array.byteLength, 3);
    assert.deepEqual(Array.from({length: 3}, (_, i) => Math.fround(geometry.attributes[name].getX(i))), before[name]);
  }
  geometry.setAttribute('hzSlope', new THREE.Float32BufferAttribute([1.5, 2, 3], 1));
  const fractional = geometry.attributes.hzSlope;
  compactHorizonAttributes(geometry);
  assert.equal(geometry.attributes.hzSlope, fractional);
  geometry.dispose();
});

void test('every signed source north byte survives compact normalized storage', () => {
  const geometry = new THREE.BufferGeometry();
  const values = Float32Array.from({length: 255}, (_, i) => (i - 127) / 127);
  geometry.setAttribute('hzNorth', new THREE.BufferAttribute(values, 1));
  compactHorizonAttributes(geometry);
  assert.ok(geometry.attributes.hzNorth.array instanceof Int8Array);
  for (let i = 0; i < values.length; i++) assert.equal(Math.fround(geometry.attributes.hzNorth.getX(i)), values[i]);
  geometry.dispose();
});

void test('analytic wedge rejection conservatively separates every actual triangle', () => {
  const angles = 128, rings = 25;
  const source = polar({angles, rings} as Parameters<typeof polar>[0], (i, j) => 700 * Math.sin(j * .31) + i * 5,
    {radii: Float32Array.from({length: rings}, (_, i) => 300 * 1.2 ** i)});
  const tiles = tileHorizonGeometry(source), camera = new THREE.PerspectiveCamera(58, 16/9, .18, 60000);
  const frustum = new THREE.Frustum(), matrix = new THREE.Matrix4(), point = new THREE.Vector3();
  let before = 0, after = 0, rejected = 0;
  for (const eye of [[0,5,0], [200,100,-150], [0,3000,0]]) for (let heading = 0; heading < 24; heading++) {
    camera.position.fromArray(eye); camera.lookAt(eye[0] + Math.cos(heading * Math.PI / 12) * 1000, 0, eye[2] + Math.sin(heading * Math.PI / 12) * 1000);
    camera.updateMatrixWorld();
    frustum.setFromProjectionMatrix(matrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse));
    for (const tile of tiles) {
      const included = horizonTileIntersectsFrustum(tile, frustum);
      if (frustum.intersectsSphere(tile.boundingSphere!)) {
        before += tile.index!.count;
        if (included) after += tile.index!.count;
      }
      if (included) continue;
      rejected++;
      assert.ok(frustum.planes.some(plane => {
        for (let i = 0; i < tile.index!.count; i++) {
          point.fromBufferAttribute(tile.attributes.position, tile.index!.getX(i));
          if (plane.distanceToPoint(point) >= 0) return false;
        }
        return true;
      }));
    }
  }
  assert.ok(rejected > 0);
  assert.ok(after < before * .98);
  source.dispose(); tiles.forEach(tile => tile.dispose());
});

void test('frustum cache shares identical tile matrices and tracks jitter, camera and model changes', () => {
  const cache = createHorizonFrustumCache(), camera = new THREE.PerspectiveCamera(58, 1.6, .1, 60000);
  const reflection = new THREE.PerspectiveCamera(90, 1, .1, 60000), model = new THREE.Matrix4();
  camera.position.set(5, 10, 15); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
  const check = (active: THREE.Camera, transform = model) => {
    const cached = cache.get(active, transform);
    const expected = new THREE.Frustum().setFromProjectionMatrix(new THREE.Matrix4().multiplyMatrices(active.projectionMatrix, active.matrixWorldInverse).multiply(transform));
    assert.deepEqual(cached.planes, expected.planes);
  };
  check(camera);
  for (let i = 0; i < 200; i++) check(camera, model.clone());
  assert.equal(cache.computations, 1);
  check(reflection); check(camera);
  assert.equal(cache.computations, 2);
  camera.projectionMatrix.elements[8] += .0001; check(camera);
  assert.equal(cache.computations, 3);
  camera.projectionMatrix.elements[8] -= .0001; check(camera);
  camera.position.x++; camera.updateMatrixWorld(); check(camera);
  model.makeTranslation(100, 0, 0); check(camera);
  assert.equal(cache.computations, 6);
});

void test('cross-product wedge supports agree with angular reference including wide and full sectors', () => {
  const source = polar({angles: 64, rings: 3} as Parameters<typeof polar>[0], () => 0, {radii: new Float32Array([100, 200, 400])});
  for (const sectors of [1, 2, 3, 16]) {
    const tiles = tileHorizonGeometry(source, sectors);
    for (const tile of tiles) for (let i = 0; i < 180; i++) {
      const theta = i * Math.PI / 90, normal = new THREE.Vector3(Math.cos(theta), 0, -Math.sin(theta));
      const [inner, outer, start, end] = tile.userData.horizonWedge;
      let dot = Math.max(normal.x * Math.cos(start) - normal.z * Math.sin(start), normal.x * Math.cos(end) - normal.z * Math.sin(end));
      const direction = ((Math.atan2(-normal.z, normal.x) - start) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
      if (direction <= end - start) dot = Math.hypot(normal.x, normal.z);
      const support = dot * (dot >= 0 ? outer : inner);
      for (const margin of [-1, 1]) {
        const plane = new THREE.Plane(normal, -support + margin);
        const frustum = new THREE.Frustum(...Array.from({length: 6}, () => plane.clone()));
        assert.equal(horizonTileIntersectsFrustum(tile, frustum), margin > 0);
      }
    }
    tiles.forEach(tile => tile.dispose());
  }
  source.dispose();
});
