import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {retireMaterialResources, disposeOwnedTextures} from './material-resources.ts';

test('retired maps release decoded backing and associations while preserving active shared images and Sources', () => {
  let closed = 0;
  const retiredImage = {width: 512, height: 512, close() { closed++; }};
  const keptImage = {width: 512, height: 512, close() { throw new Error('active image closed'); }};
  const unused = new THREE.Texture(retiredImage as unknown as ImageBitmap);
  const duplicate = unused.clone();
  const shared = new THREE.Texture(keptImage as unknown as ImageBitmap);
  const keptClone = shared.clone();
  const separateSource = new THREE.Texture(keptImage as unknown as ImageBitmap);
  const old = new THREE.MeshStandardMaterial({map: unused, normalMap: shared, roughnessMap: separateSource});
  const oldDuplicate = new THREE.MeshStandardMaterial({map: duplicate});
  const active = new THREE.MeshStandardMaterial({map: keptClone});
  const associations = new Map<unknown, unknown>([old, oldDuplicate, active, unused, duplicate, shared, separateSource, keptClone].map(v => [v, {}]));
  retireMaterialResources([old, oldDuplicate], [active], associations);
  assert.equal(closed, 1);
  assert.equal(unused.image, null);
  assert.equal(duplicate.image, null);
  assert.equal(shared.image, keptImage);
  assert.equal(keptClone.image, keptImage);
  assert.equal(separateSource.image, null);
  assert.equal(associations.size, 2);
  assert.ok(associations.has(active));
  assert.ok(associations.has(keptClone));
});


test('retired compressed maps drop DDS bytes without damaging live sampler clones', () => {
  const mipmaps = [{data: new Uint8Array(8), width: 4, height: 4}];
  const retired = new THREE.CompressedTexture(mipmaps, 4, 4, THREE.RGB_S3TC_DXT1_Format);
  const live = retired.clone();
  retireMaterialResources([new THREE.MeshStandardMaterial({map: retired})], [new THREE.MeshStandardMaterial({map: live})]);
  assert.deepEqual(retired.mipmaps, []);
  assert.equal(live.mipmaps[0].data, mipmaps[0].data);
  assert.equal(live.image.width, 4);
});


test('a material still used outside the merged meshes remains live', () => {
  const material = new THREE.MeshStandardMaterial();
  let disposed = false;
  material.addEventListener('dispose', () => {disposed = true;});
  const associations = new Map<unknown, unknown>([[material, {materials: 0}]]);
  retireMaterialResources([material], [material], associations);
  assert.equal(disposed, false);
  assert.ok(associations.has(material));
});

test('viewer cleanup closes owned bitmap once across sampler clones and drops compressed bytes', () => {
  let closed = 0, disposed = 0;
  const image = {close() {closed++;}};
  const texture = new THREE.Texture(image as unknown as ImageBitmap), clone = texture.clone();
  texture.addEventListener('dispose', () => disposed++);
  const compressed = new THREE.CompressedTexture([{data:new Uint8Array(8),width:4,height:4}],4,4,THREE.RGB_S3TC_DXT1_Format);
  disposeOwnedTextures([texture, clone, texture, compressed]);
  assert.equal(closed, 1);assert.equal(disposed, 1);
  assert.equal(texture.image, null);assert.equal(clone.image, null);
  assert.deepEqual(compressed.mipmaps, []);
});

test('viewer cleanup leaves globally cached backing images usable by another viewer', () => {
  let closed = 0;
  const image = {close() {closed++;}};
  const first = new THREE.Texture(image as unknown as ImageBitmap);
  const other = new THREE.Texture(image as unknown as ImageBitmap);
  const wasEnabled=THREE.Cache.enabled;THREE.Cache.enabled=true;THREE.Cache.add('viewer-cleanup-test',image);
  try {
    disposeOwnedTextures([first]);
    assert.equal(closed, 0);assert.equal(first.image, null);assert.equal(other.image, image);
  } finally {THREE.Cache.remove('viewer-cleanup-test');THREE.Cache.enabled=wasEnabled;}
});
