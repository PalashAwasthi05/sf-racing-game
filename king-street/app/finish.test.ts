import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {finishMaterials} from './finish.ts';

function mesh(name: string, material: THREE.MeshStandardMaterial) {
  const result = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  result.name = name;
  return result;
}

test('legacy textured timber keeps its authored roughness factor', () => {
  const material = new THREE.MeshStandardMaterial({roughness: .84, roughnessMap: new THREE.Texture()});
  finishMaterials([mesh('buildings___timber_dark#a2a88a', material)]);
  assert.equal(material.roughness, .84);
});

test('legacy flat timber still receives its class roughness', () => {
  const material = new THREE.MeshStandardMaterial({roughness: 1});
  finishMaterials([mesh('buildings___timber_dark#a2a88a', material)]);
  assert.equal(material.roughness, .7);
});

test('typed observed recipes keep their authored roughness and normal scales', () => {
  const material = new THREE.MeshStandardMaterial({
    roughness: .9,
    roughnessMap: new THREE.Texture(),
    normalMap: new THREE.Texture(),
  });
  material.normalScale.set(1.15, .85);
  finishMaterials([mesh('buildings___timber_dark~obs1~fam=timber~pat=horizontal_courses~fin=matte#a2a88a', material)]);
  assert.equal(material.roughness, .9);
  assert.deepEqual(material.normalScale.toArray(), [1.15, .85]);
});

test('legacy normal relief applies only when its scale is still the default', () => {
  const authored = new THREE.MeshStandardMaterial({normalMap: new THREE.Texture()});
  authored.normalScale.set(1.2, .8);
  finishMaterials([mesh('buildings___stucco#98a384', authored)]);
  assert.deepEqual(authored.normalScale.toArray(), [1.2, .8]);

  const defaultScale = new THREE.MeshStandardMaterial({normalMap: new THREE.Texture()});
  finishMaterials([mesh('buildings___stucco#98a384', defaultScale)]);
  assert.deepEqual(defaultScale.normalScale.toArray(), [1.3, 1.3]);
});
