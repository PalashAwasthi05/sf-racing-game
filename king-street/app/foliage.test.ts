import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildFoliage} from './foliage.ts';

void test('smaller foliage cells preserve every seeded card and cull offscreen crowns', () => {
  const geometry = new THREE.BufferGeometry();
  const centres: number[] = [];
  for (let x = 1; x < 80; x += 2) for (let z = 1; z < 80; z += 2) centres.push(x, 5, z);
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(centres, 3));
  const source = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: '#557c38'}));
  source.userData.foliageFanCentres = true;
  const texture = new THREE.Texture(), wide = buildFoliage([source], texture, 80), local = buildFoliage([source], texture);
  const cards = (meshes: THREE.Mesh[]) => meshes.flatMap(mesh => {
    const g = mesh.geometry;
    return Array.from({length: g.attributes.position.count / 4}, (_, i) =>
      ['position', 'normal', 'cardNormal', 'color', 'uv'].map(name => {
        const a = g.getAttribute(name) as THREE.BufferAttribute;
        return Array.from(a.array.slice(i * 4 * a.itemSize, (i + 1) * 4 * a.itemSize));
      })).map(card => JSON.stringify(card));
  }).sort();
  assert.deepEqual(cards(local.meshes), cards(wide.meshes));
  assert.equal(local.meshes.length, 9);
  const camera = new THREE.PerspectiveCamera(35, 1, .1, 25);
  camera.position.set(5, 5, -10); camera.lookAt(5, 5, 10); camera.updateMatrixWorld();
  const clip = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const frustum = new THREE.Frustum().setFromProjectionMatrix(clip);
  const submitted = (meshes: THREE.Mesh[]) => meshes.reduce((n, mesh) => n + (frustum.intersectsObject(mesh) ? mesh.geometry.index!.count / 3 : 0), 0);
  assert.ok(submitted(local.meshes) < submitted(wide.meshes) / 2);
  for (const mesh of [...local.meshes, ...wide.meshes]) {
    assert.equal(mesh.castShadow, true); assert.equal(mesh.receiveShadow, true);
    mesh.geometry.dispose();
  }
  local.material.dispose(); wide.material.dispose(); texture.dispose(); geometry.dispose(); source.material.dispose();
});
