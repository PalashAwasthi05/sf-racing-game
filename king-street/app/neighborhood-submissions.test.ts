import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {batchNeighborhoodSubmissions} from './neighborhood-submissions.ts';

const layerOf = (o: THREE.Object3D) => String(o.userData.layer);
function fixture(count = 3) {
  const root = new THREE.Group(), parent = new THREE.Group(); root.add(parent);
  parent.position.set(10, 2, -4); parent.rotation.y = .35;
  const material = new THREE.MeshStandardMaterial({vertexColors: false});
  const sources = Array.from({length: count}, (_, i) => {
    const geometry = new THREE.BoxGeometry(1, 2, 1); geometry.clearGroups();
    const mesh: THREE.Mesh<THREE.BufferGeometry, THREE.Material> = new THREE.Mesh(geometry, material);
    mesh.position.set(i*5, 1, -10);
    mesh.userData = {class: 'distant_exterior', bakedAppearance: true, layer: 'roofs', streamCell: 'cell-'+i};
    mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
  });
  root.updateMatrixWorld(true);
  return {root, parent, sources};
}

test('exact shared material uses one multi-draw with original transforms and independent cell visibility', () => {
  const {root, sources} = fixture(); const worlds = sources.map(s => s.matrixWorld.clone());
  const submissions = batchNeighborhoodSubmissions(sources, root, {supported: true, layerOf});
  assert.equal(submissions.meshes.length, 1);
  assert.equal(submissions.stats.sourceMeshes, 3);
  const batch = submissions.meshes[0], matrix = new THREE.Matrix4();
  for (let i = 0; i < sources.length; i++) {
    batch.getMatrixAt(i, matrix);
    matrix.elements.forEach((v, k) => assert.ok(Math.abs(v-worlds[i].elements[k]) < 1e-5));
    sources[i].updateWorldMatrix(true, false);
    assert.deepEqual(sources[i].matrixWorld.elements, worlds[i].elements);
    assert.equal(sources[i].userData.streamCell, 'cell-'+i);
    assert.equal(sources[i].geometry.attributes.position.array.buffer, batch.geometry.attributes.position.array.buffer);
  }
  sources[1].visible = false; submissions.sync();
  assert.equal(batch.getVisibleAt(0), true); assert.equal(batch.getVisibleAt(1), false); assert.equal(batch.getVisibleAt(2), true);
  assert.equal(batch.perObjectFrustumCulled, true); assert.equal(batch.castShadow, true);
  submissions.dispose();
});

test('beauty-only depth callback suppresses an instance without leaking into shadow camera', () => {
  const {root, sources} = fixture(2);
  const submissions = batchNeighborhoodSubmissions(sources, root, {supported: true, layerOf});
  const camera = new THREE.PerspectiveCamera(70, 1, .1, 1000), shadow = camera.clone(), scene = new THREE.Scene();
  camera.updateMatrixWorld(); shadow.updateMatrixWorld();
  sources[0].onBeforeRender = (_r, _s, view) => {if (view === camera) sources[0].geometry.setDrawRange(0, 0);};
  sources[0].onAfterRender = () => {sources[0].geometry.setDrawRange(0, Infinity);};
  const batch = submissions.meshes[0], renderer = {} as THREE.WebGLRenderer;
  batch.onBeforeRender(renderer, scene, camera, batch.geometry, batch.material as THREE.Material, null as unknown as THREE.Group);
  assert.equal(batch.getVisibleAt(0), false); assert.equal(batch.getVisibleAt(1), true);
  assert.equal(sources[0].geometry.drawRange.count, Infinity);
  batch.onBeforeShadow(renderer, scene, camera, shadow, batch.geometry, batch.material as THREE.Material, null as unknown as THREE.Group);
  assert.equal(batch.getVisibleAt(0), true);
  submissions.dispose();
});

test('evidence fallback restores original controllers and leaves source geometry/materials intact', () => {
  const {root, sources} = fixture(2), originals = sources.map(s => s.geometry);
  const submissions = batchNeighborhoodSubmissions(sources, root, {supported: true, layerOf});
  const evidence = new THREE.MeshBasicMaterial({color: 'red'});
  submissions.setEnabled(false);
  sources[0].material = evidence;
  assert.ok(sources[0].parent); assert.equal(submissions.meshes[0].parent?.visible, false);
  sources.forEach((s, i) => assert.equal(s.geometry, originals[i]));
  assert.equal(sources[0].material, evidence);
  submissions.setEnabled(true); assert.equal(sources[0].parent, null);
  assert.equal(submissions.meshes[0].parent?.visible, true);
  let retired = 0; originals.forEach(g => g.addEventListener('dispose', () => retired++));
  submissions.dispose(); submissions.dispose(); assert.equal(retired, 0);
});

test('unsupported devices keep original submissions and geometry layouts are not mixed', () => {
  const {root, parent, sources} = fixture(2);
  const unchanged = batchNeighborhoodSubmissions(sources, root, {supported: false, layerOf});
  assert.equal(unchanged.meshes.length, 0); assert.equal(sources[0].parent, parent); unchanged.dispose();
  const secondMaterial = (sources[0].material as THREE.Material).clone(); sources[1].material = secondMaterial as THREE.MeshStandardMaterial;
  const separate = batchNeighborhoodSubmissions(sources, root, {supported: true, layerOf});
  assert.equal(separate.meshes.length, 0); separate.dispose();
});

test('releases bake-only uploads while preserving live shared buffers and the evidence fallback',()=>{
  const {root,sources}=fixture(), scene=new THREE.Scene();scene.add(root);
  const submissions=batchNeighborhoodSubmissions(sources,root,{supported:true,layerOf});
  const shared=new THREE.BufferGeometry();
  shared.setAttribute('position',sources[0].geometry.getAttribute('position'));
  scene.add(new THREE.Mesh(shared));scene.add(new THREE.Mesh(sources[1].geometry));
  const retired=new Set<THREE.BufferGeometry>();
  for(const source of sources)source.geometry.addEventListener('dispose',()=>retired.add(source.geometry));
  submissions.setEnabled(false);submissions.releaseBakeBuffers(scene);assert.equal(retired.size,0);
  submissions.setEnabled(true);submissions.releaseBakeBuffers(scene);
  assert.deepEqual([...retired],[sources[2].geometry]);
  submissions.setEnabled(false);assert.ok(sources.every(s=>s.parent&&s.geometry.getAttribute('position').count>0));
  submissions.dispose();
});
