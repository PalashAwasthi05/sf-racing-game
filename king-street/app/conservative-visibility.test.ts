import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {OcclusionGrid, conservativeVisibility} from './conservative-visibility.ts';

void test('inner raster coverage never treats a partial cell or farther triangle as an occluder', () => {
  const grid = new OcclusionGrid(8, 8);
  grid.rasterize([-1, -1, 10, 9, -1, 10, -1, 9, 10]);
  assert.equal(grid.occluded(.2, .2, 1.8, 1.8, 20), true);
  assert.equal(grid.occluded(3.8, 3.8, 4.2, 4.2, 20), false);
  assert.equal(grid.occluded(.2, .2, 1.8, 1.8, 10), false);
  assert.equal(grid.occluded(.2, .2, 1.8, 1.8, 5), false);
  assert.equal(grid.occluded(7, 7, 8, 8, 20), false);
  grid.clear(); assert.equal(grid.occluded(.2, .2, 1.8, 1.8, 20), false);
});

void test('beauty culling preserves silhouettes, camera-inside, shadows, probes, review and changing visibility', () => {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 16 / 9, .1, 1000);
  const material = new THREE.MeshStandardMaterial();
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), material); wall.position.z = -10; wall.userData.layer = 'buildings';
  const hidden = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), material); hidden.position.set(5, 0, -20);
  const partial = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 2), material); partial.position.set(0, 0, -20);
  const inside = new THREE.Mesh(new THREE.BoxGeometry(50, 50, 50), material);
  scene.add(wall, hidden, partial, inside); scene.updateMatrixWorld(true); camera.updateMatrixWorld();
  let enabled = true;
  const visibility = conservativeVisibility({camera, meshes: [wall, hidden, partial, inside], layerOf: mesh => mesh.userData.layer ?? 'facades', enabled: () => enabled});
  const renderer = {} as THREE.WebGLRenderer, group = new THREE.Group();
  function draw(mesh: THREE.Mesh, view: THREE.Camera = camera) {
    mesh.onBeforeRender(renderer, scene, view, mesh.geometry, mesh.material as THREE.Material, group);
    const count = mesh.geometry.drawRange.count;
    mesh.onAfterRender(renderer, scene, view, mesh.geometry, mesh.material as THREE.Material, group);
    assert.equal(mesh.geometry.drawRange.count, Infinity);
    return count;
  }
  visibility.prepare(); assert.equal(visibility.stats.hidden, 1); assert.equal(draw(hidden), 0);
  assert.equal(draw(partial), Infinity); assert.equal(draw(inside), Infinity); assert.equal(draw(wall), Infinity);
  const probe = camera.clone(); probe.updateMatrixWorld(); assert.equal(draw(hidden, probe), Infinity);
  const light = new THREE.OrthographicCamera();
  hidden.onBeforeShadow(renderer, scene, camera, light, hidden.geometry, material, group);
  assert.equal(hidden.geometry.drawRange.count, Infinity);
  scene.overrideMaterial = new THREE.MeshNormalMaterial(); assert.equal(draw(hidden), Infinity); scene.overrideMaterial.dispose(); scene.overrideMaterial = null;
  enabled = false; visibility.prepare(); assert.equal(draw(hidden), Infinity);
  enabled = true; visibility.prepare(); assert.equal(draw(hidden), 0);
  wall.visible = false; visibility.prepare(); assert.equal(draw(hidden), Infinity);
  wall.visible = true; camera.position.x = 40; camera.updateMatrixWorld(); visibility.prepare(); assert.equal(draw(hidden), Infinity);
  visibility.dispose();
  for (const mesh of [wall, hidden, partial, inside]) mesh.geometry.dispose(); material.dispose();
});

void test('unproductive views skip CPU rasterization and probe again after a camera move', () => {
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
  const material = new THREE.MeshStandardMaterial();
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(10,10), material);
  wall.position.z = -10; scene.add(wall); scene.updateMatrixWorld(true);
  const visibility = conservativeVisibility({camera, meshes:[wall], layerOf:()=>'buildings', enabled:()=>true});
  visibility.prepare(); visibility.prepare(); assert.equal(visibility.stats.ready,true);
  visibility.prepare(); assert.equal(visibility.stats.ready,false);
  assert.equal(wall.geometry.drawRange.count,Infinity);
  camera.position.x += 8; visibility.prepare(); assert.equal(visibility.stats.ready,true);
  visibility.dispose(); wall.geometry.dispose(); material.dispose();
});

void test('masked, stencil-tested and colorless surfaces cannot hide a building behind them',()=>{
  const camera=new THREE.PerspectiveCamera(60,1,.1,100);
  for(const params of [{alphaHash:true},{stencilWrite:true},{colorWrite:false}]){
    const material=new THREE.MeshStandardMaterial(params);
    const wall=new THREE.Mesh(new THREE.PlaneGeometry(30,30),material);
    wall.position.z=-10;
    const visibility=conservativeVisibility({camera,meshes:[wall],layerOf:()=>'buildings',enabled:()=>true});
    assert.equal(visibility.stats.occluders,0);
    visibility.dispose();wall.geometry.dispose();material.dispose();
  }
});
