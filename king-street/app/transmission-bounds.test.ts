import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {cullTransmission, type MultisampledRenderPass} from './render-budget.ts';

test('transmission reuses static world bounds and invalidates every spatial input', () => {
  const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshPhysicalMaterial({transmission: 1, roughness: 0, thickness: 0}));
  glass.position.z = -5; glass.updateMatrixWorld();
  const candidate = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
  candidate.position.z = -10; candidate.updateMatrixWorld();
  const unculled = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial()); unculled.frustumCulled = false;
  // No WebGL context: prepare only needs target dimensions and installs hooks.
  const scenePass = {target: {width: 256, height: 256}, render() {}} as unknown as MultisampledRenderPass;
  const renderer = {setRenderTarget() {}} as unknown as THREE.WebGLRenderer;
  const cull = cullTransmission({renderer, camera, scenePass, candidates: () => [candidate, unculled], glassRoot: () => glass, enabled: () => true});
  const original = THREE.Sphere.prototype.applyMatrix4;
  let transforms = 0;
  THREE.Sphere.prototype.applyMatrix4 = function (matrix) { transforms++; return original.call(this, matrix); };
  const shown = () => (candidate.layers.mask & (1 << 7)) !== 0;
  try {
    cull.prepare(); assert.equal(shown(), true); assert.equal(transforms, 1);
    camera.position.x = .1; cull.prepare();
    assert.equal(shown(), true); assert.equal(transforms, 1, 'camera motion does not transform static bounds again');
    assert.equal(unculled.geometry.boundingSphere, null, 'unculled objects need no bounds');

    candidate.position.x = 20; candidate.updateMatrixWorld(); cull.prepare();
    assert.equal(shown(), false); assert.equal(transforms, 2);
    candidate.geometry.boundingSphere!.radius = 30; cull.prepare();
    assert.equal(shown(), true); assert.equal(transforms, 3, 'in-place bound edits invalidate');

    const oldGeometry = candidate.geometry;
    candidate.geometry = new THREE.BoxGeometry(1, 1, 1); cull.prepare();
    assert.equal(shown(), false); assert.equal(transforms, 4, 'replacement geometry invalidates');
    oldGeometry.dispose();
    candidate.matrixWorld.elements[12] = 0; cull.prepare();
    assert.equal(shown(), true); assert.equal(transforms, 5, 'manual world matrix edits invalidate');
    cull.prepare(); assert.equal(transforms, 5);
  } finally {
    THREE.Sphere.prototype.applyMatrix4 = original;
    glass.geometry.dispose(); glass.material.dispose(); candidate.geometry.dispose(); candidate.material.dispose(); unculled.geometry.dispose(); unculled.material.dispose();
  }
});
