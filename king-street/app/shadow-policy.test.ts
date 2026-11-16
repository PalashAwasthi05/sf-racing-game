import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildDepthProxies} from './render-budget.ts';
import {NO_SUN_SHADOW,smallShadowDetail} from './shadow-policy.ts';

test('opaque window backing remains in distant building shadow proxies', () => {
  const backing = new THREE.Mesh(new THREE.BoxGeometry(2, 3, .035), new THREE.MeshStandardMaterial());
  backing.name = 'interior';
  backing.castShadow = !NO_SUN_SHADOW.test(backing.name);
  backing.position.set(500, 12, 10); backing.updateMatrixWorld();
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(2, 3), new THREE.MeshStandardMaterial());
  glass.name = 'glass'; glass.castShadow = !NO_SUN_SHADOW.test(glass.name);
  glass.position.set(500, 12, 10.2); glass.updateMatrixWorld();
  const proxies = buildDepthProxies([backing, glass], () => 'facades', 120,
    mesh => smallShadowDetail(mesh.name, 'facades'));
  const opaque = proxies.meshes.filter(mesh => mesh.castShadow);
  assert.equal(opaque.length, 1);
  assert.equal(opaque[0].userData.detail, false); // not distance-culled as trim
  const ray = new THREE.Raycaster(new THREE.Vector3(500, 12, 12), new THREE.Vector3(0, 0, -1));
  assert.ok(ray.intersectObjects(opaque).length > 0);
  assert.equal(proxies.meshes.filter(mesh => !mesh.castShadow).length, 1);
  proxies.dispose(); backing.geometry.dispose(); glass.geometry.dispose();
});
