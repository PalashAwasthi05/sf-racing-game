import type * as THREE from 'three';

/** Permanent physics geometry stays out of render and layer-toggle paths. */
export function collisionOnly(object: THREE.Object3D): boolean {
  return object.userData.collisionOnly === true || object.userData.class === 'static_collision_proxy';
}
