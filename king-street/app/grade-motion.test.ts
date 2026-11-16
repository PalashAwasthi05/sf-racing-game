import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import type {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {createGrade} from './grade.ts';

test('parking disables the shutter immediately, including residual camera movement',()=>{
 const grade=createGrade({addPass(){}} as unknown as EffectComposer,undefined,undefined,undefined,new THREE.DepthTexture(16,16));
 const camera=new THREE.PerspectiveCamera(58,1.6,.18,60000);
 const car=new THREE.Group();
 const mesh=new THREE.Mesh(new THREE.BoxGeometry(2,1.5,4));car.add(mesh);
 camera.position.set(0,2,8);camera.lookAt(0,0,0);
 grade.motion(camera,car,60,1/60);
 camera.position.z-=.1;grade.motion(camera,car,60,1/60);
 assert.ok(grade.composite.material.uniforms.uShutter.value>0);
 camera.position.x+=.01;grade.motion(camera,car,0,1/60);
 assert.equal(grade.composite.material.uniforms.uShutter.value,0);
 grade.motion(camera,car,60,1/60); // Restart needs fresh history, so no old moving trail survives.
 assert.equal(grade.composite.material.uniforms.uShutter.value,0);
 grade.dispose();mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();
});
