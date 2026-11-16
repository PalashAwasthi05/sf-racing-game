import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { CSM } from 'three/addons/csm/CSM.js';
import { patchCoatedLight } from './coated-light.ts';

void test('CSM fade weights clearcoat in the same branch as the base reflection', () => {
  const csm = new CSM({
    camera: new THREE.PerspectiveCamera(),
    parent: new THREE.Scene(),
  });
  const shader = {
    fragmentShader:
      '#include <lights_physical_pars_fragment>\n' +
      THREE.ShaderChunk.lights_fragment_begin +
      '\n#include <lights_fragment_end>\n/* local lamp additive hook */',
  } as THREE.WebGLProgramParametersWithUniforms;
  patchCoatedLight(shader);
  const snapshot = shader.fragmentShader.indexOf('vec3 previousCoatDirect');
  const direct = shader.fragmentShader.indexOf('RE_Direct(', snapshot);
  const weight = shader.fragmentShader.indexOf(
    'clearcoatSpecularDirect = mix(previousCoatDirect',
  );
  assert.ok(snapshot >= 0 && direct > snapshot && weight > direct);
  assert.equal(
    shader.fragmentShader.match(/vec3 previousCoatDirect/g)?.length,
    1,
  );
  assert.ok(
    shader.fragmentShader.includes('clearcoatSpecularDirect, blendRatio)'),
  );
  assert.ok(
    shader.fragmentShader.indexOf('coatedBeforeSunBase+broadSun') <
      shader.fragmentShader.indexOf('/* local lamp additive hook */'),
  );
  assert.ok(
    shader.fragmentShader.indexOf(
      'coatedBeforeSunBase=reflectedLight.directSpecular',
    ) < shader.fragmentShader.indexOf('vec3 previousCoatDirect'),
  );
  csm.dispose();
});

void test('unsupported CSM layout fails instead of silently duplicating clearcoat light', () => {
  const shader = {
    fragmentShader: '#include <lights_fragment_end>',
  } as THREE.WebGLProgramParametersWithUniforms;
  assert.throws(() => patchCoatedLight(shader), /r180 CSM/);
});
