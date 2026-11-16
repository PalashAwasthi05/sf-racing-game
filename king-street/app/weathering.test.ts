import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {fileURLToPath} from 'node:url';

const source = readFileSync(fileURLToPath(new URL('./weathering.ts', import.meta.url)), 'utf8');
const surface = source.split('vec3 wxSurface(vec3 albedo){')[1]?.split('vec3 wxBump(')[0] ?? '';

test('render surface phase and tone use continuous world position, not facet normals', () => {
  assert.ok(surface.includes('wxFbm3(P*1.3'));
  assert.ok(surface.includes('wxNoise3(P*190.)'));
  assert.ok(surface.includes('wxFbm3(P*.11)'));
  assert.doesNotMatch(surface, /dot\(P\.xz,T\.xz\)|floor\(N\.xz|\bnq\b|float plane=/);
  assert.doesNotMatch(surface, /vec2\(u,y\)|cell=floor|pick<\.08/);
});

test('window matching still uses the wall normal only to find its pane column', () => {
  assert.ok(surface.includes('ivec2 ent=wmEntries(P)'));
  assert.ok(surface.includes('abs(dot(N.xz,ax))>.3'));
  assert.ok(surface.includes('float uu=dot(P.xz,ax)'));
});
