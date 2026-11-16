import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildFacadeMap} from './facademap.ts';
import type {GlazedWindow} from './glazing.ts';

function lookup(map: ReturnType<typeof buildFacadeMap>, x: number, z: number) {
  const [reach, step, n] = map.grid.toArray();
  const gx = Math.floor((x + reach) / step), gz = Math.floor((z + reach) / step);
  if (gx < 0 || gz < 0 || gx >= n || gz >= n) return [0, 0];
  const lx = gx - map.cellOffset.x, lz = gz - map.cellOffset.y;
  const texture = map.textures.uWmCell;
  const {width, height, data} = texture.image as {width:number;height:number;data:Float32Array};
  if (lx < 0 || lz < 0 || lx >= width || lz >= height) return [0, 0];
  const at = 2 * (lz * width + lx);
  return [data[at], data[at + 1]];
}
const windowAt = (x: number, z: number): GlazedWindow => ({ax:1,az:0,u0:x-3,u1:x+3,y0:1,y1:2,off:z});

test('empty facade maps use a zeroed 1x1 RG32F cell texture', () => {
  const map = buildFacadeMap([], 60);
  const cell = map.textures.uWmCell;
  assert.equal(cell.format, THREE.RGFormat);
  assert.equal(cell.type, THREE.FloatType);
  assert.deepEqual([cell.image.width, cell.image.height], [1, 1]);
  assert.deepEqual(Array.from(cell.image.data as Float32Array), [0, 0]);
  assert.equal(map.grid.w, 0);
  assert.deepEqual(lookup(map, 0, 0), [0, 0]);
  map.dispose();
});

test('cropped cells preserve global lookup bounds for separated occupied groups', () => {
  const reach = 300, map = buildFacadeMap([windowAt(-120, -120), windowAt(120, 120)], reach);
  const cell = map.textures.uWmCell, n = map.grid.z;
  assert.equal(cell.format, THREE.RGFormat);
  assert.ok(cell.image.width < n && cell.image.height < n);
  assert.ok(lookup(map, -120, -120)[1] > 0);
  assert.ok(lookup(map, 120, 120)[1] > 0);
  assert.deepEqual(lookup(map, 0, 0), [0, 0], 'unoccupied gap in the cropped rectangle stays empty');
  assert.deepEqual(lookup(map, -reach-10, 0), [0, 0], 'out-of-world coordinates remain empty');
  assert.deepEqual(lookup(map, 0, -250), [0, 0], 'inside-world cells outside the crop remain empty');
  map.dispose();
});

test('occupied cells spanning both corridor axes retain full bounds', () => {
  const reach = 100, windows:GlazedWindow[] = [];
  for (const z of [-99,-48,0,48,99]) for (const x of [-99,-48,0,48,99]) windows.push(windowAt(x,z));
  const map = buildFacadeMap(windows, reach);
  const cell = map.textures.uWmCell;
  assert.equal(cell.image.width, map.grid.z);
  assert.equal(cell.image.height, map.grid.z);
  assert.ok(lookup(map, -99, -99)[1] > 0);
  assert.ok(lookup(map, 99, 99)[1] > 0);
  map.dispose();
});
