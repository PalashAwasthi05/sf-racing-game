import assert from 'node:assert/strict';
import test from 'node:test';
import {routePackage} from './route-package.ts';

const archive = {contentHash: '1d2ae3f6566f', chunks: [], compressedBytes: 1, cars:'/compare/cars.json', overhead:'/compare/overhead.json'};
const trial = {routeEN: [[199.0053, -282.3433], [54.1648, -306.4548]]};
const valid = () => ({scope: 'route-block', sourceWorldHash: archive.contentHash,
  routeEN: trial.routeEN.map(point => [...point]), contentHash: 'b156e43a3d14', format: 'gzip-chunks',
  chunks: ['/world-b156e43a3d14-00.bin'], compressedBytes: 10, uncompressedBytes: 28,
  world: '/compare/route-block-world-123.json', cars: '/compare/route-block-cars-123.json',
  overhead: '/compare/route-block-overhead-123.json'});

test('accepts the exact trial world and numeric route without changing the package', () => {
  const manifest = valid();
  assert.equal(routePackage(manifest, archive, trial), manifest);
});

test('rejects stale packages and route mismatches without numeric coercion', () => {
  for (const changes of [
    {scope: 'neighborhood'}, {sourceWorldHash: '65b440720d5f'},
    {routeEN: [[199.0053, -282.3433], [54.1648, -306.4547]]},
    {routeEN: [['199.0053', -282.3433], [54.1648, -306.4548]]},
    {routeEN: [[199.0053, NaN], [54.1648, -306.4548]]}, {routeEN: []},
  ]) assert.equal(routePackage({...valid(), ...changes}, archive, trial), null);
});

test('requires complete local chunk descriptors and positive finite byte sizes', () => {
  for (const changes of [
    {chunks: []}, {chunks: ['/world-b156e43a3d15-00.bin']},
    {chunks: ['https://example.com/world-b156e43a3d14-00.bin']},
    {chunks: ['/world-b156e43a3d14-01.bin']}, {contentHash: '../bad'},
    {compressedBytes: 0}, {compressedBytes: Infinity}, {compressedBytes: -1},
    {uncompressedBytes: NaN}, {uncompressedBytes: 0}, {uncompressedBytes: 8 * 1024 * 1024 + 1},
    {world: '//example.com/world.json'}, {cars: '/compare/../cars.json'},
    {overhead: 'https://example.com/overhead.json'}, {cars: undefined},
  ]) assert.equal(routePackage({...valid(), ...changes}, archive, trial), null);
  assert.equal(routePackage(null, archive, trial), null);
});

test('a world without parked cars or overhead sources can still use its small package', () => {
  const manifest = {...valid(), cars:undefined, overhead:undefined};
  assert.equal(routePackage(manifest, {...archive, cars:undefined, overhead:undefined}, trial), manifest);
});
