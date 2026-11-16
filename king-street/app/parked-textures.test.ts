import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import * as THREE from 'three';
import {loadParkedAtlasPair, parseParkedBc7} from './parked-textures.ts';

function fixture() {
  const bytes = new ArrayBuffer(24 + 3 * 16), header = new DataView(bytes);
  [0x37434250, 1, 4, 4, 3, 1].forEach((value, i) => header.setUint32(i * 4, value, true));
  return bytes;
}
const capable = {extensions: {has: (name: string) => name === 'EXT_texture_compression_bptc'}} as unknown as Pick<THREE.WebGLRenderer, 'extensions'>;
void test('BC7 parser validates complete chains and color roles before texture creation', () => {
  const bytes = fixture(), texture = parseParkedBc7(bytes, true);
  assert.equal(texture.format, THREE.RGBA_BPTC_Format);
  assert.equal(texture.colorSpace, THREE.SRGBColorSpace);
  assert.equal(texture.flipY, false);
  assert.equal(texture.generateMipmaps, false);
  assert.deepEqual(texture.mipmaps.map(mip => [mip.width, mip.height, mip.data.byteLength]), [[4,4,16],[2,2,16],[1,1,16]]);
  assert.throws(() => parseParkedBc7(bytes, false));
  assert.throws(() => parseParkedBc7(bytes.slice(0, -1), true));
  assert.throws(() => parseParkedBc7(new Uint8Array(bytes.byteLength + 1).buffer, true));
  for (const [field, value] of [[0,0], [1,2], [2,8193], [3,3], [4,2]]) {
    const invalid = bytes.slice(0); new DataView(invalid).setUint32(field * 4, value, true);
    assert.throws(() => parseParkedBc7(invalid, true));
  }
});
void test('unsupported devices request only portable atlas pair', async t => {
  t.mock.method(globalThis, 'fetch', async () => { throw new Error('Unexpected companion request'); });
  const requested: string[] = [];
  t.mock.method(THREE.TextureLoader.prototype, 'loadAsync', async (url: string) => {requested.push(url); return new THREE.Texture();});
  await loadParkedAtlasPair('/vehicle/parked/');
  assert.deepEqual(requested.sort(), ['/vehicle/parked/atlas-colour.webp','/vehicle/parked/atlas-surface.webp']);
});
void test('accepted companion avoids portable color request and corruption falls back per atlas', async t => {
  const requests: string[] = [], portable: string[] = [];
  const file = 'atlas-colour-0123456789abcdef.bc7';
  let corrupt = false;
  t.mock.method(globalThis, 'fetch', async (url: string) => {
    requests.push(url);
    return url.endsWith('.json') ? Response.json({'atlas-colour': {file}}) : new Response(corrupt ? fixture().slice(0,-1) : fixture());
  });
  t.mock.method(THREE.TextureLoader.prototype, 'loadAsync', async (url: string) => {portable.push(url); return new THREE.Texture();});
  const [colour] = await loadParkedAtlasPair('/vehicle/parked/', capable);
  assert.ok((colour as THREE.CompressedTexture).isCompressedTexture);
  assert.deepEqual(portable, ['/vehicle/parked/atlas-surface.webp']);
  assert.deepEqual(requests, ['/vehicle/parked/atlas-bc7.json','/vehicle/parked/'+file]);
  corrupt = true; portable.length = 0;
  await loadParkedAtlasPair('/vehicle/parked/', capable);
  assert.deepEqual(portable.sort(), ['/vehicle/parked/atlas-colour.webp','/vehicle/parked/atlas-surface.webp']);
});

void test('bundled BC7 atlas matches its source identity and complete quality-gated mip chain', async () => {
  const base = new URL('../public/vehicle/parked/', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('atlas-bc7.json', base), 'utf8'));
  const entry = manifest['atlas-colour'];
  assert.ok(entry);
  const [encoded, source] = await Promise.all([readFile(new URL(entry.file, base)), readFile(new URL('atlas-colour.webp', base))]);
  assert.equal(createHash('sha256').update(source).digest('hex'), entry.sourceSha256);
  assert.equal(entry.file, 'atlas-colour-' + createHash('sha256').update(encoded).digest('hex').slice(0,16) + '.bc7');
  const texture = parseParkedBc7(Uint8Array.from(encoded).buffer, true);
  assert.equal(texture.image.width, 2048);
  assert.equal(texture.image.height, 2048);
  assert.equal(texture.mipmaps.length, 12);
  assert.equal(entry.quality.length, texture.mipmaps.length);
  entry.quality.forEach((metric: {rmsByte: number; p99Byte: number; maxByte: number}, i: number) => {
    assert.ok(metric.rmsByte <= (texture.mipmaps[i].width <= 16 ? 4.5 : 3));
    assert.ok(metric.p99Byte <= 10);
    assert.ok(metric.maxByte <= 32);
  });
});
