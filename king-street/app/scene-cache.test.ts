import test from 'node:test';
import assert from 'node:assert/strict';
import {gzipSync} from 'node:zlib';
import {displayChunk} from './scene-cache.ts';

test('a saved chunk loads without contacting its URL and a damaged copy is repaired', async () => {
  const bytes = new Uint8Array(gzipSync('geometry'));
  const url = 'data:application/octet-stream;base64,' + Buffer.from(bytes).toString('base64');
  const headers = {'X-World-Chunk-Bytes': String(bytes.byteLength)};
  let saved = new Response('corrupt compressed data', {headers});
  const cache = {
    match: async () => saved.clone(),
    put: async (_url: string, response: Response) => { saved = response; },
  };
  const signal = new AbortController().signal;
  const repaired = await displayChunk(url, cache, signal);
  assert.equal(repaired.compressedSize, bytes.byteLength);
  assert.equal(new TextDecoder().decode(repaired.buffer), 'geometry');
  const offline = await displayChunk('unreachable:', cache, signal);
  assert.equal(new TextDecoder().decode(offline.buffer), 'geometry');
});

test('static-host chunk is streamed, counted and persisted before permitting the next chunk', async t => {
  const bytes = gzipSync('street detail');
  const headers = {'Content-Length': String(bytes.byteLength)};
  t.mock.method(globalThis, 'fetch', async () => new Response(new Uint8Array(bytes), {headers}));
  let persisted = false;
  const cache = {
    match: async () => undefined,
    async put(_key: string, response: Response) {
      assert.equal(response.headers.get('X-World-Chunk-Bytes'), String(bytes.byteLength));
      assert.deepEqual(new Uint8Array(await response.arrayBuffer()), new Uint8Array(bytes));
      await new Promise(resolve => setTimeout(resolve, 5));
      persisted = true;
    },
  };
  const result = await displayChunk('/sample', cache, new AbortController().signal);
  assert.equal(new TextDecoder().decode(result.buffer), 'street detail');
  assert.equal(persisted, true);
  assert.equal(result.compressedSize, bytes.byteLength);
});
