import assert from 'node:assert/strict';
import test from 'node:test';
import {gzipSync} from 'node:zlib';
import {loadSceneManifest, downloadScene, sceneForDisplay, sceneStream, withStartupPackage, type SceneManifest} from './scene-package.ts';

test('display inflates independent chunks in order and download keeps the archival GLB', async () => {
  const url = (data: Uint8Array) => 'data:application/octet-stream;base64,' + Buffer.from(data).toString('base64');
  const parts = ['first', 'second', 'third'].map(v => gzipSync(v));
  const archive = gzipSync('original model');
  const manifest: SceneManifest = {
    contentHash: '0123456789ab', chunks: [url(archive)], compressedBytes: archive.length, uncompressedBytes: 14,
  };
  const display: SceneManifest = {format:'gzip-chunks',chunks:parts.map(url),
    compressedBytes:parts.reduce((n,p) => n + p.length,0),uncompressedBytes:16};
  assert.equal(await new Response(sceneStream(display, () => {})).text(), 'firstsecondthird');
  assert.equal(await (await downloadScene(manifest)).text(), 'original model');
});

test('a frozen manifest uses only a valid display file for its own world', async t => {
  const archive: SceneManifest = {contentHash:'0123456789ab',chunks:['/world-0123456789ab-00.bin'],compressedBytes:100};
  const descriptor = {format:'gzip-chunks',version:3,sourceHash:archive.contentHash,contentHash:'aaaaaaaaaaaa',
    chunks:['/world-aaaaaaaaaaaa-00.bin'],compressedBytes:50,uncompressedBytes:200};
  const lookup = t.mock.method(globalThis, 'fetch', async (url: string) => {
    assert.ok(['/startup-v3-0123456789ab.json', '/startup-v2-0123456789ab.json'].includes(url));
    return Response.json(descriptor);
  });
  const found = await withStartupPackage(archive);
  assert.deepEqual(sceneForDisplay(found), descriptor);
  assert.deepEqual(found.chunks, archive.chunks);
  assert.equal(await withStartupPackage(found), found);
  assert.equal(lookup.mock.callCount(), 1);
  for (const invalid of [
    {...descriptor,sourceHash:'bbbbbbbbbbbb'}, {...descriptor,chunks:['https://other.invalid/chunk.bin']},
    {...descriptor,version:7}, {...descriptor,uncompressedBytes:30_000_000},
  ]) {
    lookup.mock.mockImplementation(async () => Response.json(invalid));
    assert.equal(await withStartupPackage(archive), archive);
  }
  lookup.mock.mockImplementation(async () => new Response(null,{status:404}));
  assert.equal(await withStartupPackage(archive), archive);
});

test('missing v3 preserves an inline v2 and frozen worlds can discover v2', async t => {
  const archive: SceneManifest = {contentHash:'0123456789ab',chunks:['/world-0123456789ab-00.bin'],compressedBytes:100};
  const descriptor = {format:'gzip-chunks' as const,version:2,sourceHash:archive.contentHash!,contentHash:'aaaaaaaaaaaa',
    chunks:['/world-aaaaaaaaaaaa-00.bin'],compressedBytes:50,uncompressedBytes:200};
  const requests: string[] = [];
  t.mock.method(globalThis,'fetch',async (url: string) => {
    requests.push(url);
    return url.includes('startup-v2-') ? Response.json(descriptor) : new Response(null,{status:404});
  });
  const inline = {...archive,startup:descriptor};
  assert.equal(await withStartupPackage(inline),inline);
  assert.deepEqual(requests,['/startup-v3-0123456789ab.json']);
  requests.length=0;
  assert.deepEqual(sceneForDisplay(await withStartupPackage(archive)),descriptor);
  assert.deepEqual(requests,['/startup-v3-0123456789ab.json','/startup-v2-0123456789ab.json']);
});


test('normal manifest entry discovers its frozen display copy without changing archival identity', async t => {
  const archive: SceneManifest = {contentHash:'0123456789ab',chunks:['/world-0123456789ab-00.bin'],compressedBytes:100};
  const descriptor = {format:'gzip-chunks',version:3,sourceHash:archive.contentHash,contentHash:'aaaaaaaaaaaa',
    chunks:['/world-aaaaaaaaaaaa-00.bin'],compressedBytes:50,uncompressedBytes:200};
  const previous = Object.getOwnPropertyDescriptor(globalThis,'window');
  Object.defineProperty(globalThis,'window',{configurable:true,value:{location:{href:'https://local.test/?manifest=/compare/frozen.json'}}});
  t.after(()=>{if(previous)Object.defineProperty(globalThis,'window',previous);else delete (globalThis as {window?:unknown}).window;});
  const requests: string[] = [];
  t.mock.method(globalThis,'fetch',async (url: string) => {
    requests.push(url);
    if(url==='/compare/frozen.json') return Response.json(archive);
    assert.equal(url,'/startup-v3-0123456789ab.json');
    return Response.json(descriptor);
  });
  const found = await loadSceneManifest();
  assert.equal(found.contentHash,archive.contentHash);
  assert.deepEqual(found.chunks,archive.chunks);
  assert.deepEqual(sceneForDisplay(found),descriptor);
  assert.deepEqual(requests,['/compare/frozen.json','/startup-v3-0123456789ab.json']);
});

test('S3TC startup is optional and must match both source and portable packages', async t => {
  const {withGpuStartupPackage} = await import('./scene-package.ts');
  const source: SceneManifest = {contentHash: '0123456789ab', chunks: ['/world-0123456789ab-00.bin'], compressedBytes: 100,
    startup: {format: 'gzip-chunks', version: 3, sourceHash: '0123456789ab', contentHash: 'aaaaaaaaaaaa', chunks: ['/world-aaaaaaaaaaaa-00.bin'], compressedBytes: 80, uncompressedBytes: 200}};
  const gpu = {...source.startup!, profile: 's3tc-v1', portableHash: 'aaaaaaaaaaaa', contentHash: 'bbbbbbbbbbbb', chunks: ['/world-bbbbbbbbbbbb-00.bin']};
  const manifest = {...source, startupGpu: gpu};
  assert.equal(sceneForDisplay(manifest), source.startup);
  assert.equal(sceneForDisplay(manifest, true), gpu);
  for (const invalid of [{...gpu, sourceHash: 'cccccccccccc'}, {...gpu, portableHash: 'cccccccccccc'}, {...gpu, profile: 'unknown'}, {...gpu, chunks: ['https://invalid.test/texture.bin']}]) {
    assert.equal(sceneForDisplay({...source, startupGpu: invalid}, true), source.startup);
  }
  const lookup = t.mock.method(globalThis, 'fetch', async () => Response.json(gpu));
  assert.equal(sceneForDisplay(await withGpuStartupPackage(source), true).contentHash, gpu.contentHash);
  assert.equal(lookup.mock.callCount(), 1);
  lookup.mock.mockImplementation(async () => new Response(null, {status: 404}));
  assert.equal(await withGpuStartupPackage(source), source);
});
