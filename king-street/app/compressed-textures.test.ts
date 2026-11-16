import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {mergeTintVariants} from './material-merge.ts';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';
import {compressedTexturePlugin, releaseTextureCpuImages, supportsStartupS3tc, readTextureCpuImage, mergeSampleImageSources} from './compressed-textures.ts';

const dds = Uint8Array.from(Buffer.from('RERTIHwAAAAHEAoACAAAAAgAAAAsAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAEAAAARFhUMSAAAAAAAAAAAAAAAAAAAAAAAAAACBBAAAAAAAAAAAAAAAAAAAAAAAAQhBCEAAAAABCEEIQAAAAAEIQQhAAAAAAQhBCEAAAAABCEEIQAAAAAEIQQhAAAAAAQhBCEAAAAAA==', 'base64')).buffer;

test('compressed startup requires both linear and sRGB GPU formats', () => {
  const renderer = (has: (name: string) => boolean) => ({extensions: {has}}) as Parameters<typeof supportsStartupS3tc>[0];
  assert.equal(supportsStartupS3tc(renderer(() => true)), true);
  assert.equal(supportsStartupS3tc(renderer(name => !name.endsWith('_srgb'))), false);
  assert.equal(supportsStartupS3tc(renderer(name => name.endsWith('_srgb'))), false);
});

test('DDS plugin preserves mip chain, sampler and exact temporary tint image, then frees the CPU companion', async t => {
  let closed = 0, decoded = 0;
  const cpu = {width: 8, height: 8, close() {closed++;}};
  const previous = globalThis.createImageBitmap;
  t.after(() => { globalThis.createImageBitmap = previous; });
  globalThis.createImageBitmap = (async (_blob: Blob, options: ImageBitmapOptions) => {
    decoded++;
    assert.deepEqual(options, {premultiplyAlpha: 'none', colorSpaceConversion: 'none'});
    return cpu as ImageBitmap;
  }) as typeof createImageBitmap;
  const parser = {json: {images: [{mimeType: 'image/vnd-ms.dds', bufferView: 0, extras: {viewerCpuBufferView: 1, viewerCpuMimeType: 'image/png'}}],
    textures: [{source: 0, sampler: 0}, {source: 0}, {source: 0, sampler: 0}], samplers: [{magFilter: 9728, minFilter: 9986, wrapS: 33071, wrapT: 33648}]},
    associations: new Map(), getDependency: async (_type: string, index: number) => index === 0 ? dds : new ArrayBuffer(0)};
  const plugin = compressedTexturePlugin(parser as unknown as GLTFParser);
  const texture = await plugin.loadTexture(0);
  const other = await plugin.loadTexture(1);
  const duplicate = await plugin.loadTexture(2);
  assert.equal(duplicate, texture);
  assert.equal(other!.source, texture!.source);
  assert.equal(other!.mipmaps[0].data, texture!.mipmaps[0].data);
  assert.ok(texture?.isCompressedTexture);
  assert.equal(texture.flipY, false);
  assert.equal(texture.generateMipmaps, false);
  assert.equal(texture.magFilter, THREE.NearestFilter);
  assert.equal(texture.minFilter, THREE.NearestMipmapLinearFilter);
  assert.equal(texture.wrapS, THREE.ClampToEdgeWrapping);
  assert.equal(texture.wrapT, THREE.MirroredRepeatWrapping);
  assert.equal(texture.format, THREE.RGB_S3TC_DXT1_Format);
  assert.deepEqual(texture.mipmaps.map(m => [m.width, m.height]), [[8, 8], [4, 4], [2, 2], [1, 1]]);
  assert.equal(decoded, 0);
  assert.deepEqual(parser.associations.get(texture), {textures: 0});
  const uvClone = texture.clone();
  assert.equal(await readTextureCpuImage(uvClone, image => image), cpu);
  assert.equal(decoded, 1);
  assert.equal(closed, 1);
  await assert.rejects(readTextureCpuImage(texture, () => {throw new Error('sampling failed');}), /sampling failed/);
  assert.equal(closed, 2);
  assert.equal(releaseTextureCpuImages([new THREE.MeshStandardMaterial({map: texture}), new THREE.MeshStandardMaterial({map: other})]), 1);
  assert.equal(await readTextureCpuImage(uvClone, image => image), null);
  assert.equal(releaseTextureCpuImages([new THREE.MeshStandardMaterial({map: uvClone})]), 0);
  assert.equal(decoded, 2);
  assert.equal(closed, 2);
  assert.ok(texture.mipmaps[0].data.byteLength > 0);
});


test('companion role evidence covers merge maps and keeps unknown materials conservative', () => {
  const json = {textures: [{source: 2}, {source: 3}, {source: 4}, {source: 5}, {source: 6}], materials: [{
    pbrMetallicRoughness: {baseColorTexture: {index: 0}, metallicRoughnessTexture: {index: 1}},
    normalTexture: {index: 2}, occlusionTexture: {index: 3}, emissiveTexture: {index: 4},
  }]};
  assert.deepEqual(mergeSampleImageSources(json), new Set([2, 3, 4, 5]));
  assert.deepEqual(mergeSampleImageSources({...json, materials: [{normalTexture: {index: 2}}]}), new Set());
  assert.equal(mergeSampleImageSources({textures: json.textures}), undefined);
  assert.equal(mergeSampleImageSources({...json, materials: [{extensions: {custom: {}}}]}), undefined);
});

test('DDS used only for emissive shading keeps its GPU data without decoding a CPU companion', async () => {
  const requested: number[] = [];
  const parser = {json: {images: [{mimeType: 'image/vnd-ms.dds', bufferView: 0, extras: {viewerCpuBufferView: 1}}],
    textures: [{source: 0}], materials: [{emissiveTexture: {index: 0}}]}, associations: new Map(),
    getDependency: async (_type: string, index: number) => {requested.push(index); return dds;}};
  const texture = await compressedTexturePlugin(parser as unknown as GLTFParser).loadTexture(0);
  assert.deepEqual(requested, [0]);
  assert.equal(await readTextureCpuImage(texture!, image => image), null);
  assert.ok(texture!.mipmaps[0].data.byteLength > 0);
});


test('async merge decodes only sampled DDS maps, preserves linear tint, and holds at most one bitmap', async t => {
  const oldDecode = globalThis.createImageBitmap, oldDocument = globalThis.document;
  t.after(() => {globalThis.createImageBitmap = oldDecode; globalThis.document = oldDocument;});
  let decoded = 0, open = 0, peak = 0;
  globalThis.createImageBitmap = (async (blob: Blob) => {
    decoded++; open++; peak = Math.max(peak, open);
    const value = new Uint8Array(await blob.arrayBuffer())[0];
    return {width: 8, height: 8, value, close() {open--;}} as unknown as ImageBitmap;
  }) as typeof createImageBitmap;
  let value = 0;
  globalThis.document = {createElement() {return {getContext() {return {
    clearRect() {}, drawImage(image: {value: number}) {value = image.value;},
    getImageData(_x: number, _y: number, w: number, h: number) {
      return {data: Uint8ClampedArray.from({length: w*h*4}, (_, i) => i % 4 === 3 ? 255 : value)};
    },
  };}};}} as unknown as Document;
  const parser = {json: {
    images: [0, 1, 2].map(i => ({mimeType: 'image/vnd-ms.dds', bufferView: 0, extras: {viewerCpuBufferView: i+1}})),
    textures: [{source: 0}, {source: 1}, {source: 2}],
  }, associations: new Map(), getDependency: async (_type: string, index: number) => index === 0 ? dds : Uint8Array.of(index*50).buffer};
  const plugin = compressedTexturePlugin(parser as unknown as GLTFParser);
  const maps = await Promise.all([0, 1, 2].map(i => plugin.loadTexture(i)));
  assert.equal(decoded, 0);
  const materials = maps.map(map => {map!.colorSpace = THREE.SRGBColorSpace; return new THREE.MeshStandardMaterial({map});});
  const meshes = materials.map(material => new THREE.Mesh(
    new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3)), material));
  try {
    const result = await mergeTintVariants(meshes, mesh => mesh === meshes[2] ? 'singleton' : 'render');
    assert.equal(result.replaced.length, 1);
    assert.equal(meshes[1].material, meshes[0].material);
    const linear = (v: number) => ((v/255+.055)/1.055)**2.4;
    assert.ok(Math.abs(meshes[1].geometry.attributes.color.getX(0) - linear(100)/linear(50)) < 1e-5);
    assert.equal(decoded, 2);
    assert.equal(peak, 1);
    assert.equal(open, 0);
  } finally { assert.equal(releaseTextureCpuImages(materials), 3); }
  assert.equal(await readTextureCpuImage(maps[2]!, image => image), null);
});

void test('adjacent DDS cells share exact Source and read-only mip bytes until final eviction, with independent tint companions',async t=>{
 const {createSceneImagePool,retainParserImageSources,releaseParserImageSources}=await import('./scene-image-pool.ts');
 const {disposeOwnedTextures,retireMaterialResources}=await import('./material-resources.ts');
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});let decoded=0,closed=0;
 globalThis.createImageBitmap=(async(blob:Blob)=>{decoded++;const value=new Uint8Array(await blob.arrayBuffer())[0];return {value,close(){closed++;}} as unknown as ImageBitmap;}) as typeof createImageBitmap;
 const make=()=>({json:{images:[{mimeType:'image/vnd-ms.dds',bufferView:0,extras:{viewerCpuBufferView:1,viewerCpuMimeType:'image/png',viewerImageSha256:'a'.repeat(64),viewerCpuImageSha256:'b'.repeat(64)}}],textures:[{source:0}],materials:[{pbrMetallicRoughness:{baseColorTexture:{index:0}}}]},associations:new Map(),getDependency:async(_type:string,index:number)=>index===0?dds:Uint8Array.of(77).buffer}) as unknown as GLTFParser;
 const pool=createSceneImagePool({anisotropy:8}),a=make(),b=make();
 const [first,second]=await Promise.all([compressedTexturePlugin(a,pool).loadTexture(0),compressedTexturePlugin(b,pool).loadTexture(0)]);
 assert.ok(first&&second);assert.equal(first.source,second.source);assert.equal(first.mipmaps,second.mipmaps);assert.equal(first.mipmaps[0].data,second.mipmaps[0].data);
 assert.equal(first.format,THREE.RGB_S3TC_DXT1_Format);assert.equal(first.anisotropy,8);assert.equal(first.source.version,0);
 const shared=second.mipmaps,firstMat=new THREE.MeshStandardMaterial({map:first}),secondMat=new THREE.MeshStandardMaterial({map:second});
 assert.equal(await readTextureCpuImage(first,image=>(image as unknown as {value:number}).value),77);
 assert.equal(releaseTextureCpuImages([firstMat],a),1);
 assert.equal(await readTextureCpuImage(second,image=>(image as unknown as {value:number}).value),77);
 retireMaterialResources([firstMat],[],a.associations);retainParserImageSources(a,[]);
 assert.equal(first.mipmaps.length,0);assert.equal(shared.length,4);assert.ok(shared[0].data.byteLength>0);assert.equal(second.image.width,8);
 assert.equal(releaseTextureCpuImages([secondMat],b),1);assert.equal(await readTextureCpuImage(second,image=>image),null);
 disposeOwnedTextures([second]);releaseParserImageSources(b);releaseParserImageSources(a);
 assert.equal(shared.length,0);assert.equal(second.image,null);assert.equal(decoded,2);assert.equal(closed,2);
});

void test('equal DDS payloads with different exact originals never share a tint sampling Source',async t=>{
 const {createSceneImagePool,releaseParserImageSources}=await import('./scene-image-pool.ts');
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});
 globalThis.createImageBitmap=(async(blob:Blob)=>({value:new Uint8Array(await blob.arrayBuffer())[0],close(){}} as unknown as ImageBitmap)) as typeof createImageBitmap;
 const make=(value:number)=>({json:{images:[{mimeType:'image/vnd-ms.dds',bufferView:0,extras:{viewerCpuBufferView:1,viewerCpuMimeType:'image/png',viewerImageSha256:'a'.repeat(64)}}],textures:[{source:0}]},associations:new Map(),getDependency:async(_type:string,index:number)=>index===0?dds:Uint8Array.of(value).buffer}) as unknown as GLTFParser;
 const pool=createSceneImagePool(),a=make(17),b=make(23);
 const first=await compressedTexturePlugin(a,pool).loadTexture(0),second=await compressedTexturePlugin(b,pool).loadTexture(0);assert.ok(first&&second);
 assert.notEqual(first.source,second.source);
 assert.equal(await readTextureCpuImage(first,image=>(image as unknown as {value:number}).value),17);
 assert.equal(await readTextureCpuImage(second,image=>(image as unknown as {value:number}).value),23);
 releaseParserImageSources(a);releaseParserImageSources(b);
});

void test('a DDS owner released while awaiting its original cannot register a stale companion',async()=>{
 const {createSceneImagePool,releaseParserImageSources}=await import('./scene-image-pool.ts');
 let resolveCompanion!:(value:ArrayBuffer)=>void,ready!:()=>void;
 const waiting=new Promise<void>(resolve=>{ready=resolve;});
 const parser={json:{images:[{mimeType:'image/vnd-ms.dds',bufferView:0,extras:{viewerCpuBufferView:1,viewerCpuMimeType:'image/png',viewerImageSha256:'a'.repeat(64),viewerCpuImageSha256:'b'.repeat(64)}}],textures:[{source:0}]},associations:new Map(),getDependency:async(_type:string,index:number)=>{if(index===0)return dds;ready();return new Promise<ArrayBuffer>(resolve=>{resolveCompanion=resolve;});}} as unknown as GLTFParser;
 const pool=createSceneImagePool();const pending=compressedTexturePlugin(parser,pool).loadTexture(0)!;
 await waiting;releaseParserImageSources(parser);resolveCompanion(Uint8Array.of(44).buffer);
 await assert.rejects(pending,{name:'AbortError'});
 // A fresh parser using the same pool must not receive a cleared/failed Source.
 assert.equal(parser.associations.size,0);const {compressedImagePoolStats}=await import('./compressed-textures.ts');assert.deepEqual(compressedImagePoolStats(pool),{entries:0,references:0});
});


void test('failed DDS loads cannot poison a later cell or leave a strongly retained pool entry',async()=>{
 const {createSceneImagePool,releaseParserImageSources}=await import('./scene-image-pool.ts');
 const {compressedImagePoolStats}=await import('./compressed-textures.ts');
 const invalid=dds.slice(0);new DataView(invalid).setUint32(12,0,true);
 const make=(bytes:ArrayBuffer)=>({json:{images:[{mimeType:'image/vnd-ms.dds',bufferView:0,extras:{viewerImageSha256:'c'.repeat(64)}}],textures:[{source:0}]},associations:new Map(),getDependency:async()=>bytes}) as unknown as GLTFParser;
 const pool=createSceneImagePool(),failed=make(invalid);
 await assert.rejects(compressedTexturePlugin(failed,pool).loadTexture(0)!,/Invalid startup DDS/);releaseParserImageSources(failed);
 assert.deepEqual(compressedImagePoolStats(pool),{entries:0,references:0});
 const valid=make(dds),texture=await compressedTexturePlugin(valid,pool).loadTexture(0);assert.ok(texture?.mipmaps.length);
 assert.deepEqual(compressedImagePoolStats(pool),{entries:1,references:1});releaseParserImageSources(valid);
 assert.equal(texture.mipmaps.length,0);assert.deepEqual(compressedImagePoolStats(pool),{entries:0,references:0});
});
