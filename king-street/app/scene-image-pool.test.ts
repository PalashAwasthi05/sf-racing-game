import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import type {GLTFParser} from 'three/addons/loaders/GLTFLoader.js';
import {createSceneImagePool,pooledImagePlugin,retainParserImageSources,releaseParserImageSources} from './scene-image-pool.ts';
import {disposeOwnedTextures,retireMaterialResources} from './material-resources.ts';

const hash='a'.repeat(64);
function parser(withHash=true){
 let reads=0;
 const json={images:[{bufferView:0,mimeType:'image/png',extras:withHash?{viewerImageSha256:hash}:undefined}],textures:[{source:0,sampler:0},{source:0,sampler:1}],samplers:[{magFilter:9728,minFilter:9986,wrapS:33071,wrapT:33648},{}]};
 const value={json,textureLoader:{isImageBitmapLoader:true},associations:new Map(),async getDependency(){reads++;return Uint8Array.of(1,2,3).buffer;}} as unknown as GLTFParser;
 return {value,reads:()=>reads};
}
void test('neighboring cells share exact decode and Source while preserving sampler roles and stable upload version',async t=>{
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});
 let decodes=0,closed=0;
 const image={width:8,height:8,close(){closed++;}};
 globalThis.createImageBitmap=(async(_blob:Blob,options:ImageBitmapOptions)=>{
  decodes++;assert.deepEqual(options,{premultiplyAlpha:'none',colorSpaceConversion:'none'});return image as ImageBitmap;
 }) as typeof createImageBitmap;
 const pool=createSceneImagePool({anisotropy:8}),a=parser(),b=parser();
 const pa=pooledImagePlugin(a.value,pool),pb=pooledImagePlugin(b.value,pool);
 const [first,second]=await Promise.all([pa.loadTexture(0),pb.loadTexture(0)]);
 assert.ok(first&&second);assert.notEqual(first,second);assert.equal(first.source,second.source);
 const version=first.source.version;const alternate=await pb.loadTexture(1);assert.ok(alternate);
 assert.equal(alternate.source,first.source);assert.equal(first.source.version,version);
 assert.equal(first.version,1);assert.equal(first.anisotropy,8);assert.equal(first.flipY,false);
 assert.equal(first.magFilter,THREE.NearestFilter);assert.equal(first.minFilter,THREE.NearestMipmapLinearFilter);
 assert.equal(first.wrapS,THREE.ClampToEdgeWrapping);assert.equal(first.wrapT,THREE.MirroredRepeatWrapping);
 assert.equal(alternate.magFilter,THREE.LinearFilter);assert.equal(alternate.wrapS,THREE.RepeatWrapping);
 assert.equal(decodes,1);assert.equal(a.reads()+b.reads(),1);assert.deepEqual(pool.stats(),{entries:1,references:2});
 const retired=new THREE.MeshStandardMaterial({map:first});
 retireMaterialResources([retired],[],a.value.associations);retainParserImageSources(a.value,[]);
 assert.equal(closed,0);assert.equal(second.image,image);assert.deepEqual(pool.stats(),{entries:1,references:1});
 // The remaining bitmap can be re-uploaded after a context reset.
 assert.equal(second.image.width,8);
 disposeOwnedTextures([second,alternate]);releaseParserImageSources(b.value);releaseParserImageSources(a.value);
 assert.equal(closed,1);assert.equal(second.image,null);assert.deepEqual(pool.stats(),{entries:0,references:0});
});

void test('old packages deduplicate exact embedded bytes without metadata',async t=>{
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});let decodes=0,closed=0;
 globalThis.createImageBitmap=(async()=>{decodes++;return {close(){closed++;}} as ImageBitmap;}) as typeof createImageBitmap;
 const pool=createSceneImagePool(),a=parser(false),b=parser(false);
 const [first,second]=await Promise.all([pooledImagePlugin(a.value,pool).loadTexture(0),pooledImagePlugin(b.value,pool).loadTexture(0)]);
 assert.ok(first&&second);assert.equal(first.source,second.source);assert.equal(decodes,1);assert.equal(a.reads()+b.reads(),2);
 releaseParserImageSources(a.value);releaseParserImageSources(b.value);assert.equal(closed,1);assert.equal(pool.stats().entries,0);
});

void test('failed and aborted decodes leave no strong pool entries or leaked late bitmaps',async t=>{
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});
 const pool=createSceneImagePool();const failed=parser();
 globalThis.createImageBitmap=(async()=>{throw new Error('bad image');}) as typeof createImageBitmap;
 await assert.rejects(pooledImagePlugin(failed.value,pool).loadTexture(0)!,/bad image/);
 releaseParserImageSources(failed.value);assert.deepEqual(pool.stats(),{entries:0,references:0});
 let resolveImage!:(image:ImageBitmap)=>void,closed=0,started!:()=>void;
 const decoding=new Promise<void>(resolve=>{started=resolve;});
 globalThis.createImageBitmap=(()=>{started();return new Promise<ImageBitmap>(resolve=>{resolveImage=resolve;});}) as typeof createImageBitmap;
 const pending=parser();const texture=pooledImagePlugin(pending.value,pool).loadTexture(0)!;
 await decoding;releaseParserImageSources(pending.value);assert.equal(pool.stats().entries,0);
 resolveImage({close(){closed++;}} as ImageBitmap);
 await assert.rejects(texture,{name:'AbortError'});assert.equal(closed,1);assert.equal(pool.stats().entries,0);
});

void test('HTML image fallback and nonportable formats retain Three loader behavior',()=>{
 const pool=createSceneImagePool(),p=parser();
 (p.value as unknown as {textureLoader:unknown}).textureLoader={};
 assert.equal(pooledImagePlugin(p.value,pool).loadTexture(0),null);releaseParserImageSources(p.value);
 const q=parser();q.value.json.images[0].mimeType='image/vnd-ms.dds';
 assert.equal(pooledImagePlugin(q.value,pool).loadTexture(0),null);releaseParserImageSources(q.value);
 assert.equal(pool.stats().entries,0);
});

function embeddedModel(){
 const json={asset:{version:'2.0'},scenes:[{nodes:[0]}],nodes:[{mesh:0}],meshes:[{primitives:[{attributes:{POSITION:0},indices:1,material:0}]}],materials:[{pbrMetallicRoughness:{baseColorTexture:{index:0}}}],textures:[{source:0}],images:[{bufferView:2,mimeType:'image/png',extras:{viewerImageSha256:hash}}],buffers:[{byteLength:48}],bufferViews:[{buffer:0,byteOffset:0,byteLength:36},{buffer:0,byteOffset:36,byteLength:6},{buffer:0,byteOffset:44,byteLength:4}],accessors:[{bufferView:0,componentType:5126,type:'VEC3',count:3,min:[0,0,0],max:[1,1,0]},{bufferView:1,componentType:5123,type:'SCALAR',count:3}]};
 const text=new TextEncoder().encode(JSON.stringify(json)),padded=Math.ceil(text.length/4)*4;
 const bytes=new Uint8Array(28+padded+48),header=new DataView(bytes.buffer);
 [0x46546c67,2,bytes.length,padded,0x4e4f534a].forEach((value,i)=>header.setUint32(i*4,value,true));
 bytes.fill(32,20,20+padded);bytes.set(text,20);header.setUint32(20+padded,48,true);header.setUint32(24+padded,0x004e4942,true);
 new Float32Array(bytes.buffer,28+padded,9).set([0,0,0,1,0,0,0,1,0]);new Uint16Array(bytes.buffer,28+padded+36,3).set([0,1,2]);
 return bytes;
}
void test('scene stream integrates shared image leases and releases all entries after a failed load',async t=>{
 const {parseSceneStream}=await import('./scene-stream.ts');
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});let closed=0,decodes=0;
 globalThis.createImageBitmap=(async()=>{decodes++;return {width:4,height:4,close(){closed++;}} as ImageBitmap;}) as typeof createImageBitmap;
 const pool=createSceneImagePool(),bytes=embeddedModel();
 const stream=()=>new ReadableStream<Uint8Array>({start(controller){controller.enqueue(bytes);controller.close();}});
 const first=await parseSceneStream(stream(),bytes.length,undefined,{imagePool:pool});
 const second=await parseSceneStream(stream(),bytes.length,undefined,{imagePool:pool});
 const a=(first.scene.children[0] as THREE.Mesh).material as THREE.MeshStandardMaterial;
 const b=(second.scene.children[0] as THREE.Mesh).material as THREE.MeshStandardMaterial;
 assert.equal(decodes,1);assert.equal(a.map!.source,b.map!.source);assert.equal(a.map!.colorSpace,THREE.SRGBColorSpace);
 releaseParserImageSources(first.parser);assert.equal(closed,0);releaseParserImageSources(second.parser);assert.equal(closed,1);assert.equal(pool.stats().entries,0);
 globalThis.createImageBitmap=(async()=>{throw new Error('decode failed');}) as typeof createImageBitmap;
 await assert.rejects(parseSceneStream(stream(),bytes.length,undefined,{imagePool:pool}),/decode failed/);
 assert.deepEqual(pool.stats(),{entries:0,references:0});
});

void test('decode concurrency is bounded and unreferenced queued images never decode',async t=>{
 const old=globalThis.createImageBitmap;t.after(()=>{globalThis.createImageBitmap=old;});
 const waiting:((image:ImageBitmap)=>void)[]=[];let active=0,peak=0,decodes=0,closed=0;
 globalThis.createImageBitmap=(()=>{
  decodes++;active++;peak=Math.max(peak,active);
  return new Promise<ImageBitmap>(resolve=>{waiting.push(image=>{active--;resolve(image);});});
 }) as typeof createImageBitmap;
 const pool=createSceneImagePool({maxConcurrentDecodes:2});
 const parsers=Array.from({length:6},(_,index)=>{const p=parser();p.value.json.images[0].extras.viewerImageSha256=index.toString(16).repeat(64);return p;});
 const promises=parsers.map(p=>pooledImagePlugin(p.value,pool).loadTexture(0)!);
 const settled=Promise.allSettled(promises);
 await new Promise<void>(resolve=>setTimeout(resolve,0));
 assert.deepEqual(pool.decodeStats(),{active:2,queued:4,maxConcurrent:2});
 releaseParserImageSources(parsers[3].value);releaseParserImageSources(parsers[5].value);
 assert.equal(pool.decodeStats().queued,2);
 // Complete each admitted decode, allowing the next queued item to start.
 for(let index=0;index<4;index++){
  waiting[index]({close(){closed++;}} as ImageBitmap);
  await new Promise<void>(resolve=>setTimeout(resolve,0));
 }
 const results=await settled;assert.equal(results.filter(result=>result.status==='rejected').length,2);
 assert.equal(decodes,4);assert.equal(peak,2);
 for(const p of parsers)releaseParserImageSources(p.value);
 assert.equal(closed,4);assert.deepEqual(pool.stats(),{entries:0,references:0});assert.equal(pool.decodeStats().active,0);
});
