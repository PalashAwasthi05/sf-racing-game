import assert from 'node:assert/strict';
import test from 'node:test';
import {parseSceneStream, releaseSceneLoadingData, finishDrawableParser} from './scene-stream.ts';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

function fixture(overlap = false, flat = false) {
  // The index view precedes the position view in the file but follows it in
  // the directory. Both have leading padding and the position accessor starts
  // four bytes inside its view. Force even the GLB headers across stream reads.
  const json = {
    asset: {version: '2.0'}, scenes: [{nodes: [0]}], nodes: [{mesh: 0}],
    meshes: [{primitives: [{attributes: {POSITION: 0}, indices: 1}]}],
    buffers: [{byteLength: 52}], bufferViews: [
      {buffer: 0, byteOffset: 12, byteLength: 40}, {buffer: 0, byteOffset: 4, byteLength: overlap ? 10 : 6},
    ],
    accessors: [
      {bufferView: 0, byteOffset: 4, componentType: 5126, type: 'VEC3', count: 3, min: [0,0,0], max: [1,1,0]},
      {bufferView: 1, componentType: 5123, type: 'SCALAR', count: 3},
    ],
  };
  if (flat) {
    Object.assign(json, {materials: [{name: 'paint', pbrMetallicRoughness: {baseColorFactor: [.25,.5,.75,1], roughnessFactor: .75}}]});
    Object.assign(json.nodes[0], {name:'facades:front.wall', matrix:[1,0,0,0, 0,1,0,0, 0,0,1,0, 5,6,7,1], extras:{layer:'facades',owner:'node'}});
    Object.assign(json.meshes[0], {name:'another.mesh', extras:{featureId:'source-id',owner:'mesh'}});
    Object.assign(json.meshes[0].primitives[0], {material:0,mode:4});
  }
  const text = new TextEncoder().encode(JSON.stringify(json)), padded = Math.ceil(text.length / 4) * 4;
  const bytes = new Uint8Array(28 + padded + 52), header = new DataView(bytes.buffer);
  [0x46546c67, 2, bytes.length, padded, 0x4e4f534a].forEach((n, i) => header.setUint32(i*4, n, true));
  bytes.fill(32, 20, 20 + padded); bytes.set(text, 20);
  header.setUint32(20 + padded, 52, true); header.setUint32(24 + padded, 0x004e4942, true);
  new Uint16Array(bytes.buffer, 28 + padded + 4, 3).set([2,1,0]);
  new Float32Array(bytes.buffer, 28 + padded + 16, 9).set([0,0,0, 1,0,0, 0,1,0]);
  return bytes;
}
function stream(bytes: Uint8Array) {
  let offset = 0;
  return new ReadableStream<Uint8Array>({pull(controller) {
    if (offset >= bytes.length) return controller.close();
    controller.enqueue(bytes.subarray(offset, offset + 7)); offset += 7;
  }});
}
for (const overlap of [false, true]) {
  test(overlap ? 'overlapping views use the standard GLB path' : 'streamed views preserve BIN and accessor offsets', async () => {
    const bytes = fixture(overlap), model = await parseSceneStream(stream(bytes), bytes.length);
    const mesh = model.scene.children[0] as import('three').Mesh;
    assert.deepEqual(Array.from(mesh.geometry.attributes.position.array), [0,0,0, 1,0,0, 0,1,0]);
    assert.deepEqual(Array.from(mesh.geometry.index!.array), [2,1,0]);
    const loading = model.parser as unknown as {cache: {get(key: string): unknown}; extensions?: Record<string, {body: unknown}>};
    assert.equal(loading.cache.get('bufferView:0'), undefined);
    if (overlap) assert.equal(loading.extensions!.KHR_binary_glTF.body, null);
  });
}
test('a truncated BIN rejects before the model is usable', async () => {
  const bytes = fixture();
  await assert.rejects(parseSceneStream(stream(bytes.subarray(0,-1)), bytes.length), /Incomplete model/);
});

test('a flat static scene matches Three for names, material, metadata, geometry and world bounds', async () => {
  const bytes = fixture(false, true);
  const standard = await new GLTFLoader().parseAsync(bytes.buffer, '/');
  const fast = await parseSceneStream(stream(bytes), bytes.length);
  const read = (model: typeof fast) => {
    const m = model.scene.children[0] as import('three').Mesh<import('three').BufferGeometry, import('three').MeshStandardMaterial>;
    return [m.name, m.userData, m.matrixWorld.toArray(), m.geometry.boundingBox, m.geometry.boundingSphere,
      m.geometry.attributes.position.array, m.geometry.index?.array, m.material.name, m.material.color.toArray(),
      m.material.roughness, m.material.flatShading, m.material.normalScale.toArray(), model.parser.associations.get(m)];
  };
  assert.deepEqual(read(fast), read(standard));
});

test('retiring a resident cell parser releases its source document and plugins while its rendered mesh survives',async()=>{
 const model=await parseSceneStream(stream(fixture(false,true)));
 const mesh=model.scene.children[0] as import('three').Mesh;
 const position=mesh.geometry.attributes.position;
 assert.ok(model.parser.json.nodes.length);
 finishDrawableParser(model.parser);
 assert.equal(mesh.geometry.attributes.position,position);
 assert.equal(position.count,3);
 assert.deepEqual(Object.keys(model.parser.json),['asset']);
 assert.equal(model.parser.associations.size,0);
 assert.deepEqual((model.parser as unknown as {plugins:object}).plugins,{});
});


test('loader release clears separate image caches without closing live texture images', () => {
  let cleared = 0;
  const image = {close() {throw new Error('live image closed');}};
  const parser = {
    cache: {removeAll() {cleared++;}},
    sourceCache: {image: Promise.resolve(image)}, textureCache: {sampler: Promise.resolve(image)},
    associations: new Map([[image, {textures: 0}]]),
    extensions: {KHR_binary_glTF: {body: new ArrayBuffer(8)}},
  };
  releaseSceneLoadingData(parser as unknown as Parameters<typeof releaseSceneLoadingData>[0]);
  assert.equal(cleared, 1);
  assert.deepEqual(parser.sourceCache, {});
  assert.deepEqual(parser.textureCache, {});
  assert.ok(parser.associations.has(image));
  assert.equal(parser.extensions.KHR_binary_glTF.body, null);
});

function treeFixture() {
 const original=fixture(false,true),oldLength=new DataView(original.buffer).getUint32(12,true);
 const doc=JSON.parse(new TextDecoder().decode(original.subarray(20,20+oldLength)));
 const body=new Uint8Array(96);body.set(original.subarray(28+oldLength));
 body.set(original.subarray(28+oldLength+12,28+oldLength+52),52);
 doc.buffers[0].byteLength=96;
 doc.bufferViews.push({buffer:0,byteOffset:52,byteLength:40},{buffer:0,byteOffset:92,byteLength:4});
 doc.accessors.push({...doc.accessors[0],bufferView:2});
 doc.materials.push({pbrMetallicRoughness:{baseColorTexture:{index:0}}});
 doc.textures=[{source:0}];doc.images=[{bufferView:3,mimeType:'image/png'}];
 doc.meshes.push({primitives:[{attributes:{POSITION:2},indices:1,material:1}]});
 doc.nodes.push({...doc.nodes[0],name:'vegetation___leaf1',mesh:1,extras:{layer:'vegetation'}});
 doc.scenes[0].nodes.push(1);
 const text=new TextEncoder().encode(JSON.stringify(doc)),padded=Math.ceil(text.length/4)*4;
 const bytes=new Uint8Array(28+padded+96),header=new DataView(bytes.buffer);
 [0x46546c67,2,bytes.length,padded,0x4e4f534a].forEach((n,i)=>header.setUint32(i*4,n,true));
 bytes.fill(32,20,20+padded);bytes.set(text,20);header.setUint32(20+padded,96,true);header.setUint32(24+padded,0x004e4942,true);bytes.set(body,28+padded);
 return bytes;
}
test('stream tree omission never requests its exclusive material, undecodable image or vertex view',async()=>{
 const bytes=treeFixture();
 // The image deliberately has no valid PNG data or browser decoder in this Node test.
 // Loading the omitted material would fail instead of returning a model.
 const model=await parseSceneStream(stream(bytes),bytes.length,undefined,{omitStreamedTrees:new Set(['leaf1'])});
 assert.equal(model.scene.children.length,1);assert.equal(model.scene.children[0].userData.layer,'facades');
 assert.equal([...model.parser.associations.values()].some(a=>a.materials===1),false);
 const mesh=model.scene.children[0] as import('three').Mesh;
 assert.deepEqual(Array.from(mesh.geometry.attributes.position.array),[0,0,0,1,0,0,0,1,0]);
});
