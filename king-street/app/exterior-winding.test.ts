import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {orientWallGeometry} from './exterior-winding.ts';

function wall(){
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute([0,0,0,1,0,0,0,1,0,1,1,0],3));
  g.setIndex([0,2,1,1,3,2]);
  return g;
}
const faces={mesh:0,primitive:0,indexCount:6,vertexCount:4,triangles:[0]};
test('fixes only recorded faces before normal generation and leaves shared indices alone',()=>{
  const g=wall(),original=g.index!;
  assert.ok(orientWallGeometry(g,faces));assert.deepEqual([...g.index!.array],[0,1,2,1,3,2]);assert.deepEqual([...original.array],[0,2,1,1,3,2]);
  g.computeVertexNormals();assert.equal(g.attributes.normal.getZ(0),1);
});
test('splits wrong authored normals while preserving shared colors and UVs',()=>{
  const g=wall();
  g.setAttribute('normal',new THREE.Int8BufferAttribute([0,0,-127,0,0,-127,0,0,127,0,0,127],3,true));
  g.setAttribute('uv',new THREE.Float32BufferAttribute([0,0,1,0,0,1,1,1],2));
  g.setAttribute('color',new THREE.Uint8BufferAttribute([1,2,3,4,5,6,7,8,9,10,11,12],3,true));
  assert.ok(orientWallGeometry(g,faces));assert.deepEqual([...g.index!.array],[4,5,2,1,3,2]);
  assert.equal(g.attributes.normal.getZ(1),-1);assert.equal(g.attributes.normal.getZ(5),1);
  assert.deepEqual([...g.attributes.uv.array.slice(-4)],[0,0,1,0]);
  assert.deepEqual([...g.attributes.color.array.slice(-6)],[1,2,3,4,5,6]);
});
test('a mismatched record cannot partially alter archived topology',()=>{
  const g=wall(),index=g.index;
  assert.equal(orientWallGeometry(g,{...faces,triangles:[0,55]}),false);
  assert.equal(orientWallGeometry(g,{...faces,indexCount:18}),false);assert.equal(g.index,index);
});
