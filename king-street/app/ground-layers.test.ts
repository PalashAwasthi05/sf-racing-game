import assert from 'node:assert/strict';
import test from 'node:test';
import {packGroundLayers} from './ground-detail.ts';

test('ground packing preserves slope/tone and corresponding exact height without another image allocation',()=>{
 const surface=new Uint8ClampedArray([1,2,3,255,4,5,6,255]),height=new Uint8ClampedArray([71,72,73,255,91,92,93,255]);
 const packed=packGroundLayers(surface,height);
 assert.equal(packed.buffer,surface.buffer);assert.deepEqual([...packed],[1,2,3,71,4,5,6,91]);
 const fallback=packGroundLayers(new Uint8ClampedArray([1,2,3,11]),null);assert.deepEqual([...fallback],[1,2,3,255]);
});
