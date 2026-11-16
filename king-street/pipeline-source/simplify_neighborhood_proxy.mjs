// Offline, one building's roof only. Open boundaries and a 15cm absolute
// geometric error limit prevent cross-building and metre-scale LOD distortion.
import {readSync, writeSync} from 'node:fs';
import {MeshoptSimplifier} from './vendor/meshoptimizer/simplifier.mjs';
await MeshoptSimplifier.ready;
function readBytes(length, allowEof=false) {
  const data=Buffer.allocUnsafeSlow(length); let offset=0;
  while(offset<length) {
    const count=readSync(0,data,offset,length-offset);
    if(!count) {if(allowEof&&!offset)return null;throw Error('Incomplete proxy');}
    offset+=count;
  }
  return data.buffer;
}
function send(array) {
  const data=Buffer.from(array.buffer,array.byteOffset,array.byteLength);
  for(let offset=0;offset<data.length;)offset+=writeSync(1,data,offset);
}
for(;;) {
  const header=readBytes(12,true); if(!header)break;
  const [vertices,indices,channels]=new Uint32Array(header);
  const positions=new Float32Array(readBytes(vertices*12));
  readBytes(vertices*channels*4);
  const triangles=new Uint32Array(readBytes(indices*4));
  const [out]=MeshoptSimplifier.simplify(triangles,positions,3,
    Math.max(3,Math.floor(indices/60)*3),.15,['LockBorder','ErrorAbsolute']);
  send(new Uint32Array([out.length]));send(out);
}
