import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {Texture} from 'three';
import {driveResidualBinding,driveResidualUrl,loadDriveResiduals} from './drive-residuals.ts';
import type {DriveStream} from './drive-stream.ts';

test('matches the Python binding regardless of field order and hands visible surfaces back on eviction',async()=>{
  const piece=(id:string)=>({id,contentHash:id.padStart(12,'0'),startM:0,endM:50,boundsEN:[0,0,50,20],
    compressedBytes:1,chunks:[],format:'gzip-chunks'});
  const a=piece('a'),b=piece('b');
  const index={scope:'drive-stream',version:1,base:a,sourceWorldHash:'test',world:'/compare/test.json',routeEN:[[0,0],[50,0]],
    cells:[a,b]} as DriveStream;
  index.distantExteriors=await driveResidualUrl(index);
  const binding=driveResidualBinding(index);
  const binary=new Uint8Array(140),view=new DataView(binary.buffer);
  for(const at of [0,64]){
    for(const [i,v] of [0,0,0,1,0,0,0,1,0].entries())view.setFloat32(at+i*4,v,true);
    binary.fill(200,at+45,at+54);
    new Uint16Array(binary.buffer,at+56,3).set([0,1,2]);
  }
  const binaryHash=createHash('sha256').update(binary).digest('hex');
  const metadata={...binding,pieces:binding.pieces.map(p=>({contentHash:p.contentHash,id:p.id})),binaryHash,
    binary:'/compare/drive-residual-'+binaryHash.slice(0,16)+'.bin',byteLength:binary.length,
    entries:[...[a,b].map((p,i)=>({id:p.id,layer:'facades',appearance:'matte',vertices:3,indices:3,indexBytes:2,offset:i*64})),
      {id:'a',layer:'vegetation',appearance:'foliage',vertices:1,offset:128,runs:[{count:1,color:[.2,.3,.1]}]}],stats:{}};
  const originalFetch=globalThis.fetch;
  globalThis.fetch=async url=>{
    if(url===index.distantExteriors)return Response.json(metadata);
    if(url===metadata.binary)return new Response(binary);
    throw Error('Unexpected request');
  };
  try{
    const leafTexture=new Texture();
    const distant=await loadDriveResiduals(index,undefined,leafTexture);
    const shown=()=>distant.root.children.filter(m=>m.visible&&!m.userData.persistentFoliage).map(m=>m.userData.streamCell);
    assert.deepEqual(shown(),['a','b']);
    const canopy=distant.root.children.find(m=>m.userData.persistentFoliage);
    assert.ok(canopy?.visible);assert.ok(distant.sharedFoliagePieces.has('a'));
    distant.setResident(['a']);assert.deepEqual(shown(),['b']);assert.equal(canopy.visible,true);
    distant.setLayer('vegetation',false);assert.equal(canopy.visible,false);
    distant.setLayer('vegetation',true);assert.equal(canopy.visible,true);
    distant.setResident(['b']);assert.deepEqual(shown(),['a']);
    distant.setLayer('facades',false);assert.deepEqual(shown(),[]);
    distant.setLayer('facades',true);distant.setResident([]);assert.deepEqual(shown(),['a','b']);
    distant.dispose();assert.equal(distant.root.children.length,0);
    metadata.pieces[0].contentHash='wrong';
    await assert.rejects(loadDriveResiduals(index),/binding mismatch/);
  }finally{globalThis.fetch=originalFetch;}
});
