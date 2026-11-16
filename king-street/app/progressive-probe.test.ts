import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {ProgressiveProbe} from './progressive-probe.ts';

test('moving reflections publish only the complete filtered atlas and restore the previous render target',()=>{
  const levels:number[]=[];
  const spare=new THREE.WebGLRenderTarget(768,1024);
  const cube=new THREE.CubeTexture(),axis=new THREE.Vector3(0,1,0);
  const filter={
    _blur(_out:THREE.WebGLRenderTarget,_from:number,to:number,_sigma:number,_axis:THREE.Vector3){levels.push(to);},
    fromCubemap(_cube:THREE.CubeTexture,out:THREE.WebGLRenderTarget){
      for(let i=1;i<=10;i++)this._blur(out,i-1,i,i*.1,axis);
      return out;
    },
  };
  const renderer={autoClear:true,xr:{enabled:true},getRenderTarget:()=>null,getActiveCubeFace:()=>4,getActiveMipmapLevel:()=>1,
    setRenderTarget:(out:unknown,face:number,mip:number)=>{assert.equal(out,null);assert.equal(face,4);assert.equal(mip,1);}};
  const job=new ProgressiveProbe(filter as unknown as THREE.PMREMGenerator);
  const step=()=>job.advance(renderer as unknown as THREE.WebGLRenderer);
  job.start(cube,spare);assert.equal(job.pending,true);assert.deepEqual(levels,[]);
  assert.equal(step(),null);assert.deepEqual(levels,[1]);assert.equal(step(),null);assert.deepEqual(levels,[1,2]);
  assert.equal(step(),spare);assert.deepEqual(levels,[1,2,3,4,5,6,7,8,9,10]);
  assert.equal(spare.scissorTest,false);assert.equal(renderer.xr.enabled,true);assert.equal(renderer.autoClear,true);
  job.start(cube,spare);job.cancel();assert.equal(step(),null);assert.equal(job.pending,false);
  cube.dispose();spare.dispose();
});
