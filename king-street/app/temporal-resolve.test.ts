import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {TemporalResolve} from './temporal-resolve.ts';

function rig(){
  const target=new THREE.WebGLRenderTarget(320,200,{depthTexture:new THREE.DepthTexture(320,200)});
  const camera=new THREE.PerspectiveCamera(58,1.6,.18,60000);
  const pass=new TemporalResolve(target,camera);
  pass.setSize(320,200);
  let uniforms:Record<string,THREE.IUniform>={};
  const renderer={setRenderTarget(){},render(quad:THREE.Mesh<THREE.BufferGeometry,THREE.ShaderMaterial>){uniforms=quad.material.uniforms;}} as unknown as THREE.WebGLRenderer;
  const finish=()=>{pass.restore();pass.render(renderer);return uniforms;};
  return {target,camera,pass,finish};
}

test('jitter is scoped to the beauty image and history is invalidated for cuts and resizing',()=>{
  const {target,camera,pass,finish}=rig(),original=camera.projectionMatrix.clone();
  pass.begin();
  assert.equal(pass.reflectionUniforms.uStreetHistory.value,0);
  assert.ok(!camera.projectionMatrix.equals(original));
  finish();
  assert.ok(camera.projectionMatrix.equals(original));
  assert.ok(camera.projectionMatrixInverse.equals(original.clone().invert()));
  const first=pass.source.value;
  assert.notEqual(first,target.texture);
  pass.begin();
  assert.equal(pass.reflectionUniforms.uStreetHistory.value,1);
  finish();
  assert.notEqual(pass.source.value,first);
  assert.equal(pass.reflectionUniforms.uStreetHistory.value,0);
  camera.position.x+=10;
  pass.begin();assert.equal(pass.reflectionUniforms.uStreetHistory.value,0);finish();
  pass.setSize(640,400);assert.equal(pass.source.value,target.texture);
  pass.begin();assert.equal(pass.reflectionUniforms.uStreetHistory.value,0);finish();
  pass.dispose();target.dispose();
});

test('jittered source pixels are sampled back onto the stable presentation grid',()=>{
  const {camera,pass,finish}=rig();
  camera.updateMatrixWorld();
  const stableProjection=camera.projectionMatrix.clone();
  const stableClip=new THREE.Matrix4().multiplyMatrices(stableProjection,camera.matrixWorldInverse);
  const point=new THREE.Vector4(.7,.4,-14,1);
  const uv=(clip:THREE.Vector4)=>new THREE.Vector2(clip.x/clip.w*.5+.5,clip.y/clip.w*.5+.5);
  const stable=uv(point.clone().applyMatrix4(stableClip));
  pass.begin();
  const rasterClip=new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
  const raster=uv(point.clone().applyMatrix4(rasterClip));
  const sampled=stable.clone().add(pass.source.sampleOffset);
  assert.ok(sampled.distanceTo(raster)<1e-7,`sample offset ${pass.source.sampleOffset.toArray()} did not select the matching raster sample`);
  finish();pass.dispose();
});

test('sky reprojection follows camera rotation and ignores translation',()=>{
  const {camera,pass,finish}=rig();
  pass.begin();finish();
  camera.position.set(1.3,.5,-1);
  camera.rotation.set(.08,.18,0,'YXZ');camera.updateMatrixWorld(true);
  pass.begin();
  const uniforms=finish();
  assert.equal(uniforms.validHistory.value,1);
  const sourceUv=new THREE.Vector2(.5,.5).add(pass.source.sampleOffset);
  const clip=new THREE.Vector4(sourceUv.x*2-1,sourceUv.y*2-1,1,1);
  const farWorld=clip.clone().applyMatrix4(uniforms.worldFromClip.value);
  const direction=new THREE.Vector4(
    farWorld.x-uniforms.sourceEye.value.x*farWorld.w,
    farWorld.y-uniforms.sourceEye.value.y*farWorld.w,
    farWorld.z-uniforms.sourceEye.value.z*farWorld.w,0,
  );
  const previous=direction.clone().applyMatrix4(uniforms.previousStableClip.value);
  const projected=new THREE.Vector2(previous.x/previous.w*.5+.5,previous.y/previous.w*.5+.5);
  // Compare with the prior camera's stable projection of the current forward ray.
  const ray=camera.getWorldDirection(new THREE.Vector3());
  const expectedClip=new THREE.Vector4(ray.x,ray.y,ray.z,0).applyMatrix4(uniforms.previousStableClip.value);
  const expected=new THREE.Vector2(expectedClip.x/expectedClip.w*.5+.5,expectedClip.y/expectedClip.w*.5+.5);
  assert.ok(projected.distanceTo(expected)<1e-6,`sky reprojection ${projected.toArray()} != ${expected.toArray()}`);
  pass.dispose();
});

test('a point on a moving car reprojects to its previous pixel and depth under independent camera motion',()=>{
  const {target,camera,pass,finish}=rig();
  const body=new THREE.Group(),mesh=new THREE.Mesh(new THREE.BoxGeometry(2,1.5,4),new THREE.MeshBasicMaterial());
  body.add(mesh);body.position.set(0,0,-12);pass.track(body);
  const local=new THREE.Vector4(.5,.3,1,1);
  const stableProjection=camera.projectionMatrix.clone();
  function stablePoint(){return local.clone().applyMatrix4(body.matrixWorld).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(stableProjection);}
  function jitteredPoint(){return local.clone().applyMatrix4(body.matrixWorld).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);}
  pass.begin();const before=stablePoint();finish();
  body.position.set(1,0,-10);body.rotation.y=.2;camera.position.x=.3;
  pass.begin();const now=jitteredPoint(),u=finish();
  const homogeneous=now.clone().divideScalar(now.w).applyMatrix4(u.localFromClip.value);
  const prev=homogeneous.clone().applyMatrix4(u.previousBodyFromLocal.value);
  assert.ok(Math.abs(prev.x/prev.w-before.x/before.w)<1e-6);
  assert.ok(Math.abs(prev.w*now.w-before.w)<1e-6);
  assert.ok(u.boxMin.value.x<=local.x && local.z<=u.boxMax.value.z);
  pass.track(null);assert.equal(pass.source.value,target.texture);
  pass.dispose();target.dispose();mesh.geometry.dispose();mesh.material.dispose();
});

test('stationary accumulation exits on the first camera or body movement',()=>{
  const {camera,pass,finish}=rig();
  const body=new THREE.Group();body.position.z=-8;pass.track(body);
  pass.begin();assert.equal(finish().staticFrame.value,0);
  for(let i=0;i<16;i++){
    pass.begin();assert.equal(finish().staticFrame.value,1);
  }
  // Even subpixel motion must resume depth-tested reprojection immediately.
  camera.position.x+=1e-5;
  pass.begin();assert.equal(finish().staticFrame.value,0);
  pass.begin();assert.equal(finish().staticFrame.value,1);
  body.position.x-=1e-5;
  pass.begin();assert.equal(finish().staticFrame.value,0);
  pass.begin();assert.equal(finish().staticFrame.value,1);
  pass.reset();pass.begin();assert.equal(finish().staticFrame.value,0);
  pass.dispose();
});
