import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {renderPostQuad,OCTAHEDRAL_NORMAL_GLSL} from './postprocessing-utils.ts';
import {reuseDepthForOcclusion} from './render-budget.ts';
import {createGrade} from './grade.ts';
import type {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';

// Numerical error budget over both octahedral hemispheres, including each
// axis and the folding seam. This tests quantization, not visual thresholds.
function quantizedRoundTrip(v:number[]){
 let [x,y,z]=v;const l=Math.abs(x)+Math.abs(y)+Math.abs(z);x/=l;y/=l;z/=l;
 if(z<0){const ox=x;x=(1-Math.abs(y))*(ox>=0?1:-1);y=(1-Math.abs(ox))*(y>=0?1:-1);}
 x=Math.round((x*.5+.5)*255)/255*2-1;y=Math.round((y*.5+.5)*255)/255*2-1;
 z=1-Math.abs(x)-Math.abs(y);const t=Math.max(0,Math.min(1,-z));x+=x>=0?-t:t;y+=y>=0?-t:t;
 const length=Math.hypot(x,y,z);return [x/length,y/length,z/length];
}
test('RG8 normal encoding has bounded angular error across the sphere and folding seam',()=>{
 let worst=0;
 const samples=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
 for(let i=0;i<50000;i++){const y=1-2*(i+.5)/50000,a=i*2.399963229728653,r=Math.sqrt(1-y*y);samples.push([r*Math.cos(a),y,r*Math.sin(a)]);}
 for(const n of samples){const d=quantizedRoundTrip(n);worst=Math.max(worst,Math.acos(Math.max(-1,Math.min(1,n.reduce((s,v,i)=>s+v*d[i],0))))*180/Math.PI);}
 assert.ok(worst<.95,`normal quantization error ${worst} degrees`);
 assert.match(OCTAHEDRAL_NORMAL_GLSL,/step\(vec2\(0\.\),e\)\*2\.-1\./);
});

test('fullscreen post draws never clear and restore autoClear even when drawing throws',()=>{
 for(const initial of [true,false]){
  const renderer={autoClear:initial,setRenderTarget(){assert.equal(renderer.autoClear,false);}} as unknown as THREE.WebGLRenderer;
  const quad={render(){assert.equal(renderer.autoClear,false);}};
  renderPostQuad(renderer,null,quad as never);assert.equal(renderer.autoClear,initial);
  assert.throws(()=>renderPostQuad(renderer,null,{render(){throw Error('draw failed');}} as never));
  assert.equal(renderer.autoClear,initial);
 }
});

test('GTAO and denoiser decode compact normals while depth, resolution and debug paths stay intact',()=>{
 const camera=new THREE.PerspectiveCamera(58,16/9,.18,60000),ao=new GTAOPass(new THREE.Scene(),camera,960,540);
 const target=(ao as GTAOPass & {normalRenderTarget:THREE.WebGLRenderTarget}).normalRenderTarget,originalDepth=target.depthTexture;
 const depth=new THREE.DepthTexture(1920,1080);
 const projection=camera.projectionMatrix.clone();projection.elements[8]+=.001;
 let calls=0;ao.render=()=>{calls++;assert.ok(camera.projectionMatrix.equals(projection));};
 const dispose=reuseDepthForOcclusion(ao,depth,projection.clone().invert(),projection);
 assert.equal(target.texture.format,THREE.RGFormat);assert.equal(target.texture.type,THREE.UnsignedByteType);
 assert.equal(target.texture.internalFormat,'RG8');assert.equal(target.depthTexture,originalDepth);
 assert.equal(target.width,960);assert.equal(target.height,540);
 for(const material of [ao.gtaoMaterial,ao.pdMaterial]){
  assert.equal(material.defines.NORMAL_VECTOR_TYPE,3);
  assert.match(material.fragmentShader,/return unpackOctNormal\(textureLod\(tNormal,uv,0\.\)\.rg\)/);
  assert.equal(material.uniforms.tDepth.value,originalDepth);
 }
 let reconstruction='';const renderer={autoClear:true,setRenderTarget(){},render(mesh:THREE.Mesh<THREE.BufferGeometry,THREE.ShaderMaterial>){reconstruction=mesh.material.fragmentShader;assert.equal(renderer.autoClear,false);}} as unknown as THREE.WebGLRenderer;
 const before=camera.projectionMatrix.clone();
 ao.output=GTAOPass.OUTPUT.Normal;ao.render(renderer,new THREE.WebGLRenderTarget(),new THREE.WebGLRenderTarget(),1/60,false);
 assert.equal(ao.copyMaterial.uniforms.octNormalOutput.value,1);
 assert.ok(camera.projectionMatrix.equals(before));assert.equal(calls,1);assert.equal(renderer.autoClear,true);
 assert.match(reconstruction,/gl_FragDepth=depth/);assert.match(reconstruction,/vec4\(packOctNormal\(n\),0\.,1\.\)/);
 ao.output=GTAOPass.OUTPUT.AO;ao.render(renderer,new THREE.WebGLRenderTarget(),new THREE.WebGLRenderTarget(),1/60,false);
 assert.equal(ao.copyMaterial.uniforms.octNormalOutput.value,0);
 dispose();ao.dispose();depth.dispose();
});

test('grade retains all exposure/bloom/sharpen passes but omits redundant target clears',()=>{
 const passes:unknown[]=[];const source=new THREE.Texture();
 const grade=createGrade({addPass(p:unknown){passes.push(p);}} as unknown as EffectComposer,source);
 let draws=0;const renderer={autoClear:true,setRenderTarget(){},render(){assert.equal(renderer.autoClear,false);draws++;},toneMappingExposure:1} as unknown as THREE.WebGLRenderer;
 const read=new THREE.WebGLRenderTarget(1920,1080),write=read.clone();
 for(const pass of passes as {setSize(w:number,h:number):void;render(...args:unknown[]):void}[]){pass.setSize(1920,1080);pass.render(renderer,write,read,1/60,false);assert.equal(renderer.autoClear,true);}
 assert.equal(passes.length,4);assert.equal(draws,13); // meter+adapt, composite, nine bloom draws, finish
 assert.equal(grade.composite.output.width,1920);assert.equal(grade.composite.output.height,1080);
 grade.dispose();source.dispose();read.dispose();write.dispose();
});
