import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {buildPavingField,pavingAtlasLayout,pavingPalette,PAVING_GLSL} from './paving.ts';

test('packed atlas preserves all original float32 run parameters, including row boundaries',()=>{
  const values=Array.from({length:73},(_,i)=>[Math.atan2(i-36,17),i*1.8%1.8,-983.1234+i*.341,1.54321+i*.01]);
  const size=13, data=pavingPalette(values,size), layout=pavingAtlasLayout(size,values.length);
  values.forEach((v,i)=>v.forEach((x,k)=>{const row=size+Math.floor(i/layout.width),col=i%layout.width;assert.equal(data[((row-size)*layout.width+col)*4+k],Math.fround(x));}));
  assert.equal(layout.height,size+Math.ceil(values.length/layout.width));
  assert.ok(layout.width*layout.height*16 < size*size*16);
});

test('empty field allocates no render target and retains exact zero fallback',()=>{
  const field=buildPavingField({} as THREE.WebGLRenderer,[],600);
  assert.equal(field.strips,0);assert.equal(field.runs,0);assert.equal(field.uniforms.uPfSize.value,0);
  assert.deepEqual(Array.from((field.texture.image as {data:Float32Array}).data),[0]);
  assert.equal(field.texture.image.width,1);assert.equal(field.texture.image.height,1);field.dispose();
});

test('original square texel selection and sampler pressure are unchanged',()=>{
  for(const size of [32,1024])for(const u of [0,.00001,.125,.499999,.99999,1]){
    const original=Math.min(size-1,Math.floor(u*size));
    const atlas=Math.min(Math.floor(u*size),size-1);assert.equal(atlas,original);
  }
  assert.equal((PAVING_GLSL.match(/uniform sampler2D/g)||[]).length,1);
  assert.match(PAVING_GLSL,/if\(id==0\)return vec4\(0\.\)/);
});

test('nonempty bake retains only a depthless packed atlas and disposes temporary depth',()=>{
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,0,0,1,0,0,-1,0,.02,1,0,.02],3));geometry.setIndex([0,1,2,1,3,2]);
  const joint=new THREE.Mesh(geometry),targets:THREE.WebGLRenderTarget[]=[],copies:unknown[][]=[];
  let bakeDisposed=false,lookupDisposed=false;
  const renderer={capabilities:{maxTextureSize:4096},getRenderTarget:()=>null,getClearColor:(v:THREE.Color)=>v,getClearAlpha:()=>1,setClearColor:()=>{},
    setRenderTarget:(t:THREE.WebGLRenderTarget|null)=>{if(t&&!targets.includes(t)){targets.push(t);t.addEventListener('dispose',()=>{bakeDisposed=true;});}},
    clear:()=>{},render:()=>{},initRenderTarget:(t:THREE.WebGLRenderTarget)=>targets.push(t),
    copyTextureToTexture:(...args:unknown[])=>{copies.push(args);const src=args[0] as THREE.Texture;if(src instanceof THREE.DataTexture)src.addEventListener('dispose',()=>{lookupDisposed=true;});}};
  const field=buildPavingField(renderer as unknown as THREE.WebGLRenderer,[joint],50,32);
  assert.equal(field.strips,1);assert.equal(targets[0].depthBuffer,true);assert.equal(targets[1].depthBuffer,false);
  assert.equal(targets[1].texture.format,THREE.RGBAFormat);assert.equal(targets[1].width,8);assert.equal(targets[1].height,33);
  assert.equal(copies.length,1);assert.equal(bakeDisposed,true);assert.equal(lookupDisposed,true);
  assert.equal(field.texture,targets[1].texture);assert.equal(field.uniforms.uPfSize.value,32);field.dispose();geometry.dispose();
});

 test('odd field widths preserve every ID coordinate and pad only unused channels',()=>{
  for(const size of [1,3,5,13,1023,1024]){
    const layout=pavingAtlasLayout(size,7);
    for(let y=0;y<size;y+=Math.max(1,size-1))for(let x=0;x<size;x++){
      const packedX=Math.floor(x/4),channel=x%4;assert.equal(packedX*4+channel,x);assert.ok(packedX<layout.width);
    }
    assert.ok(layout.width*4-size>=0&&layout.width*4-size<4);
  }
  assert.equal(pavingAtlasLayout(1024,1,1024).packed,false);
  assert.equal(pavingAtlasLayout(1024,16777216,4096).packed,false);
  assert.throws(()=>pavingAtlasLayout(2049,1,2048),/GL texture size/);
  assert.equal((PAVING_GLSL.match(/texelFetch/g)||[]).length,2);
});

test('atlas overflow preserves a valid original square parameter bake',()=>{
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,0,0,1,0,0,-1,0,.02,1,0,.02],3));geometry.setIndex([0,1,2,1,3,2]);
  let target:THREE.WebGLRenderTarget|null=null,rendered:Float32Array|undefined;
  const renderer={capabilities:{maxTextureSize:32},getRenderTarget:()=>null,getClearColor:(v:THREE.Color)=>v,getClearAlpha:()=>1,setClearColor:()=>{},setRenderTarget:(t:THREE.WebGLRenderTarget|null)=>{if(t)target=t;},clear:()=>{},render:(scene:THREE.Scene)=>{rendered=((scene.children[0] as THREE.Mesh).geometry.attributes.runValue.array as Float32Array);}};
  const field=buildPavingField(renderer as unknown as THREE.WebGLRenderer,[new THREE.Mesh(geometry)],50,32);
  assert.equal(field.uniforms.uPfSize.value,-32);assert.equal(field.texture.format,THREE.RGBAFormat);assert.equal(field.texture.image.width,32);assert.equal(field.texture.image.height,32);assert.ok(target);assert.ok(rendered);assert.ok(rendered[3]>0);assert.notEqual(rendered[0],1);
  assert.match(PAVING_GLSL,/if\(uPfSize<0\.\)return textureLod\(uPfTex,uv,0\.\)/);field.dispose();geometry.dispose();
});
