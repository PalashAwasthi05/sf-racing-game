import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {buildRoadField,roadFieldLayout} from './roadfield.ts';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);
test('rectangular crop retains world pixel centres, density, exact half seeds and 30m margin',()=>{
 const bounds=[-48,100,48,420],l=roadFieldLayout(bounds,512,2048,1);
 for(let i=0;i<2048;i++){const seed=(i+.5)/2048-.5;assert.equal(THREE.DataUtils.fromHalfFloat(THREE.DataUtils.toHalfFloat(seed)),seed);}
 assert.equal(l.cropped,true);assert.equal(l.width,512);assert.equal(l.height,1024);close(l.texelM,.5);
 assert.ok(l.minE<=bounds[0]-30&&l.minN<=bounds[1]-30&&l.minE+l.spanE>=bounds[2]+30&&l.minN+l.spanN>=bounds[3]+30);
 for(const [min,span,count] of [[l.minE,l.spanE,l.width],[l.minN,l.spanN,l.height]]){
  const offset=(min+512)/l.texelM;assert.ok(Number.isInteger(offset));
  for(let i=0;i<count;i++){close(min+(i+.5)*span/count,-512+(offset+i+.5)*.5);const seed=(i+.5)/count-.5;assert.equal(THREE.DataUtils.fromHalfFloat(THREE.DataUtils.toHalfFloat(seed)),seed);}
 }
});
test('clips original square; full, empty, invalid and irregular cases retain exact legacy layout',()=>{
 const edge=roadFieldLayout([470,-10,600,10],512,2048);assert.equal(edge.minE+edge.spanE,512);assert.ok(edge.minE>=-512);
 for(const b of [null,[-900,-900,900,900],[-500,-500,500,500],[NaN,0,1,1],[2,2,1,1]])assert.deepEqual(roadFieldLayout(b,512,2048),{width:2048,height:2048,minE:-512,minN:-512,spanE:1024,spanN:1024,texelM:.5,cropped:false});
 assert.equal(roadFieldLayout([-1,-1,1,1],512,1500).cropped,false);
 const full=roadFieldLayout(null,512,2048);for(const e of [-512,-123,.25,511.75])close((e-full.minE)/full.spanE,e/1024+.5);
});
function rig(full=false,size=2048,withBuilding=false){
 const scene=new THREE.Scene(),road=new THREE.Mesh(new THREE.BoxGeometry(full?1024:96,1,full?1024:320));if(!full)road.position.set(80,0,-260);
 const parent=new THREE.Group();parent.position.x=12;parent.add(road);scene.add(parent);
 const building=withBuilding?new THREE.Mesh(road.geometry,road.material):null;if(building){building.position.set(-250,0,-260);scene.add(building);}
 let current:THREE.WebGLRenderTarget|null=null,rasterCamera:THREE.Camera|null=null;
 const targets=new Set<THREE.WebGLRenderTarget>(),passes:Array<{material:THREE.ShaderMaterial,target:THREE.WebGLRenderTarget|null}>=[];
 const renderer={shadowMap:{autoUpdate:true},autoClear:true,getRenderTarget:()=>null,getClearColor:(out:THREE.Color)=>out.set(0),getClearAlpha:()=>1,setClearColor(){},clear(){},
 setRenderTarget(target:THREE.WebGLRenderTarget|null){current=target;if(target)targets.add(target);},
 render(object:THREE.Scene,camera:THREE.Camera){if(object===scene)rasterCamera=camera;else passes.push({material:(object.children[0] as THREE.Mesh).material as THREE.ShaderMaterial,target:current});}} as unknown as THREE.WebGLRenderer;
 const field=buildRoadField(renderer,scene,[road],512,size,building?[building]:[]);
 return {field,targets,passes,camera:()=>rasterCamera!,dispose(){field.dispose();road.geometry.dispose();(road.material as THREE.Material).dispose();}};
}
test('GPU camera and flood/pack uniforms preserve rectangular world-space distances',()=>{
 const r=rig(),l=r.field.extent;assert.equal(l.cropped,true);assert.notEqual(l.width,l.height);assert.ok(l.minE<=44-30&&l.minE+l.spanE>=140+30);
 const ndc=new THREE.Vector3(l.minE+.5*l.texelM,0,-(l.minN+.5*l.texelM)).project(r.camera());close((ndc.x+1)/2,.5/l.width);close((ndc.y+1)/2,.5/l.height);
 const pack=r.passes.find(p=>p.material.uniforms.uSize)!.material.uniforms;assert.deepEqual(pack.uSize.value.toArray(),[l.spanE,l.spanN]);close(pack.uPx.value.x*pack.uSize.value.x,.5);close(pack.uPx.value.y*pack.uSize.value.y,.5);
 const flood=r.passes.find(p=>p.material.uniforms.uMetric)!.material.uniforms,delta=new THREE.Vector2(7/l.spanE,11/l.spanN);
 close(delta.clone().multiply(pack.uSize.value).length(),Math.hypot(7,11));close(delta.multiply(flood.uMetric.value).length()*Math.max(l.spanE,l.spanN),Math.hypot(7,11));
 assert.ok([...r.targets].every(t=>t.width===l.width&&t.height===l.height));assert.equal(r.field.target.texture.type,THREE.HalfFloatType);assert.deepEqual(r.field.uniforms.uRfExtent.value.toArray(),[l.minE,l.minN,l.spanE,l.spanN]);r.dispose();
});
test('legacy full-square GPU path preserves dimensions, camera and float seed fallback',()=>{
 const r=rig(true,1500);assert.equal(r.field.extent.cropped,false);assert.equal(r.field.target.width,1500);assert.equal(r.field.target.height,1500);
 const flood=r.passes.find(p=>p.material.uniforms.uMetric)!;assert.deepEqual(flood.material.uniforms.uMetric.value.toArray(),[1,1]);assert.equal(flood.target!.texture.type,THREE.FloatType);assert.deepEqual(r.camera().position.toArray(),[0,2000,0]);
 close(new THREE.Vector3(123,0,-234).project(r.camera()).x,123/512);close(new THREE.Vector3(123,0,-234).project(r.camera()).y,234/512);r.dispose();
});

test('building geometry participates in the crop union independently of roads',()=>{
 const r=rig(false,2048,true);assert.ok(r.field.extent.minE<=-298-30);assert.ok(r.field.extent.minE+r.field.extent.spanE>=140+30);assert.equal(r.field.extent.texelM,.5);r.dispose();
});

test('half-precision road bake packs into the free seed buffer, keeping the source separate',()=>{
 for(const size of [512,1024,2048]){
   const r=rig(true,size);
   const pack=r.passes.at(-1)!;
   assert.equal(pack.target,r.field.target);
   assert.equal(r.targets.size,3); // road mask and two seed buffers
   assert.notEqual(pack.material.uniforms.tSeeds.value,r.field.texture);
   assert.equal(r.field.texture.magFilter,THREE.LinearFilter);
   assert.equal(pack.material.uniforms.tSeeds.value.magFilter,THREE.NearestFilter);
   r.dispose();
 }
 const fallback=rig(true,1500);assert.equal(fallback.targets.size,4);fallback.dispose();
});
