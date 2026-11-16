import * as THREE from 'three';

export type SceneryLevel = 'detail'|'facade'|'tree'|'treeFine';
export const drawTier = {baseFacade:1,baseRoof:2,baseTree:3,detail:4,detailTree:5,facade:6,tree:7,treeFine:8,treeCoarse:9} as const;
type Tier = typeof drawTier[keyof typeof drawTier];
const channel = {detail:0,facade:1,tree:2,treeFine:3} as const;
const durationMs = 280;
const width = 256;
type CompileHook=THREE.Material['onBeforeCompile'];
const materialOwners=new WeakMap<THREE.Material,{owner:object;before:CompileHook;cache:()=>string}>();
const geometryOwners=new WeakMap<THREE.BufferGeometry,{owner:object;code:number}>();

// A tab suspension or missed display frame should never skip most of a fade.
export function advancePresentationTime(previous:number,elapsedMs:number){
 return previous+(Number.isFinite(elapsedMs)?Math.min(24,Math.max(0,elapsedMs)):0);
}

type Ramp={slot:number; component:number; from:number; to:number; start:number; finish:()=>void};

/** Each screen sample belongs to exactly one available representation; independent
 * trees and facades can arrive before or after the full tile, in either order. */
export function coverageRange(tier:Tier|0,d:number,m:number,t:number,independentTree:boolean,f=0):[number,number]{
 const facadeEnd=d+(1-d)*m,treeEnd=t+(1-t)*(independentTree?0:d);
 switch(tier){
 case drawTier.baseFacade:return [facadeEnd,1];
 case drawTier.baseRoof:return [d,1];
 case drawTier.baseTree:return [treeEnd,1];
 case drawTier.detail:return [0,d];
 case drawTier.detailTree:return [t,treeEnd];
 case drawTier.facade:return [d,facadeEnd];
 case drawTier.tree:return [0,t];
 case drawTier.treeFine:return [0,t*f];
 case drawTier.treeCoarse:return [t*f,t];
 default:return [0,1];
 }
}

const vertex=/* glsl */`
attribute float streamTier;
uniform highp sampler2D sfStreamLevels;
flat varying vec2 sfCoverage;
void sfSetCoverage(){
 int code=int(streamTier+0.5),slot=code/16,role=code-slot*16;
 sfCoverage=vec2(0.,1.);
 if(role==0 || role>9 || slot==0)return;
 vec4 s=texelFetch(sfStreamLevels,ivec2(slot%256,slot/256),0);
 int flags=int(s.a*255.+0.5);
 float independent=flags>=128?1.:0.,fine=float(flags%128)/127.;
 float facadeEnd=s.r+(1.-s.r)*s.g,treeEnd=s.b+(1.-s.b)*(1.-independent)*s.r;
 if(role==1)sfCoverage.x=facadeEnd;
 else if(role==2)sfCoverage.x=s.r;
 else if(role==3)sfCoverage.x=treeEnd;
 else if(role==4)sfCoverage.y=s.r;
 else if(role==5)sfCoverage=vec2(s.b,treeEnd);
 else if(role==6)sfCoverage=vec2(s.r,facadeEnd);
 else if(role==7)sfCoverage.y=s.b;
 else if(role==8)sfCoverage.y=s.b*fine;
 else if(role==9)sfCoverage=vec2(s.b*fine,s.b);
}`;
const fragment=/* glsl */`
flat varying vec2 sfCoverage;
void sfStreamCoverage(){
 if(sfCoverage.x<=0. && sfCoverage.y>=1.)return;
 float rank=fract(52.9829189*fract(dot(floor(gl_FragCoord.xy),vec2(.06711056,.00583715))));
 if(rank<sfCoverage.x || rank>=sfCoverage.y)discard;
}`;

/** Insert global declarations immediately before main, after any #version and
 * precision declarations. Source hooks may format main differently or remove
 * clipping chunks; neither changes the transition's coverage contract. */
export function injectStreamCoverage(vertexShader:string,fragmentShader:string){
 const main=/void\s+main\s*\(\s*\)\s*\{/;
 if(!main.test(vertexShader)||!main.test(fragmentShader))throw Error('Transition shader has no main function');
 return {vertexShader:vertexShader.replace(main,match=>vertex+'\n'+match+'\n sfSetCoverage();'),
  fragmentShader:fragmentShader.replace(main,match=>fragment+'\n'+match+'\n sfStreamCoverage();')};
}

export function createStreamTransitions(ids:string[],onChange:(id:string)=>void,clock=()=>performance.now()){
 if(new Set(ids).size!==ids.length)throw Error('Transition owners must be unique');
 const slots=new Map(ids.map((id,i)=>[id,i+1]));
 const pixels=new Uint8Array(width*Math.ceil((ids.length+1)/width)*4);
 const texture=new THREE.DataTexture(pixels,width,pixels.length/(4*width),THREE.RGBAFormat);
 texture.magFilter=texture.minFilter=THREE.NearestFilter;texture.needsUpdate=true;
 const ramps=new Map<number,Ramp>();
 const patched=new WeakSet<THREE.Material>();
 const owner={},ownedMaterials=new Set<THREE.Material>();
 const ownedGeometry=new Map<THREE.BufferGeometry,{mesh:THREE.Mesh;original:THREE.BufferGeometry}>();
 const restorations=new Map<THREE.Material,()=>void>();
 type ShadowPair={key:string;depth:THREE.MeshDepthMaterial;distance:THREE.MeshDistanceMaterial;users:number};
 type ShadowBinding={mesh:THREE.Mesh;pair?:ShadowPair;geometry:THREE.BufferGeometry;depth?:THREE.Material;distance?:THREE.Material;before:THREE.Object3D['onBeforeShadow'];release:()=>void};
 const shadowPairs=new Map<string,ShadowPair>(),shadowBindings=new Map<THREE.Mesh,ShadowBinding>();
 let disposed=false;
 const rawValue=(offset:number,component:number)=>component===3?pixels[offset]&127:pixels[offset];
 const writeValue=(offset:number,component:number,v:number)=>{pixels[offset]=component===3?(pixels[offset]&128)|v:v;};
 const value=(id:string,level:SceneryLevel)=>rawValue((slots.get(id)??0)*4+channel[level],channel[level])/(level==='treeFine'?127:255);
 function advance(now:number){
   for(const [offset,r] of ramps){
     const u=Math.min(1,Math.max(0,(now-r.start)/durationMs)),smooth=u*u*(3-2*u);
     const v=Math.round(r.from+(r.to-r.from)*smooth);
     if(v!==rawValue(offset,r.component)){writeValue(offset,r.component,v);texture.needsUpdate=true;onChange(ids[r.slot-1]);}
     if(u===1){ramps.delete(offset);r.finish();}
   }
 }
 function target(id:string,level:SceneryLevel,present:boolean,snap=false){
   if(disposed)return Promise.resolve();
   const slot=slots.get(id);if(slot===undefined)return Promise.resolve();
   const component=channel[level],offset=slot*4+component,to=present?(level==='treeFine'?127:255):0;
   const previous=ramps.get(offset);if(previous){ramps.delete(offset);previous.finish();}
   if(snap||rawValue(offset,component)===to){writeValue(offset,component,to);texture.needsUpdate=true;onChange(id);return Promise.resolve();}
   onChange(id);
   return new Promise<void>(finish=>ramps.set(offset,{slot,component,from:rawValue(offset,component),to,start:clock(),finish}));
 }
 function patch(material:THREE.Material):THREE.Material{
   const priorOwner=materialOwners.get(material);
   if(priorOwner&&priorOwner.owner!==owner){
     const copy=material.clone();copy.onBeforeCompile=priorOwner.before;copy.customProgramCacheKey=priorOwner.cache;
     ownedMaterials.add(copy);copy.addEventListener('dispose',()=>ownedMaterials.delete(copy));material=copy;
   }
   if(patched.has(material))return material;
   patched.add(material);
   const originalBefore=material.onBeforeCompile,originalCache=material.customProgramCacheKey;
   const stableDefault=originalCache===THREE.Material.prototype.customProgramCacheKey?originalBefore.toString():undefined;
   const cache=()=>stableDefault??originalCache.call(material);
   const compile:CompileHook=(shader,r)=>{
     originalBefore.call(material,shader,r);
     shader.uniforms.sfStreamLevels={value:texture};
     Object.assign(shader,injectStreamCoverage(shader.vertexShader,shader.fragmentShader));
   };
   material.onBeforeCompile=compile;
   material.customProgramCacheKey=()=>cache()+'|sf-stream-coverage-3';material.needsUpdate=true;
   materialOwners.set(material,{owner,before:originalBefore,cache:originalCache});
   // A shared material can still draw untagged support. Its missing attribute
   // must select slot zero's full-coverage role, independent of prior bindings.
   const typed=material as THREE.Material&{defaultAttributeValues?:Record<string,number[]>};
   const oldDefaults=typed.defaultAttributeValues;typed.defaultAttributeValues={...oldDefaults,streamTier:[0]};
   const restore=()=>{if(material.onBeforeCompile===compile){material.onBeforeCompile=originalBefore;material.customProgramCacheKey=originalCache;typed.defaultAttributeValues=oldDefaults;material.needsUpdate=true;}if(materialOwners.get(material)?.owner===owner)materialOwners.delete(material);};
   const retired=()=>{restorations.delete(material);patched.delete(material);material.removeEventListener('dispose',retired);restore();};
   material.addEventListener('dispose',retired);
   restorations.set(material,()=>{material.removeEventListener('dispose',retired);restore();});
   return material;
 }
 function tag(mesh:THREE.Mesh,id:string,tier:Tier){
   const slot=slots.get(id);if(slot===undefined||disposed)return;
   const code=slot*16+tier;let g=mesh.geometry;
   const old=geometryOwners.get(g);
   if(old&&(old.owner!==owner||old.code!==code)){
     if(ownedGeometry.get(g)?.mesh===mesh)g.deleteAttribute('streamTier');
     else {const original=g;g=g.clone();mesh.geometry=g;g.deleteAttribute('streamTier');
      ownedGeometry.set(g,{mesh,original});g.addEventListener('dispose',()=>ownedGeometry.delete(g));}
   }
   if(!g.getAttribute('streamTier')){
     const ArrayType=ids.length<4096?Uint16Array:Float32Array;
     g.setAttribute('streamTier',new THREE.BufferAttribute(new ArrayType(g.getAttribute('position').count).fill(code),1));
     geometryOwners.set(g,{owner,code});
   }
   mesh.material=Array.isArray(mesh.material)?mesh.material.map(patch):patch(mesh.material);
   if(mesh.customDepthMaterial)mesh.customDepthMaterial=patch(mesh.customDepthMaterial);
   if(mesh.customDistanceMaterial)mesh.customDistanceMaterial=patch(mesh.customDistanceMaterial);
 }
 const shadowLayout=(mesh:THREE.Mesh)=>[(mesh as THREE.InstancedMesh).isInstancedMesh?1:0,(mesh as THREE.BatchedMesh).isBatchedMesh?1:0,
   (mesh as THREE.SkinnedMesh).isSkinnedMesh?1:0,mesh.geometry.morphAttributes.position?.length??0,mesh.geometry.getAttribute('streamTier')?.array.constructor.name??'none'].join(':');
 function shadowSignature(material:THREE.Material){
   const m=material as THREE.MeshStandardMaterial,alpha=m.alphaTest>0||m.alphaToCoverage;
   return [alpha?(m.map?.uuid??''):'',alpha?(m.alphaMap?.uuid??''):'',alpha?(m.alphaToCoverage?.5:m.alphaTest):0,m.side,m.shadowSide??'-',
    m.displacementMap?.uuid??'',m.displacementMap?m.displacementScale:0,m.displacementMap?m.displacementBias:0,
    m.clipShadows?JSON.stringify(m.clippingPlanes?.map(p=>[...p.normal.toArray(),p.constant])):'',m.clipIntersection,!!m.wireframe].join('|');
 }
 function shadow(mesh:THREE.Mesh,id?:string,tier?:Tier){
   if(disposed)return;
   if(id!==undefined&&tier!==undefined)tag(mesh,id,tier);
   if(!mesh.castShadow||!mesh.geometry.getAttribute('streamTier'))return;
   const current=shadowBindings.get(mesh);if(current&&current.geometry===mesh.geometry)return;
   current?.release();
   const sources=Array.isArray(mesh.material)?mesh.material:[mesh.material],source=sources[0] as THREE.MeshStandardMaterial;
   const originalDepth=mesh.customDepthMaterial,originalDistance=mesh.customDistanceMaterial,before=mesh.onBeforeShadow;
   if(originalDepth)mesh.customDepthMaterial=patch(originalDepth);
   if(originalDistance)mesh.customDistanceMaterial=patch(originalDistance);
   let pair:ShadowPair|undefined;
   if(!originalDepth||!originalDistance){
     const signature=shadowSignature(source);
     // One native custom material serves every geometry group. Stock mixed
     // alpha groups keep Three's correct per-group depth selection instead.
     if(sources.some(m=>shadowSignature(m)!==signature))return;
     const key=shadowLayout(mesh)+'|'+signature;
     pair=shadowPairs.get(key);
     if(!pair){
       const alpha=source.alphaTest>0||source.alphaToCoverage;
       const settings={map:alpha?source.map:null,alphaMap:alpha?source.alphaMap:null,alphaTest:alpha?(source.alphaToCoverage?.5:source.alphaTest):0,
        side:source.shadowSide??(source.side===THREE.FrontSide?THREE.BackSide:source.side===THREE.BackSide?THREE.FrontSide:THREE.DoubleSide),
        displacementMap:source.displacementMap??null,displacementScale:source.displacementScale??1,displacementBias:source.displacementBias??0,
        clippingPlanes:source.clippingPlanes,clipIntersection:source.clipIntersection,clipShadows:source.clipShadows};
       const depth=new THREE.MeshDepthMaterial({...settings,depthPacking:THREE.RGBADepthPacking}),distance=new THREE.MeshDistanceMaterial(settings);
       depth.name='stream-transition-depth';distance.name='stream-transition-distance';depth.toneMapped=distance.toneMapped=false;
       patch(depth);patch(distance);pair={key,depth,distance,users:0};shadowPairs.set(key,pair);
     }
     pair.users++;
   }
   if(originalDepth)mesh.customDepthMaterial=patch(originalDepth);else mesh.customDepthMaterial=pair!.depth;
   if(originalDistance)mesh.customDistanceMaterial=patch(originalDistance);else mesh.customDistanceMaterial=pair!.distance;
   const geometry=mesh.geometry;
   const callback:THREE.Object3D['onBeforeShadow']=function(this:THREE.Object3D,...args){
     before.apply(this,args);
     const depth=args[5] as THREE.MeshDepthMaterial|THREE.MeshDistanceMaterial;
     // WebGLShadowMap copies each beauty map onto the chosen depth material.
     // Opaque depth must never acquire unrelated facade PBR maps/UV variants.
     if(depth===pair?.depth||depth===pair?.distance){
       const group=args[6] as unknown as {materialIndex?:number}|null,material=(Array.isArray(mesh.material)?mesh.material[group?.materialIndex??0]:mesh.material) as THREE.MeshStandardMaterial;
       if(!(material.alphaTest>0||material.alphaToCoverage)){depth.map=null;depth.alphaMap=null;depth.alphaTest=0;}
     }
   };
   mesh.onBeforeShadow=callback;
   const release=()=>{
     if(shadowBindings.get(mesh)!==binding)return;
     shadowBindings.delete(mesh);geometry.removeEventListener('dispose',release);
     (mesh as THREE.InstancedMesh).removeEventListener('dispose',release);
     if(mesh.onBeforeShadow===callback)mesh.onBeforeShadow=before;
     if(mesh.customDepthMaterial===pair?.depth)mesh.customDepthMaterial=originalDepth;
     if(mesh.customDistanceMaterial===pair?.distance)mesh.customDistanceMaterial=originalDistance;
     if(pair&&--pair.users===0){shadowPairs.delete(pair.key);pair.depth.dispose();pair.distance.dispose();}
   };
   const binding:ShadowBinding={mesh,pair,geometry,depth:originalDepth,distance:originalDistance,before,release};
   shadowBindings.set(mesh,binding);geometry.addEventListener('dispose',release);
   // Native instanced/batched objects may own instance buffers while sharing
   // a permanent leaf geometry; their own disposal releases the binding too.
   (mesh as THREE.InstancedMesh).addEventListener('dispose',release);
 }
 function shadowWarmup(meshes:readonly THREE.Mesh[],includeDistance=false){
   const seen=new Set<string>(),result:THREE.Mesh[]=[];
   for(const mesh of meshes){
     if(!mesh.castShadow)continue;
     for(const material of includeDistance?[mesh.customDepthMaterial,mesh.customDistanceMaterial]:[mesh.customDepthMaterial]){
       if(!material)continue;const key=material.uuid+'|'+shadowLayout(mesh);if(seen.has(key))continue;seen.add(key);
       // Inherit the real native instancing/batching/skinning fields. These
       // temporary compile shells never own or dispose source buffers.
       const shell=Object.create(mesh) as THREE.Mesh;shell.material=material;shell.children=[];shell.visible=true;result.push(shell);
     }
   }
   return result;
 }
 return {value,target,advance,tag,shadow,shadowWarmup,
   prepareMaterial<T extends THREE.Material>(material:T):T{return (disposed?material:patch(material)) as T;},
   shows(id:string,tier:Tier){const slot=slots.get(id);if(slot===undefined||disposed)return false;const p=slot*4;
     const [lo,hi]=coverageRange(tier,pixels[p]/255,pixels[p+1]/255,pixels[p+2]/255,(pixels[p+3]&128)!==0,(pixels[p+3]&127)/127);return hi>lo;},
   independentTree(id:string,present=true){const slot=slots.get(id),to=present?128:0;if(slot&&!disposed&&(pixels[slot*4+3]&128)!==to){pixels[slot*4+3]=(pixels[slot*4+3]&127)|to;texture.needsUpdate=true;onChange(id);}},
   status:()=>({patchedMaterials:restorations.size,ownedMaterials:ownedMaterials.size,ownedGeometry:ownedGeometry.size,ramps:ramps.size,shadowVariants:shadowPairs.size,shadowMeshes:shadowBindings.size}),
   dispose(){if(disposed)return;disposed=true;for(const r of ramps.values())r.finish();ramps.clear();for(const binding of [...shadowBindings.values()])binding.release();for(const restore of restorations.values())restore();restorations.clear();
     for(const [g,{mesh,original}] of ownedGeometry){if(mesh.geometry===g)mesh.geometry=original;g.dispose();}ownedGeometry.clear();
     for(const material of ownedMaterials)material.dispose();ownedMaterials.clear();texture.dispose();}};
}
