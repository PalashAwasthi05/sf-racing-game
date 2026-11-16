import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {advancePresentationTime,coverageRange,createStreamTransitions,drawTier,injectStreamCoverage} from './stream-transition.ts';
import {buildDepthProxies} from './render-budget.ts';

const geometry=()=>new THREE.BufferGeometry().setAttribute('position',new THREE.BufferAttribute(new Float32Array([0,0,0,1,0,0,0,1,0]),3));
const mesh=(material=new THREE.MeshStandardMaterial())=>new THREE.Mesh(geometry(),material);

test('facade, roof and tree tiers form disjoint complete coverage in either arrival order',()=>{
 const values=[0,1/255,.1,.5,.99,1];
 for(const d of values)for(const m of values)for(const t of values)for(const independent of [false,true])for(const fine of values){
  for(const tiers of [[drawTier.detail,drawTier.facade,drawTier.baseFacade],[drawTier.detail,drawTier.baseRoof],[drawTier.tree,drawTier.detailTree,drawTier.baseTree],[drawTier.treeFine,drawTier.treeCoarse,drawTier.detailTree,drawTier.baseTree]]){
   const ranges=tiers.map(tier=>coverageRange(tier,d,m,t,independent,fine));
   assert.ok(Math.abs(ranges.reduce((sum,[a,b])=>sum+b-a,0)-1)<1e-12);
   for(const rank of [0,.003,.091,.251,.499,.701,.999999])assert.equal(ranges.filter(([a,b])=>rank>=a&&rank<b).length,1);
  }
 }
});

test('ramps reverse continuously, resolve superseded targets and expose partial tier visibility',async()=>{
 let now=0;const changed:string[]=[];const transitions=createStreamTransitions(['cell'],id=>changed.push(id),()=>now);
 assert.equal(transitions.shows('cell',drawTier.baseFacade),true);assert.equal(transitions.shows('cell',drawTier.detail),false);
 const incoming=transitions.target('cell','detail',true);now=140;transitions.advance(now);
 assert.ok(Math.abs(transitions.value('cell','detail')-.5)<.003);assert.equal(transitions.shows('cell',drawTier.detail),true);assert.equal(transitions.shows('cell',drawTier.baseRoof),true);
 const outgoing=transitions.target('cell','detail',false);await incoming;const before=transitions.value('cell','detail');transitions.advance(now);assert.equal(transitions.value('cell','detail'),before);
 now=420;transitions.advance(now);await outgoing;assert.equal(transitions.shows('cell',drawTier.detail),false);assert.ok(changed.length>=3);
 await transitions.target('cell','facade',true,true);assert.equal(transitions.shows('cell',drawTier.facade),true);assert.equal(transitions.shows('cell',drawTier.baseFacade),false);
 transitions.dispose();assert.equal(transitions.shows('cell',drawTier.facade),false);assert.equal(transitions.shows('unknown',drawTier.baseRoof),false);
});

test('independent trees change coarse visibility immediately and pending shutdown resolves once',async()=>{
 let changes=0;const transitions=createStreamTransitions(['a'],()=>changes++,()=>0);
 await transitions.target('a','detail',true,true);assert.equal(transitions.shows('a',drawTier.detailTree),true);
 transitions.independentTree('a');assert.equal(transitions.shows('a',drawTier.detailTree),false);assert.equal(transitions.shows('a',drawTier.baseTree),true);assert.equal(changes,2);
 transitions.independentTree('a',false);assert.equal(transitions.shows('a',drawTier.detailTree),true);assert.equal(transitions.shows('a',drawTier.baseTree),false);assert.equal(changes,3);
 let resolved=0;const pending=transitions.target('a','tree',true).then(()=>resolved++);transitions.dispose();transitions.dispose();await pending;assert.equal(resolved,1);
});

test('packed fine-tree ramps preserve the independent flag and replace fine/coarse in both directions',async()=>{
 let now=0;const transitions=createStreamTransitions(['a'],()=>{},()=>now);
 const material=transitions.prepareMaterial(new THREE.MeshStandardMaterial());
 const shader={uniforms:{},vertexShader:THREE.ShaderLib.standard.vertexShader,fragmentShader:THREE.ShaderLib.standard.fragmentShader};material.onBeforeCompile(shader as Parameters<THREE.Material['onBeforeCompile']>[0],{} as THREE.WebGLRenderer);
 const table=((shader.uniforms as Record<string,{value:THREE.DataTexture}>).sfStreamLevels.value.image.data) as Uint8Array;
 await transitions.target('a','tree',true,true);transitions.independentTree('a',true);
 assert.equal(table[7],128);assert.equal(transitions.shows('a',drawTier.treeCoarse),true);assert.equal(transitions.shows('a',drawTier.treeFine),false);
 const fine=transitions.target('a','treeFine',true);assert.equal(transitions.status().ramps,1);now=140;transitions.advance(now);
 assert.equal(table[7],192);assert.equal(transitions.value('a','treeFine'),64/127);assert.equal(transitions.shows('a',drawTier.treeFine),true);assert.equal(transitions.shows('a',drawTier.treeCoarse),true);
 transitions.independentTree('a',false);assert.equal(table[7],64);transitions.independentTree('a',true);assert.equal(table[7],192);
 const trees=transitions.target('a','tree',false);assert.equal(transitions.status().ramps,2);
 now=280;transitions.advance(now);await fine;assert.equal(table[7],255);assert.equal(transitions.status().ramps,1);assert.equal(transitions.shows('a',drawTier.treeCoarse),false);
 now=420;transitions.advance(now);await trees;assert.equal(transitions.status().ramps,0);assert.equal(transitions.shows('a',drawTier.treeFine),false);assert.equal(transitions.shows('a',drawTier.baseTree),true);
 await transitions.target('a','tree',true,true);const coarse=transitions.target('a','treeFine',false);now=700;transitions.advance(now);await coarse;
 assert.equal(table[7],128);assert.equal(transitions.shows('a',drawTier.treeFine),false);assert.equal(transitions.shows('a',drawTier.treeCoarse),true);assert.equal(transitions.shows('a',drawTier.detailTree),false);
 assert.equal(transitions.status().patchedMaterials,1);material.dispose();assert.equal(transitions.status().patchedMaterials,0);transitions.dispose();
});

test('stride16 owner codes fit Uint16 through4095 slots and widen at4096 without adding attributes',()=>{
 for(const count of [4095,4096]){
  const ids=Array.from({length:count},(_,i)=>String(i)),transitions=createStreamTransitions(ids,()=>{}),source=mesh();transitions.tag(source,ids.at(-1)!,drawTier.treeCoarse);
  const attribute=source.geometry.getAttribute('streamTier');assert.equal(attribute.getX(0),count*16+drawTier.treeCoarse);
  assert.equal(attribute.array.constructor,count===4095?Uint16Array:Float32Array);assert.deepEqual(Object.keys(source.geometry.attributes).sort(),['position','streamTier']);
  source.geometry.dispose();source.material.dispose();transitions.dispose();
 }
});

test('permanent untagged supports can share the transitioned shader with slot-zero full coverage',()=>{
 const transitions=createStreamTransitions(['a'],()=>{}),support=mesh(),original=support.material;
 support.material=transitions.prepareMaterial(support.material);transitions.prepareMaterial(support.material);assert.equal(support.material,original);assert.equal(transitions.status().patchedMaterials,1);assert.equal(support.geometry.getAttribute('streamTier'),undefined);
 const future=mesh(original.clone());transitions.tag(future,'a',drawTier.detail);assert.equal(support.material.customProgramCacheKey(),future.material.customProgramCacheKey());
 const result=injectStreamCoverage(THREE.ShaderLib.standard.vertexShader,THREE.ShaderLib.standard.fragmentShader);
 assert.ok(result.vertexShader.indexOf('if(role==0')<result.vertexShader.indexOf('texelFetch(sfStreamLevels'));
 assert.deepEqual(coverageRange(0,.4,.3,.8,true,.6),[0,1]);
 support.material.dispose();future.material.dispose();support.geometry.dispose();future.geometry.dispose();assert.equal(transitions.status().patchedMaterials,0);transitions.dispose();
});

test('shader injection executes coverage with varied main formatting and preserves GLSL version/precision',()=>{
 const result=injectStreamCoverage('#version 300 es\nprecision highp float;\nvoid main(){gl_Position=vec4(0.);}', '#version 300 es\nprecision highp float;\nvoid  main ( )\n{gl_FragColor=vec4(1.);}');
 for(const shader of [result.vertexShader,result.fragmentShader]){assert.ok(shader.startsWith('#version 300 es'));assert.ok(shader.indexOf('precision highp float;')<shader.indexOf('flat varying'));}
 assert.match(result.vertexShader,/void main\(\)\{\n sfSetCoverage\(\);/);assert.match(result.fragmentShader,/sfStreamCoverage\(\);/);
 assert.throws(()=>injectStreamCoverage('no main','void main(){}'),/no main/);
 // Actual stock beauty and shadow shaders all have valid insertion points.
 for(const name of ['standard','basic','depth','distanceRGBA'] as const){const shader=THREE.ShaderLib[name];const result=injectStreamCoverage(shader.vertexShader,shader.fragmentShader);assert.match(result.vertexShader,/sfSetCoverage\(\);/);assert.match(result.fragmentShader,/sfStreamCoverage\(\);/);}
});

test('shared materials preserve source hooks and untagged support; controller scopes cannot double inject',()=>{
 const material=new THREE.MeshStandardMaterial();let calls=0;const prior:THREE.Material['onBeforeCompile']=shader=>{calls++;shader.uniforms.prior={value:2};};material.onBeforeCompile=prior;material.customProgramCacheKey=()=>'upstream-v1';
 const a=createStreamTransitions(['a'],()=>{}),b=createStreamTransitions(['b'],()=>{});const first=mesh(material),second=mesh(material);
 a.tag(first,'a',drawTier.facade);b.tag(second,'b',drawTier.facade);
 assert.notEqual(first.material,second.material);assert.equal((first.material as unknown as {defaultAttributeValues:{streamTier:number[]}}).defaultAttributeValues.streamTier[0],0);
 const compile=(mat:THREE.Material)=>{const shader={uniforms:{},vertexShader:THREE.ShaderLib.standard.vertexShader,fragmentShader:THREE.ShaderLib.standard.fragmentShader};mat.onBeforeCompile(shader as Parameters<THREE.Material['onBeforeCompile']>[0],{} as THREE.WebGLRenderer);return shader;};
 const one=compile(first.material),two=compile(second.material);assert.equal(calls,2);
 assert.equal(one.vertexShader.split('attribute float streamTier;').length-1,1);assert.equal(two.vertexShader.split('attribute float streamTier;').length-1,1);
 assert.notEqual((one.uniforms as Record<string,{value:unknown}>).sfStreamLevels.value,(two.uniforms as Record<string,{value:unknown}>).sfStreamLevels.value);
 assert.equal(first.material.customProgramCacheKey(),'upstream-v1|sf-stream-coverage-3');a.dispose();assert.equal(material.onBeforeCompile,prior);b.dispose();
});

test('shared geometry receives independent owner attributes and owned clones release once',()=>{
 const a=createStreamTransitions(['a','b'],()=>{}),first=mesh(),second=new THREE.Mesh(first.geometry,first.material),original=first.geometry;
 a.tag(first,'a',drawTier.baseRoof);a.tag(second,'b',drawTier.baseRoof);
 assert.notEqual(first.geometry,second.geometry);assert.equal(first.geometry.getAttribute('streamTier').getX(0),18);assert.equal(second.geometry.getAttribute('streamTier').getX(0),34);
 let disposed=0;second.geometry.addEventListener('dispose',()=>disposed++);second.geometry.dispose();assert.equal(a.status().ownedGeometry,0);a.dispose();assert.equal(disposed,1);assert.equal(first.geometry,original);
});

test('retired streamed materials leave no controller registry retaining their textures or hooks',()=>{
 const transitions=createStreamTransitions(['a'],()=>{});
 for(let i=0;i<100;i++){
  const source=mesh();source.material.map=new THREE.Texture();transitions.tag(source,'a',drawTier.detail);assert.equal(transitions.status().patchedMaterials,1);
  source.material.dispose();assert.equal(transitions.status().patchedMaterials,0);source.geometry.dispose();
 }
 assert.equal(transitions.status().ownedMaterials,0);transitions.dispose();assert.deepEqual(transitions.status(),{patchedMaterials:0,ownedMaterials:0,ownedGeometry:0,ramps:0,shadowVariants:0,shadowMeshes:0});
});

test('hundreds of opaque casters share one depth pair and release all variant references',()=>{
 const transitions=createStreamTransitions(['a'],()=>{}),sources=Array.from({length:100},()=>{const source=mesh();source.castShadow=true;transitions.tag(source,'a',drawTier.detail);transitions.shadow(source);return source;});
 assert.equal(transitions.status().shadowVariants,1);assert.equal(transitions.status().shadowMeshes,100);
 assert.equal(new Set(sources.map(source=>source.material.customProgramCacheKey())).size,1,'owner codes must not create per-cell shader keys');
 assert.ok(sources.every(s=>s.customDepthMaterial===sources[0].customDepthMaterial));assert.ok(sources.every(s=>s.customDistanceMaterial===sources[0].customDistanceMaterial));
 assert.equal(transitions.shadowWarmup(sources).length,1);assert.equal(transitions.shadowWarmup(sources,true).length,2);
 let depthDisposed=0;sources[0].customDepthMaterial!.addEventListener('dispose',()=>depthDisposed++);
 for(const source of sources){source.geometry.dispose();source.material.dispose();}
 assert.equal(depthDisposed,1);assert.equal(transitions.status().shadowMeshes,0);assert.equal(transitions.status().shadowVariants,0);assert.equal(transitions.status().patchedMaterials,0);transitions.dispose();assert.equal(depthDisposed,1);
});

test('tree alpha cutouts survive depth hooks while opaque PBR maps never enter shadow variants',()=>{
 const transitions=createStreamTransitions(['a'],()=>{}),texture=new THREE.Texture();const source=mesh(new THREE.MeshStandardMaterial({map:texture,alphaTest:.45,side:THREE.DoubleSide}));source.castShadow=true;
 transitions.tag(source,'a',drawTier.tree);transitions.shadow(source);const depth=source.customDepthMaterial as THREE.MeshDepthMaterial;
 assert.equal(depth.map,texture);assert.equal(depth.alphaTest,.45);assert.equal(depth.side,THREE.DoubleSide);
 const shader={uniforms:{},vertexShader:THREE.ShaderLib.depth.vertexShader,fragmentShader:THREE.ShaderLib.depth.fragmentShader};depth.onBeforeCompile(shader as Parameters<THREE.Material['onBeforeCompile']>[0],{} as THREE.WebGLRenderer);
 assert.match(shader.fragmentShader,/#include <alphatest_fragment>/);assert.match(shader.fragmentShader,/sfStreamCoverage\(\);/);assert.match(shader.vertexShader,/#include <batching_vertex>/);
 const opaque=mesh(new THREE.MeshStandardMaterial({map:texture}));opaque.castShadow=true;transitions.tag(opaque,'a',drawTier.detail);transitions.shadow(opaque);
 const solid=opaque.customDepthMaterial as THREE.MeshDepthMaterial;assert.equal(solid.map,null);
 solid.map=texture;opaque.onBeforeShadow({} as THREE.WebGLRenderer,new THREE.Scene(),new THREE.Camera(),new THREE.Camera(),opaque.geometry,solid,null as unknown as THREE.Group);assert.equal(solid.map,null);
 source.geometry.dispose();opaque.geometry.dispose();transitions.dispose();texture.dispose();
});

test('merged tile shadow proxies preserve every owner/tier code before acquiring shared variants',()=>{
 const transitions=createStreamTransitions(['a','b'],()=>{}),a=mesh(),b=mesh();
 for(const source of [a,b]){source.castShadow=true;source.geometry.setAttribute('normal',new THREE.Float32BufferAttribute([0,0,1,0,0,1,0,0,1],3));source.userData.layer='buildings';}
 transitions.tag(a,'a',drawTier.detail);transitions.tag(b,'b',drawTier.detailTree);
 const proxies=buildDepthProxies([a,b],()=> 'buildings',54,()=>false,true);assert.equal(proxies.meshes.length,1);
 const proxy=proxies.meshes[0],attribute=proxy.geometry.getAttribute('streamTier');assert.deepEqual([...attribute.array],[20,20,20,37,37,37]);
 transitions.shadow(proxy);assert.equal(proxy.geometry.getAttribute('streamTier'),attribute);assert.ok(proxy.customDepthMaterial);assert.equal(proxy.geometry.getAttribute('normal'),undefined);
 // Source sharing uses the scalar attribute's own stride, not position stride.
 assert.deepEqual([...a.geometry.getAttribute('streamTier').array],[20,20,20]);assert.deepEqual([...b.geometry.getAttribute('streamTier').array],[37,37,37]);
 proxies.dispose();a.geometry.dispose();b.geometry.dispose();transitions.dispose();
});

test('native instanced and batched shadow compilation keeps the real renderer layout without duplicate leaves',()=>{
 const transitions=createStreamTransitions(['a'],()=>{}),g=geometry(),material=new THREE.MeshStandardMaterial();
 const instance=new THREE.InstancedMesh(g,material,3);instance.castShadow=true;transitions.tag(instance,'a',drawTier.tree);transitions.shadow(instance);
 const batch=new THREE.BatchedMesh(1,3,3,material);batch.addInstance(batch.addGeometry(g));batch.castShadow=true;transitions.shadow(batch,'a',drawTier.detail);
 const warm=transitions.shadowWarmup([instance,batch]);assert.equal(warm.length,2);assert.equal((warm[0] as THREE.InstancedMesh).isInstancedMesh,true);assert.equal((warm[0] as THREE.InstancedMesh).instanceMatrix,instance.instanceMatrix);
 assert.equal((warm[1] as THREE.BatchedMesh).isBatchedMesh,true);assert.equal(warm[1].geometry,batch.geometry);assert.equal(warm[1].children.length,0);
 instance.dispose();batch.dispose();assert.equal(transitions.status().shadowMeshes,0);assert.equal(transitions.status().shadowVariants,0);g.dispose();batch.geometry.dispose();transitions.dispose();
});

test('custom shadow hooks remain intact and mixed stock alpha groups fall back without allocation or throws',()=>{
 const transitions=createStreamTransitions(['a'],()=>{}),source=mesh();source.castShadow=true;
 const custom=new THREE.MeshDepthMaterial();let previous=0;custom.onBeforeCompile=()=>{previous++;};source.customDepthMaterial=custom;
 transitions.tag(source,'a',drawTier.detail);transitions.shadow(source);assert.equal(source.customDepthMaterial,custom);
 const shader={uniforms:{},vertexShader:THREE.ShaderLib.depth.vertexShader,fragmentShader:THREE.ShaderLib.depth.fragmentShader};custom.onBeforeCompile(shader as Parameters<THREE.Material['onBeforeCompile']>[0],{} as THREE.WebGLRenderer);assert.equal(previous,1);assert.match(shader.fragmentShader,/sfStreamCoverage\(\);/);
 const mixed=new THREE.Mesh(geometry(),[new THREE.MeshStandardMaterial(),new THREE.MeshStandardMaterial({alphaTest:.5,map:new THREE.Texture()})]);mixed.castShadow=true;
 transitions.tag(mixed,'a',drawTier.detail);const count=transitions.status().shadowVariants;assert.doesNotThrow(()=>transitions.shadow(mixed));assert.equal(mixed.customDepthMaterial,undefined);assert.equal(transitions.status().shadowVariants,count);
 source.geometry.dispose();mixed.geometry.dispose();transitions.dispose();custom.dispose();
});

test('a late frame during a facade handoff preserves both surfaces and a bounded visual step',async()=>{
 let presented=0;
 const transitions=createStreamTransitions(['a'],()=>{},()=>presented);
 const fade=transitions.target('a','detail',true);
 for(const elapsed of [17,17,17,17,17,17,17,133,5000,17,17,17,17,17,17,17,17]){
   const before=transitions.value('a','detail');
   presented=advancePresentationTime(presented,elapsed);
   transitions.advance(presented);
   const after=transitions.value('a','detail');
   assert.ok(after>=before && after-before<=.133);
   const surface=coverageRange(drawTier.detail,after,0,0,false);
   assert.equal(surface[1],coverageRange(drawTier.baseFacade,after,0,0,false)[0]);
 }
 await fade;
 assert.equal(transitions.value('a','detail'),1);
 transitions.dispose();
});
