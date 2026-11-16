import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import type {GLTF} from 'three/addons/loaders/GLTFLoader.js';
import {prepareSceneDrawables,createDrawableTextureUpload,type SceneDrawableShared} from './scene-drawables.ts';

function fixture(){
 const group=new THREE.Group(),geometry=new THREE.BoxGeometry(1,1,1);
 const image={width:1,height:1,close(){closed++;}};let closed=0;
 const map=new THREE.Texture(image as unknown as ImageBitmap);
 const leaf=new THREE.Texture(),environment=new THREE.Texture();
 const material=new THREE.MeshStandardMaterial({map,envMap:environment});material.name='concrete';
 const first=new THREE.Mesh(geometry,material);first.name='buildings__123___concrete';first.position.set(120,4,-90);
 const second=new THREE.Mesh(geometry,material);second.name='buildings__124___concrete';second.position.set(122,4,-90);
 group.add(first,second);
 const parser={associations:new Map<unknown,unknown>([[material,{}],[map,{}]]),cache:{removeAll(){}}};
 const model={scene:group,parser} as unknown as GLTF;
 const shared:SceneDrawableShared={data:{grid:{min:-1000,step:1000,n:2},ground:[0,0,0,0]},renderer:{capabilities:{getMaxAnisotropy:()=>8}},atmosphere:{setupMaterial(){}},leafTexture:leaf,groundAt:()=>0,wxGround:new THREE.DataTexture(),roadField:{uRfTex:{value:null},uRfR:{value:1}},paving:{uPfTex:{value:null},uPfR:{value:1}},roadDamp:{value:0},roadReflection:{value:null},surfaceNight:{value:0}};
 return {model,shared,map,leaf,environment,material,geometry,first,second,closed:()=>closed};
}

void test('cell prep keeps world transforms and owns only its local textures and unique resources',async()=>{
 const f=fixture();const phases:string[]=[];let yields=0;
 const cell=await prepareSceneDrawables(f.model,f.shared,{stage:name=>phases.push(name),yieldCpu:async()=>{yields++;}});
 assert.equal(cell.root.parent,null);assert.ok(cell.root.children.includes(f.model.scene));
 assert.deepEqual(phases,['sources','materials','weathering','batches','ready']);assert.ok(yields>=5);
 assert.equal(cell.batches.length,1);assert.equal(cell.batches[0].parent,cell.batchRoot);
 cell.root.updateMatrixWorld(true);
 assert.deepEqual(f.first.getWorldPosition(new THREE.Vector3()).toArray(),[120,4,-90]);
 assert.deepEqual([...cell.textures],[f.map]);
 let geometryDisposals=0,materialDisposals=0,leafDisposals=0,environmentDisposals=0;
 f.geometry.addEventListener('dispose',()=>geometryDisposals++);f.material.addEventListener('dispose',()=>materialDisposals++);
 f.leaf.addEventListener('dispose',()=>leafDisposals++);f.environment.addEventListener('dispose',()=>environmentDisposals++);
 const scene=new THREE.Scene();scene.add(cell.root);
 cell.dispose();cell.dispose();
 assert.equal(cell.root.parent,null);assert.equal(cell.root.children.length,0);
 assert.equal(geometryDisposals,1);assert.equal(materialDisposals,1);assert.equal(f.closed(),1);
 assert.equal(leafDisposals,0);assert.equal(environmentDisposals,0);assert.equal(f.map.image,null);
 assert.equal(f.model.parser.associations.size,0);
});

void test('cell texture upload steps are bounded and stop immediately after eviction',async()=>{
 const f=fixture();const cell=await prepareSceneDrawables(f.model,f.shared);
 const uploaded:THREE.Texture[]=[];const upload=createDrawableTextureUpload(cell,{initTexture:texture=>uploaded.push(texture)});
 assert.equal(upload.pending,1);assert.equal(upload.uploadNext(0),0);assert.equal(upload.uploadNext(),1);
 assert.deepEqual(uploaded,[f.map]);assert.equal(upload.pending,0);
 const next=createDrawableTextureUpload(cell,{initTexture(){throw new Error('disposed cell uploaded');}});
 cell.dispose();assert.equal(next.pending,0);assert.equal(next.uploadNext(),0);
});

void test('aborted preparation releases parsed resources and never attaches the cell',async()=>{
 const f=fixture(),controller=new AbortController();let disposed=0;
 f.material.addEventListener('dispose',()=>disposed++);
 await assert.rejects(prepareSceneDrawables(f.model,f.shared,{signal:controller.signal,yieldCpu:async()=>{controller.abort();}}),{name:'AbortError'});
 assert.equal(disposed,1);assert.equal(f.closed(),1);assert.equal(f.model.scene.parent,null);
 assert.equal(f.model.parser.associations.size,0);
});


void test('eviction never takes ownership of viewer overlays or reflection maps installed after preparation',async()=>{
 const f=fixture();const cell=await prepareSceneDrawables(f.model,f.shared);
 const evidence=new THREE.MeshStandardMaterial(),reflection=new THREE.Texture();let evidenceDisposals=0,reflectionDisposals=0;
 evidence.addEventListener('dispose',()=>evidenceDisposals++);reflection.addEventListener('dispose',()=>reflectionDisposals++);
 f.material.envMap=reflection;
 for(const mesh of cell.renderables)mesh.material=evidence;
 cell.dispose();
 assert.equal(evidenceDisposals,0);assert.equal(reflectionDisposals,0);assert.equal(f.closed(),1);
});


test('live preparation yields between costly source meshes before the first phase boundary',async()=>{
 const f=fixture();let reachedSources=false,beforeSources=0;
 f.shared.atmosphere.setupMaterial=()=>{const start=performance.now();while(performance.now()-start<5){}};
 const cell=await prepareSceneDrawables(f.model,f.shared,{
  stage:name=>{if(name==='sources')reachedSources=true;},
  yieldCpu:async()=>{if(!reachedSources)beforeSources++;},
 });
 assert.ok(beforeSources>=1);cell.dispose();
});
