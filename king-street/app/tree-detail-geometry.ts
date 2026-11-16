import * as THREE from 'three';
import {barkMaterials,buildFoliage,finishFoliage,taperTreeLimbs} from './foliage.ts';
import {buildLeafTwigs,sourceWoodySegments} from './tree-architecture.ts';
import {coarseLeafSources,scaleLeafClusters} from './tree-foliage-lod.ts';

type Entry={kind:'leaves'|'bark'|'frond';offset:number;vertices:number;indices?:number};
export type TreeCellAsset={id:string;contentHash:string;boundsEN:number[];binary:string;binaryHash:string;compressedBytes:number;byteLength:number;entries:Entry[];classes:string[];
 skeletonIds?:number[]};
export type TreeSkeletonAsset={binary:string;binaryHash:string;compressedBytes:number;byteLength:number;segments:number};
export type TreeCell={id:string;root:THREE.Group;renderables:THREE.Mesh[];foliageMeshes:THREE.Mesh[];bytes:number;geometryBytes:number;triangles:number;twigSegments:number;transitionBytes:number;quality:'fine'|'coarse';dispose():void};
const align=(n:number)=>Math.ceil(n/4)*4;
const hash=async(value:Uint8Array)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value as Uint8Array<ArrayBuffer>)),b=>b.toString(16).padStart(2,'0')).join('');

function mergeTreeMeshes(meshes:THREE.Mesh[],name:string){
 if(meshes.length<2){if(meshes[0])meshes[0].name=name;return meshes[0];}
 const vertexCount=meshes.reduce((n,m)=>n+m.geometry.getAttribute('position').count,0),indexCount=meshes.reduce((n,m)=>n+m.geometry.index!.count,0);
 const geometry=new THREE.BufferGeometry();
 for(const attributeName of Object.keys(meshes[0].geometry.attributes)){
  const first=meshes[0].geometry.getAttribute(attributeName),size=first.itemSize;
  const ArrayType=attributeName==='position'?Float32Array:attributeName==='uv'?Uint8Array:attributeName==='normal'||attributeName==='cardNormal'?Int16Array:Uint16Array;
  const values=new ArrayType(vertexCount*size),target=new THREE.BufferAttribute(values,size,first.normalized);let offset=0;
  for(const mesh of meshes){const source=mesh.geometry.getAttribute(attributeName) as THREE.BufferAttribute;
   for(let i=0;i<source.count;i++)for(let c=0;c<size;c++)target.setComponent(offset+i,c,source.getComponent(i,c));
   offset+=source.count;
  }
  geometry.setAttribute(attributeName,target);
 }
 const indices=vertexCount>65535?new Uint32Array(indexCount):new Uint16Array(indexCount);let offset=0,vertexOffset=0;
 for(const mesh of meshes){const source=mesh.geometry.index!;for(let i=0;i<source.count;i++)indices[offset++]=source.getX(i)+vertexOffset;vertexOffset+=mesh.geometry.getAttribute('position').count;}
 geometry.setIndex(new THREE.BufferAttribute(indices,1));geometry.computeBoundingBox();geometry.computeBoundingSphere();
 const result=new THREE.Mesh(geometry,meshes[0].material);result.name=name;result.userData={...meshes[0].userData};
 result.castShadow=result.receiveShadow=true;result.matrixAutoUpdate=false;result.updateMatrix();
 for(const mesh of meshes){mesh.removeFromParent();mesh.geometry.dispose();}
 return result;
}

export function createTreeMaterials(texture:THREE.Texture,setupMaterial:(material:THREE.Material)=>void){
 const bark=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.92,metalness:0});bark.name='tree-detail-bark';
 const leaf=new THREE.MeshStandardMaterial({map:texture,vertexColors:true,alphaTest:.45,side:THREE.DoubleSide,roughness:.85,metalness:0});leaf.name='foliage';leaf.alphaToCoverage=true;
 const frond=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.85,metalness:0,side:THREE.DoubleSide});frond.name='tree-detail-frond';
 setupMaterial(bark);barkMaterials([bark]);setupMaterial(leaf);finishFoliage(leaf);setupMaterial(frond);
 return {bark,leaf,frond,dispose(){bark.dispose();leaf.dispose();frond.dispose();}};
}
type TreeMaterials=ReturnType<typeof createTreeMaterials>;

/** Validate every byte range before allocating foliage or touching a renderer. */
export function validateTreeCell(asset:TreeCellAsset,buffer:ArrayBuffer){
 if(buffer.byteLength!==asset.byteLength||!asset.entries.length)throw Error('Invalid tree cell length');
 let end=0;
 for(const e of asset.entries){
  if(e.offset!==end||!Number.isSafeInteger(e.vertices)||e.vertices<1||!['leaves','bark','frond'].includes(e.kind))throw Error('Invalid tree entry');
  end=align(end+e.vertices*12);
  if(e.kind!=='leaves')end=align(end+e.vertices*3);
  end=align(end+e.vertices*6);
  if(e.kind!=='leaves'){
   if(!Number.isSafeInteger(e.indices)||!e.indices||e.indices%3)throw Error('Invalid tree branch indices');
   end=align(end+e.indices*4);
  }
  if(end>buffer.byteLength)throw Error('Invalid tree entry range');
  const positions=new Float32Array(buffer,e.offset,e.vertices*3);
  if(!positions.every(Number.isFinite))throw Error('Invalid tree positions');
  if(e.kind!=='leaves'&&new Uint32Array(buffer,end-e.indices!*4,e.indices).some(i=>i>=e.vertices))throw Error('Invalid tree branch vertex');
 }
 if(end!==buffer.byteLength)throw Error('Incomplete tree cell ranges');
}

/** Exact branch meshes and all leaf samples; only the leaf footprint is stylized.
 * Source faces preserve forks and holes. Smaller independent sprays expose that
 * structure rather than covering the crown with two opaque intersecting walls. */
export function buildTreeCell(asset:TreeCellAsset,buffer:ArrayBuffer,leafTexture:THREE.Texture,
 setupMaterial:(material:THREE.Material)=>void,options:{twigs?:boolean;clusterDensity?:boolean;skeleton?:Float32Array;merge?:boolean;quality?:'fine'|'coarse'}={}):TreeCell{
 validateTreeCell(asset,buffer);
 const root=new THREE.Group();root.name='tree-detail__'+asset.id;
 const renderables:THREE.Mesh[]=[],foliageMeshes:THREE.Mesh[]=[],materials:THREE.Material[]=[];
 const sources:THREE.Mesh[]=[];
 let sparseSources:THREE.Mesh[]=[];
 const barkMaterial=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.92,metalness:0});
 barkMaterial.name='tree-detail-bark';materials.push(barkMaterial);setupMaterial(barkMaterial);barkMaterials([barkMaterial]);
 const frondMaterial=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.85,metalness:0,side:THREE.DoubleSide});frondMaterial.name='tree-detail-frond';materials.push(frondMaterial);setupMaterial(frondMaterial);
 let bytes=0,vertices=0,twigSegments=0;
 const attach=(mesh:THREE.Mesh,foliage=false)=>{
  mesh.userData={...mesh.userData,treeDetail:true,layer:'vegetation',streamCell:asset.id};
  mesh.castShadow=mesh.receiveShadow=true;mesh.matrixAutoUpdate=false;mesh.updateMatrix();
  mesh.geometry.computeBoundingBox();mesh.geometry.computeBoundingSphere();
  renderables.push(mesh);root.add(mesh);if(foliage)foliageMeshes.push(mesh);
  for(const attribute of Object.values(mesh.geometry.attributes))bytes+=attribute.array.byteLength;
  bytes+=mesh.geometry.index?.array.byteLength??0;
  vertices+=mesh.geometry.getAttribute('position').count;
 };
 try{
  for(const e of asset.entries){
   const geometry=new THREE.BufferGeometry();let offset=e.offset;
   geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffer,offset,e.vertices*3),3));offset=align(offset+e.vertices*12);
   if(e.kind!=='leaves'){
    geometry.setAttribute('normal',new THREE.BufferAttribute(new Int8Array(buffer,offset,e.vertices*3),3,true));offset=align(offset+e.vertices*3);
   }
   geometry.setAttribute('color',new THREE.BufferAttribute(new Uint16Array(buffer,offset,e.vertices*3),3,true));offset=align(offset+e.vertices*6);
   if(e.kind==='leaves'){
    const source=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xffffff}));source.userData.foliageFanCentres=true;sources.push(source);
   }else{
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer,offset,e.indices),1));
    const mesh=new THREE.Mesh(geometry,e.kind==='frond'?frondMaterial:barkMaterial);mesh.name='vegetation__'+asset.id+'___'+e.kind;
    if(e.kind==='bark')taperTreeLimbs(mesh);attach(mesh);
   }
  }
  if(sources.length){
   const segments=options.skeleton&&asset.skeletonIds?asset.skeletonIds.map(id=>({root:new THREE.Vector3().fromArray(options.skeleton!,id*8),tip:new THREE.Vector3().fromArray(options.skeleton!,id*8+3),radius:options.skeleton![id*8+6],owner:String(options.skeleton![id*8+7])})):sourceWoodySegments(renderables.filter(m=>m.name.endsWith('___bark')));
   if(options.clusterDensity!==false)scaleLeafClusters(sources,segments);
   if(options.quality==='coarse')sparseSources=coarseLeafSources(sources,segments);
   const leafSources=sparseSources.length?sparseSources:sources;
   if(options.twigs!==false){
    const twigs=buildLeafTwigs(leafSources,segments,options.quality==='coarse'?768:2048);
    if(twigs){const mesh=new THREE.Mesh(twigs,barkMaterial);mesh.name='vegetation__'+asset.id+'___twigs';twigSegments=twigs.userData.twigSegments;attach(mesh);}
   }
   const canopy=buildFoliage(leafSources,leafTexture,32,options.quality==='coarse'?[.92,1.14]:[.78,1.12]);
   materials.push(canopy.material);setupMaterial(canopy.material);finishFoliage(canopy.material);
   for(const mesh of canopy.meshes)attach(mesh,true);
  }
 }catch(error){for(const mesh of renderables)mesh.geometry.dispose();for(const material of materials)material.dispose();throw error;}
 finally{for(const source of [...sources,...sparseSources]){source.geometry.dispose();(source.material as THREE.Material).dispose();}}
 if(options.merge!==false){
  const groups=[renderables.filter(m=>foliageMeshes.includes(m)),renderables.filter(m=>m.name.endsWith('___bark')||m.name.endsWith('___twigs')),renderables.filter(m=>m.name.endsWith('___frond'))];
  const merged=groups.map((group,i)=>mergeTreeMeshes(group,'vegetation__'+asset.id+'___'+['foliage','bark','frond'][i])).filter((m):m is THREE.Mesh=>!!m);
  root.clear();renderables.length=0;foliageMeshes.length=0;renderables.push(...merged);bytes=0;vertices=0;
  for(let i=0;i<merged.length;i++){
   const mesh=merged[i];root.add(mesh);if(mesh.name.endsWith('___foliage'))foliageMeshes.push(mesh);
   vertices+=mesh.geometry.getAttribute('position').count;for(const a of Object.values(mesh.geometry.attributes))bytes+=a.array.byteLength;bytes+=mesh.geometry.index?.array.byteLength??0;
  }
 }
 let disposed=false;
 // Count both CPU geometry and its GPU copy, plus the retained decoded source
 // buffer (bark views still share it). The streaming limit covers all three.
 // streamTier contains source-cell and LOD roles. Reserve its Uint16 CPU and
 // GPU copies before the caller's staging/transition hooks add the attribute.
 const transitionBytes=vertices*4;
 return {id:asset.id,root,renderables,foliageMeshes,bytes:bytes*2+buffer.byteLength+transitionBytes,geometryBytes:bytes,twigSegments,transitionBytes,quality:options.quality??'fine',
  triangles:renderables.reduce((n,m)=>n+(m.geometry.index?.count??0)/3,0),
  dispose(){if(disposed)return;disposed=true;root.removeFromParent();for(const mesh of renderables)mesh.geometry.dispose();for(const material of materials)material.dispose();root.clear();}};
}

export async function loadTreeCell(asset:TreeCellAsset,leafTexture:THREE.Texture,setupMaterial:(material:THREE.Material)=>void,signal?:AbortSignal,skeleton?:Float32Array,quality:'fine'|'coarse'='fine'){
 const response=await fetch(asset.binary,{signal});if(!response.ok)throw Error('Tree geometry unavailable');
 const encoded=new Uint8Array(await response.arrayBuffer());
 if(encoded.byteLength!==asset.compressedBytes||await hash(encoded)!==asset.binaryHash)throw Error('Tree geometry checksum mismatch');
 const buffer=await new Response(new Blob([encoded.buffer as ArrayBuffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 signal?.throwIfAborted();return buildTreeCell(asset,buffer,leafTexture,setupMaterial,{skeleton,quality});
}

export async function loadTreeSkeleton(asset:TreeSkeletonAsset){
 if(!/^\/compare\/tree-skeleton-[a-f0-9]{16}\.bin$/.test(asset.binary)||!/^[a-f0-9]{64}$/.test(asset.binaryHash)||
  !Number.isSafeInteger(asset.segments)||asset.segments<0||asset.segments>100000||asset.byteLength!==asset.segments*32||
  !Number.isSafeInteger(asset.compressedBytes)||asset.compressedBytes<=0||asset.compressedBytes>4*1024**2)throw Error('Invalid tree skeleton descriptor');
 const response=await fetch(asset.binary);if(!response.ok)throw Error('Tree skeleton unavailable');
 const encoded=new Uint8Array(await response.arrayBuffer());
 if(encoded.byteLength!==asset.compressedBytes||await hash(encoded)!==asset.binaryHash)throw Error('Tree skeleton checksum mismatch');
 const buffer=await new Response(new Blob([encoded.buffer as ArrayBuffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 if(buffer.byteLength!==asset.byteLength)throw Error('Invalid tree skeleton length');
 const table=new Float32Array(buffer);if(!table.every(Number.isFinite))throw Error('Invalid tree skeleton values');
 return table;
}

export function packTreeCell(cell:TreeCell){
 return {id:cell.id,bytes:cell.bytes,geometryBytes:cell.geometryBytes,triangles:cell.triangles,twigSegments:cell.twigSegments,transitionBytes:cell.transitionBytes,quality:cell.quality,meshes:cell.renderables.map(mesh=>({
  name:mesh.name,foliage:cell.foliageMeshes.includes(mesh),frond:mesh.name.endsWith('___frond'),
  attributes:Object.entries(mesh.geometry.attributes).map(([name,attribute])=>({name,itemSize:attribute.itemSize,normalized:attribute.normalized,array:attribute.array})),
  index:mesh.geometry.index?.array,box:[...mesh.geometry.boundingBox!.min.toArray(),...mesh.geometry.boundingBox!.max.toArray()],
  sphere:[...mesh.geometry.boundingSphere!.center.toArray(),mesh.geometry.boundingSphere!.radius],
 }))};
}
type PackedCell=ReturnType<typeof packTreeCell>;
export function hydrateTreeCell(packed:PackedCell,leafTexture:THREE.Texture,setupMaterial:(material:THREE.Material)=>void,sharedMaterials?:TreeMaterials):TreeCell{
 const root=new THREE.Group();root.name='tree-detail__'+packed.id;
 const materials=sharedMaterials??createTreeMaterials(leafTexture,setupMaterial),{bark,leaf,frond}=materials;
 const foliageMeshes:THREE.Mesh[]=[],renderables=packed.meshes.map(source=>{
  const geometry=new THREE.BufferGeometry();for(const a of source.attributes)geometry.setAttribute(a.name,new THREE.BufferAttribute(a.array,a.itemSize,a.normalized));
  if(source.index)geometry.setIndex(new THREE.BufferAttribute(source.index,1));
  geometry.boundingBox=new THREE.Box3(new THREE.Vector3().fromArray(source.box),new THREE.Vector3().fromArray(source.box,3));
  geometry.boundingSphere=new THREE.Sphere(new THREE.Vector3().fromArray(source.sphere),source.sphere[3]);
  const mesh=new THREE.Mesh(geometry,source.foliage?leaf:source.frond?frond:bark);mesh.name=source.name;
  mesh.userData={treeDetail:true,layer:'vegetation',streamCell:packed.id,original:mesh.material};
  mesh.castShadow=mesh.receiveShadow=true;mesh.matrixAutoUpdate=false;mesh.updateMatrix();root.add(mesh);
  if(source.foliage)foliageMeshes.push(mesh);return mesh;
 });
 let disposed=false;
 return {id:packed.id,root,renderables,foliageMeshes,bytes:packed.bytes,geometryBytes:packed.geometryBytes,triangles:packed.triangles,twigSegments:packed.twigSegments,transitionBytes:packed.transitionBytes,quality:packed.quality,
  dispose(){if(disposed)return;disposed=true;root.removeFromParent();for(const mesh of renderables)mesh.geometry.dispose();if(!sharedMaterials)materials.dispose();root.clear();}};
}


export type {PackedCell};
