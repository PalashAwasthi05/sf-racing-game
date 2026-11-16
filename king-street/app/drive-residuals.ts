import * as THREE from 'three';
import {buildFoliage} from './foliage.ts';
import type {DriveStream} from './drive-stream';

type Binding = {version:number;sourceWorldHash:string;pieces:{id:string;contentHash:string}[]};
type SurfaceEntry = {id:string;layer:string;appearance:'matte'|'glazing';offset:number;vertices:number;indices:number;indexBytes:2|4};
type LeafEntry = {id:string;layer:'vegetation';appearance:'foliage';offset:number;vertices:number;runs:{count:number;color:[number,number,number]}[]};
type Entry = SurfaceEntry | LeafEntry;
type ResidualManifest = Binding & {binary:string;binaryHash:string;byteLength:number;entries:Entry[];stats:Record<string,number>};
const hex=(value:ArrayBuffer)=>Array.from(new Uint8Array(value),b=>b.toString(16).padStart(2,'0')).join('');
const hash=async(value:Uint8Array)=>hex(await crypto.subtle.digest('SHA-256',value as Uint8Array<ArrayBuffer>));
export function driveResidualBinding(index:DriveStream):Binding {
  const pieces=(index.pieces??index.cells).map(p=>{
    if(!p.contentHash)throw Error('Distant exterior piece has no content hash');
    return {id:p.id,contentHash:p.contentHash};
  });
  return {version:6,sourceWorldHash:index.sourceWorldHash,pieces};
}
export async function driveResidualUrl(index:DriveStream) {
  const digest=await hash(new TextEncoder().encode(JSON.stringify(driveResidualBinding(index))));
  return '/compare/drive-residuals-'+digest.slice(0,16)+'.json';
}

/** Display-only source exteriors. Collision meshes and their visibility remain owned by the caller. */
export async function loadDriveResiduals(index:DriveStream,signal?:AbortSignal,leafTexture?:THREE.Texture) {
  const expectedUrl=await driveResidualUrl(index);
  const advertised=index.distantExteriors;
  if(advertised && advertised!==expectedUrl)throw Error('Distant exterior index binding mismatch');
  const response=await fetch(advertised??expectedUrl,{signal});
  if(!response.ok)throw Error('Distant exterior package unavailable');
  const manifest=await response.json() as ResidualManifest;
  const binding=driveResidualBinding(index);
  if(manifest.version!==binding.version||manifest.sourceWorldHash!==binding.sourceWorldHash||
    !Array.isArray(manifest.pieces)||manifest.pieces.length!==binding.pieces.length||
    manifest.pieces.some((p,i)=>p?.id!==binding.pieces[i].id||p.contentHash!==binding.pieces[i].contentHash) ||
    !/^\/compare\/drive-residual-[a-f0-9]{16}\.bin$/.test(manifest.binary) || !/^[a-f0-9]{64}$/.test(manifest.binaryHash) ||
    !Number.isSafeInteger(manifest.byteLength)||manifest.byteLength<=0||manifest.byteLength>24*1024**2)throw Error('Distant exterior binding mismatch');
  const binaryResponse=await fetch(manifest.binary,{signal});
  if(!binaryResponse.ok)throw Error('Distant exterior geometry unavailable');
  const bytes=new Uint8Array(await binaryResponse.arrayBuffer());
  if(bytes.byteLength!==manifest.byteLength || await hash(bytes)!==manifest.binaryHash)throw Error('Distant exterior checksum mismatch');
  // Validate every range before creating any renderer geometry or typed arrays.
  if(!Array.isArray(manifest.entries)||manifest.entries.length>manifest.pieces.length*8)throw Error('Invalid distant exterior entries');
  let expectedOffset=0;
  const seen=new Set<string>();
  for(const entry of manifest.entries){
    const key=entry.id+'__'+entry.layer+'__'+entry.appearance;
    if(entry.appearance==='foliage'){
      if(!leafTexture||seen.has(key)||!manifest.pieces.some(p=>p.id===entry.id)||entry.layer!=='vegetation'||
        entry.offset!==expectedOffset||!Number.isSafeInteger(entry.vertices)||entry.vertices<=0||
        !Array.isArray(entry.runs)||!entry.runs.length||entry.runs.some(r=>!Number.isSafeInteger(r.count)||r.count<=0||
          !Array.isArray(r.color)||r.color.length!==3||!r.color.every(c=>Number.isFinite(c)&&c>=0))||
        entry.runs.reduce((n,r)=>n+r.count,0)!==entry.vertices)throw Error('Invalid foliage center range');
      expectedOffset+=entry.vertices*12;if(expectedOffset>bytes.length)throw Error('Invalid foliage center range');
      seen.add(key);continue;
    }
    if(!['matte','glazing'].includes(entry.appearance)||seen.has(key)||!manifest.pieces.some(p=>p.id===entry.id)||!['buildings','facades','roofs','vegetation'].includes(entry.layer)||
      !Number.isSafeInteger(entry.vertices)||entry.vertices<3||
      !Number.isSafeInteger(entry.indices)||entry.indices<=0||entry.indices%3!==0||
      entry.indexBytes!==(entry.vertices<=65535?2:4)||
      !Number.isSafeInteger(entry.offset)||entry.offset!==expectedOffset)throw Error('Invalid distant exterior range');
    const indexOffset=Math.ceil((entry.offset+entry.vertices*18)/4)*4;
    expectedOffset=Math.ceil((indexOffset+entry.indices*entry.indexBytes)/4)*4;
    if(expectedOffset>bytes.length)throw Error('Invalid distant exterior range');
    seen.add(key);
  }
  if(expectedOffset!==bytes.length)throw Error('Incomplete distant exterior ranges');
  const root=new THREE.Group();root.name='drive-source-residuals';
  const material=new THREE.MeshStandardMaterial({vertexColors:true,roughness:1,metalness:0,side:THREE.DoubleSide});
  // Glazing retains a low-roughness specular lobe instead of becoming a matte black panel.
  const glazingMaterial=new THREE.MeshPhysicalMaterial({vertexColors:true,roughness:.18,metalness:0,clearcoat:1,clearcoatRoughness:.15,side:THREE.DoubleSide});
  // Match the detailed world's roof and facade depth priorities so coincident window
  // panels do not flicker against the wall while their detail section is unloaded.
  const roof=material.clone(),facade=material.clone();
  for(const [surface,units] of [[roof,-12],[facade,-18],[glazingMaterial,-22]] as const){
    surface.polygonOffset=true;surface.polygonOffsetFactor=0;surface.polygonOffsetUnits=units;
  }
  const materials=[material,roof,facade,glazingMaterial];
  const geometries:THREE.BufferGeometry[]=[];
  const groups=new Map<string,THREE.Mesh[]>();
  const sharedFoliagePieces=new Set<string>();
  const layers=new Map<string,boolean>();
  let residents=new Set<string>();
  const update=()=>{for(const [id,meshes] of groups)for(const mesh of meshes)mesh.visible=(mesh.userData.persistentFoliage||!residents.has(id))&&layers.get(mesh.userData.layer)!==false;};
  try {
    for(const entry of manifest.entries){
      if(entry.appearance==='foliage'){
        let start=entry.offset;
        const sources=entry.runs.map(run=>{
          const geometry=new THREE.BufferGeometry();
          const points=new Float32Array(bytes.buffer,start,run.count*3);start+=points.byteLength;
          if(!points.every(Number.isFinite))throw Error('Invalid foliage center');
          geometry.setAttribute('position',new THREE.BufferAttribute(points,3));
          const tint=new THREE.MeshStandardMaterial();tint.color.fromArray(run.color);
          const mesh=new THREE.Mesh(geometry,tint);mesh.userData.foliageFanCentres=true;return mesh;
        });
        const canopy=buildFoliage(sources,leafTexture!);
        for(const source of sources){source.geometry.dispose();source.material.dispose();}
        sharedFoliagePieces.add(entry.id);materials.push(canopy.material);
        const group=groups.get(entry.id)??[];groups.set(entry.id,group);
        for(const mesh of canopy.meshes){
          mesh.userData.persistentFoliage=true;mesh.userData.streamCell=entry.id;
          geometries.push(mesh.geometry);root.add(mesh);group.push(mesh);
        }
        continue;
      }
      const {vertices,offset,indices,indexBytes}=entry;
      // Aligned package views can go straight to WebGL with no second copy of the mesh.
      const positions=new Float32Array(bytes.buffer,offset,vertices*3);
      const normals=new Int8Array(bytes.buffer,offset+vertices*12,vertices*3);
      const colors=new Uint8Array(bytes.buffer,offset+vertices*15,vertices*3);
      const indexOffset=Math.ceil((offset+vertices*18)/4)*4;
      const index=indexBytes===2?new Uint16Array(bytes.buffer,indexOffset,indices):new Uint32Array(bytes.buffer,indexOffset,indices);
      if(!positions.every(Number.isFinite)||index.some(i=>i>=vertices))throw Error('Invalid distant exterior vertex');
      const geometry=new THREE.BufferGeometry();geometries.push(geometry);
      geometry.setIndex(new THREE.BufferAttribute(index,1));
      geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
      geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3,true));
      geometry.setAttribute('color',new THREE.BufferAttribute(colors,3,true));
      geometry.computeBoundingSphere();
      const mesh=new THREE.Mesh(geometry,entry.appearance==='glazing'?glazingMaterial:entry.layer==='facades'?facade:entry.layer==='roofs'?roof:material);mesh.name='source-residual__'+entry.id+'__'+entry.layer;
      mesh.userData={layer:entry.layer,streamCell:entry.id,displayResidual:true};
      mesh.castShadow=false;mesh.receiveShadow=true;mesh.matrixAutoUpdate=false;mesh.updateMatrix();
      root.add(mesh);const group=groups.get(entry.id)??[];group.push(mesh);groups.set(entry.id,group);
    }
  } catch(error){geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());throw error;}
  return {root,materials,sharedFoliagePieces,stats:{...manifest.stats,bytes:manifest.byteLength,drawCalls:root.children.length},
    setResident(ids:Iterable<string>){residents=new Set(ids);update();},
    setLayer(layer:string,visible:boolean){layers.set(layer,visible);update();},
    dispose(){root.removeFromParent();geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());root.clear();}};
}
