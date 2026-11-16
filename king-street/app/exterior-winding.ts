import * as THREE from 'three';
import type {GLTF} from 'three/addons/loaders/GLTFLoader.js';

export type WallFaces={mesh:number;primitive:number;indexCount:number;vertexCount:number;triangles:number[]};
export type WallCell={id:string;contentHash:string;gpuContentHash?:string;records:WallFaces[]};

export async function loadWallFaces(url:string|undefined,worldHash:string) {
  const cells=new Map<string,WallCell>();
  if(!url||!/^\/compare\/[\w.-]+\.json$/.test(url))return cells;
  try{
    const response=await fetch(url);if(!response.ok)return cells;
    const data=await response.json() as {version?:number;scope?:string;sourceWorldHash?:string;cells?:WallCell[]}|null;
    if(data?.version!==1||data.scope!=='neighborhood-winding-corrections'||data.sourceWorldHash!==worldHash||!Array.isArray(data.cells))return cells;
    for(const cell of data.cells)if(cell&&typeof cell.id==='string'&&typeof cell.contentHash==='string'&&Array.isArray(cell.records)&&cell.records.length)cells.set(cell.id,cell);
  }catch{/* Frozen worlds can still open with their original geometry. */}
  return cells;
}

// Reverse only compiler-verified triangles. Duplicate an authored normal when
// its existing direction would shade the corrected outward face from inside.
// Neighboring triangles retain their original vertex, UV, tangent and normal.
export function orientWallGeometry(g:THREE.BufferGeometry,record:WallFaces){
  const index=g.index,position=g.attributes.position,normal=g.attributes.normal;
  if(!index||!position||index.count!==record.indexCount||position.count!==record.vertexCount||
     !Array.isArray(record.triangles)||Object.keys(g.morphAttributes).length)return false;
  const triangles=new Set(record.triangles);
  if(!triangles.size||[...triangles].some(t=>!Number.isSafeInteger(t)||t<0||3*t+2>=index.count))return false;
  const duplicates=new Map<number,number>();
  const indices=new Uint32Array(index.count);
  for(let i=0;i<index.count;i++)indices[i]=index.getX(i);
  const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),n=new THREE.Vector3();
  for(const t of triangles){
    const i=t*3,oldB=indices[i+1];indices[i+1]=indices[i+2];indices[i+2]=oldB;
    if(!normal)continue;
    a.fromBufferAttribute(position,indices[i]);b.fromBufferAttribute(position,indices[i+1]);c.fromBufferAttribute(position,indices[i+2]);
    b.sub(a).cross(c.sub(a));
    for(let j=i;j<i+3;j++){
      const v=indices[j];
      if(n.fromBufferAttribute(normal,v).dot(b)>=0)continue;
      if(!duplicates.has(v))duplicates.set(v,position.count+duplicates.size);
      indices[j]=duplicates.get(v)!;
    }
  }
  if(duplicates.size){
    for(const [name,attr] of Object.entries(g.attributes)){
      const inter='data' in attr?attr:undefined;
      const array=attr.array,stride=inter?.data.stride??attr.itemSize,offset=inter?.offset??0;
      const ArrayType=array.constructor as {new(length:number):typeof array};
      const storage=new ArrayType((attr.count+duplicates.size)*attr.itemSize);
      const copy=(src:number,dst:number)=>storage.set(array.subarray(src*stride+offset,src*stride+offset+attr.itemSize),dst*attr.itemSize);
      if(stride===attr.itemSize&&offset===0)storage.set(array);
      else for(let i=0;i<attr.count;i++)copy(i,i);
      for(const [src,dst] of duplicates)copy(src,dst);
      const extended=new THREE.BufferAttribute(storage,attr.itemSize,attr.normalized);
      if(name==='normal')for(const i of duplicates.values())extended.setXYZ(i,-extended.getX(i),-extended.getY(i),-extended.getZ(i));
      if(name==='tangent')for(const i of duplicates.values())extended.setW(i,-extended.getW(i));
      g.setAttribute(name,extended);
    }
  }
  g.setIndex(new THREE.BufferAttribute(position.count+duplicates.size<=65535?new Uint16Array(indices):indices,1));
  return true;
}

export function applyWallFaces(model:GLTF,cell:WallCell){
  const byPrimitive=new Map(cell.records.map(r=>[r.mesh+'/'+r.primitive,r]));
  const visited=new Set<THREE.BufferGeometry>();
  let corrected=0;
  model.scene.traverse(o=>{
    const mesh=o as THREE.Mesh;if(!mesh.isMesh||visited.has(mesh.geometry))return;
    const ref=model.parser.associations.get(mesh) as {meshes?:number;primitives?:number}|undefined;if(!ref)return;
    const record=byPrimitive.get(ref.meshes+'/'+ref.primitives);if(!record)return;
    visited.add(mesh.geometry);if(orientWallGeometry(mesh.geometry,record))corrected+=record.triangles.length;
  });
  return corrected;
}
