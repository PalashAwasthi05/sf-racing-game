import * as THREE from 'three';
import {loadTreeCell,packTreeCell,type TreeCell,type TreeCellAsset,type PackedCell} from './tree-detail-geometry.ts';

// Texture image upload and all rendering remain on the main thread. Building
// cards, source normals, bounds and tapered branches needs no canvas or GPU.
const texture=new THREE.Texture();
self.onmessage=async(event:MessageEvent<{id:number;asset:TreeCellAsset;skeleton?:ArrayBuffer;quality?:'fine'|'coarse'}>)=>{
 if(event.data.skeleton){sourceSkeleton=new Float32Array(event.data.skeleton);return;}
 const {id,asset}=event.data;
 let cell:TreeCell|undefined,packed:PackedCell|undefined;
 try{
  cell=await loadTreeCell(asset,texture,()=>{},undefined,sourceSkeleton,event.data.quality);
  packed=packTreeCell(cell);
  const transfers=[...new Set(packed.meshes.flatMap(m=>[...m.attributes.map(a=>a.array.buffer),...(m.index?[m.index.buffer]:[])]))] as ArrayBuffer[];
  self.postMessage({id,packed},{transfer:transfers});
 }catch(error){self.postMessage({id,error:String(error)});}
 finally{
  // GPU-ready buffers have transferred ownership to the viewer. Keep only
  // the shared source-axis table between requests, including failed requests.
  cell?.dispose();cell=undefined;packed=undefined;
 }
};
let sourceSkeleton:Float32Array|undefined;
