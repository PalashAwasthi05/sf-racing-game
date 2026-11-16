import type {TreeCellAsset} from './tree-detail-geometry.ts';
import {COARSE_LEAVES_PER_TREE} from './tree-foliage-lod.ts';

export type TreeEstimate={bytes:number;geometryBytes:number;vertices:number;triangles:number;draws:number;leafCards:number;twigSegments:number};
const indexBytes=(vertices:number)=>vertices>65535?4:2;

/** Upper bounds for the requested variant, including CPU/GPU buffers and the
 * later Uint16 streamTier copies. Each coarse source-tree group retains at
 * most112 existing samples; every assigned owner comes from skeletonIds. */
export function estimateTreeCell(asset:TreeCellAsset,quality:'fine'|'coarse',sourceOwnerCount?:number):TreeEstimate{
 const entries=asset.entries,leaves=entries.filter(e=>e.kind==='leaves'),bark=entries.filter(e=>e.kind==='bark'),fronds=entries.filter(e=>e.kind==='frond');
 const samples=leaves.reduce((n,e)=>n+e.vertices,0);
 // With no source axes, coarse sampling has one uncapped unanchored group.
 const leafCards=quality==='coarse'&&sourceOwnerCount?Math.min(samples,sourceOwnerCount*COARSE_LEAVES_PER_TREE):samples;
 const twigSegments=asset.skeletonIds?.length?Math.min(leafCards,quality==='coarse'?768:2048):0;
 const barkVertices=bark.reduce((n,e)=>n+e.vertices,0),barkIndices=bark.reduce((n,e)=>n+e.indices!,0);
 const frondVertices=fronds.reduce((n,e)=>n+e.vertices,0),frondIndices=fronds.reduce((n,e)=>n+e.indices!,0);
 const twigVertices=twigSegments*6,twigIndices=twigSegments*18;
 const woodyVertices=barkVertices+twigVertices;
 const leafVertices=leafCards*4;
 const leafBytes=leafVertices*32+leafCards*6*indexBytes(leafVertices);
 // Two or more pieces merge to Float32 positions + normalized Int16 normals
 // and Uint16 colors. A single native mesh retains21bytes/vertex and its
 // original Uint32 indices. Account for either case when twig count is only
 // bounded, so even a cell producing zero twigs remains covered.
 const nativeBarkBytes=bark.length>1?barkVertices*24+barkIndices*indexBytes(barkVertices):barkVertices*21+barkIndices*4;
 const mergedWoodyBytes=woodyVertices*24+(barkIndices+twigIndices)*indexBytes(woodyVertices);
 const woodyBytes=twigSegments?Math.max(nativeBarkBytes,mergedWoodyBytes):nativeBarkBytes;
 const frondBytes=fronds.length>1?frondVertices*24+frondIndices*indexBytes(frondVertices):frondVertices*21+frondIndices*4;
 const geometryBytes=leafBytes+woodyBytes+frondBytes,vertices=leafVertices+woodyVertices+frondVertices;
 return {bytes:geometryBytes*2+asset.byteLength+vertices*4,geometryBytes,vertices,
  triangles:leafCards*2+(barkIndices+frondIndices)/3+twigSegments*6,
  draws:Number(leafCards>0)+Number(woodyVertices>0)+Number(frondVertices>0),leafCards,twigSegments};
}
