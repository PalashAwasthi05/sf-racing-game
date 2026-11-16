import * as THREE from 'three';

export type WoodySegment={root:THREE.Vector3;tip:THREE.Vector3;radius:number;owner:string};
const key=(p:THREE.Vector3)=>[p.x,p.y,p.z].map(v=>Math.round(v*100)).join(',');

/** Read intact source cylinders, recovering after ownership-clipped triangles.
 * Primary trunk/limb axes and tree positions are never synthesized. */
export function sourceWoodySegments(meshes:THREE.Mesh[]):WoodySegment[]{
 const segments:WoodySegment[]=[];
 const v=new THREE.Vector3();
 for(const mesh of meshes){
  const p=mesh.geometry.getAttribute('position'),ix=mesh.geometry.index;if(!ix)continue;
  for(let start=0;start+11<ix.count;){
   const rootIndex=ix.getX(start+1),tipIndex=ix.getX(start+11);let end=start+12;
   while(end+11<ix.count&&ix.getX(end+1)===rootIndex&&ix.getX(end+11)===tipIndex)end+=12;
   const sectors=(end-start)/12;
   if(sectors<5||sectors>32){start+=3;continue;}
   const root=new THREE.Vector3().fromBufferAttribute(p,rootIndex),tip=new THREE.Vector3().fromBufferAttribute(p,tipIndex);
   const radius=v.fromBufferAttribute(p,ix.getX(start)).distanceTo(root),length=root.distanceTo(tip);
   if(length>Math.max(.2,radius*5)&&tip.y-root.y>length*.1){segments.push({root,tip,radius,owner:''});start=end;}
   else start+=3;
  }
 }
 const stems=segments.filter(s=>s.tip.y-s.root.y>s.root.distanceTo(s.tip)*.97);
 for(const segment of segments){
  if(stems.includes(segment)){segment.owner=key(segment.tip);continue;}
  let best:WoodySegment|undefined,distance=.25;
  for(const stem of stems){const d=stem.tip.distanceTo(segment.root);if(d<distance){distance=d;best=stem;}}
  segment.owner=key(best?.tip??segment.root);
 }
 return segments;
}

export const TREE_TWIG_LIMIT=2048;
const SPACING=1.25;
type Node={point:THREE.Vector3;count:number;ix:number;iy:number;iz:number;owner:string;distance:number;anchor:THREE.Vector3;parent?:Node;weight:number;order:number};

/** A sparse connected scaffold under the existing leaf sprays. Occupied metric
 * leaf voxels grow toward existing wood, then toward earlier nearby voxels of
 * that same tree. All original samples remain exactly where they were. */
export function buildLeafTwigs(leaves:THREE.Mesh[],segments:WoodySegment[],limit=TREE_TWIG_LIMIT){
 const buckets=new Map<string,Node>();
 const point=new THREE.Vector3(),axis=new THREE.Vector3(),nearest=new THREE.Vector3();
 for(const mesh of leaves){
  const positions=mesh.geometry.getAttribute('position');
  for(let i=0;i<positions.count;i++){
   point.fromBufferAttribute(positions,i);
   const ix=Math.floor(point.x/SPACING),iy=Math.floor(point.y/SPACING),iz=Math.floor(point.z/SPACING),id=ix+','+iy+','+iz;
   let node=buckets.get(id);
   if(!node){node={point:new THREE.Vector3(),count:0,ix,iy,iz,owner:'',distance:Infinity,anchor:new THREE.Vector3(),weight:0,order:-1};buckets.set(id,node);}
   node.point.add(point);node.count++;
  }
 }
 for(const node of buckets.values()){
  node.point.divideScalar(node.count);node.weight=node.count;
  for(const s of segments){
   axis.subVectors(s.tip,s.root);const along=THREE.MathUtils.clamp(point.subVectors(node.point,s.root).dot(axis)/axis.lengthSq(),0,1);
   nearest.copy(s.root).addScaledVector(axis,along);
   const distance=nearest.distanceToSquared(node.point);
   if(distance<node.distance){node.distance=distance;node.anchor.copy(nearest);node.owner=s.owner;}
  }
 }
 // A distant unanchored sample does not justify inventing a trunk or a long
 // floating bridge. Its original foliage still renders, unchanged.
 const nodes=[...buckets.values()].filter(n=>n.distance<144).sort((a,b)=>a.distance-b.distance||a.ix-b.ix||a.iy-b.iy||a.iz-b.iz);
 const accepted:Node[]=[];
 for(const node of nodes){
  let bestCost=Math.sqrt(node.distance),parent:Node|undefined;
  for(let dx=-2;dx<=2;dx++)for(let dy=-2;dy<=2;dy++)for(let dz=-2;dz<=2;dz++){
   const other=buckets.get((node.ix+dx)+','+(node.iy+dy)+','+(node.iz+dz));
   if(!other||other.order<0||other.owner!==node.owner)continue;
   const d=node.point.distanceTo(other.point);
   // A small root-distance cost keeps growth directed out from the wood and
   // prevents nearly horizontal chains snaking around the crown surface.
   const cost=d+Math.sqrt(other.distance)*.12;
   if(d<SPACING*1.9&&cost<bestCost){bestCost=cost;parent=other;}
  }
  // Outer crown samples may grow through a chain of actual occupied leaf
  // voxels. They may not invent a long bare bridge directly to distant wood.
  if(!parent&&node.distance>36)continue;
  node.parent=parent;node.order=accepted.length;accepted.push(node);
 }
 for(let i=accepted.length-1;i>=0;i--)if(accepted[i].parent)accepted[i].parent!.weight+=accepted[i].weight;
 const candidates=accepted.filter(n=>(n.parent?.point??n.anchor).distanceToSquared(n.point)>.04);
 // Preserve complete support paths to outer clusters before filling interior
 // detail. Taking only the closest voxels left the highest tufts unsupported.
 const selected=new Set<Node>();
 for(const endpoint of [...candidates].sort((a,b)=>b.distance-a.distance||b.count-a.count||a.order-b.order)){
  const chain:Node[]=[];let node:Node|undefined=endpoint;
  while(node&&!selected.has(node)){if((node.parent?.point??node.anchor).distanceToSquared(node.point)>.04)chain.push(node);node=node.parent;}
  if(selected.size+chain.length<=limit)for(const part of chain)selected.add(part);
 }
 const paths=candidates.filter(n=>selected.has(n));
 if(!paths.length)return null;
 const positions=new Float32Array(paths.length*18),normals=new Int16Array(paths.length*18),colors=new Uint8Array(paths.length*18).fill(255);
 const indices=paths.length*6>65535?new Uint32Array(paths.length*18):new Uint16Array(paths.length*18);
 const direction=new THREE.Vector3(),a=new THREE.Vector3(),b=new THREE.Vector3(),normal=new THREE.Vector3();
 for(let i=0;i<paths.length;i++){
  const node=paths[i],root=node.parent?.point??node.anchor;
  direction.subVectors(node.point,root).normalize();a.crossVectors(direction,Math.abs(direction.y)>.95?new THREE.Vector3(1,0,0):new THREE.Vector3(0,1,0)).normalize();b.crossVectors(direction,a).normalize();
  const radius=Math.min(.034,.010+Math.sqrt(node.weight)*.0014),tipRadius=Math.max(.003,radius*.30);
  for(let ring=0;ring<2;ring++)for(let sector=0;sector<3;sector++){
   normal.copy(a).multiplyScalar(Math.cos(sector*Math.PI*2/3)).addScaledVector(b,Math.sin(sector*Math.PI*2/3));
   point.copy(ring?node.point:root).addScaledVector(normal,ring?tipRadius:radius);
   const offset=(i*6+ring*3+sector)*3;positions.set(point.toArray(),offset);normals.set(normal.toArray().map(v=>Math.round(v*32767)),offset);
  }
  for(let k=0;k<3;k++){
   const v=i*6,next=(k+1)%3,j=i*18+k*6;
   indices.set([v+k,v+next,v+k+3,v+next,v+next+3,v+k+3],j);
  }
 }
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3,true));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3,true));geometry.setIndex(new THREE.BufferAttribute(indices,1));
 geometry.userData.twigSegments=paths.length;
 geometry.userData.twigOwners=paths.map(n=>n.owner);
 return geometry;
}
