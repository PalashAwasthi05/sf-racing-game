import * as THREE from 'three';
import type {WoodySegment} from './tree-architecture.ts';

export const COARSE_LEAVES_PER_TREE=112;
export function leafSampleHash(x:number,y:number,z:number){
 let h=Math.imul(Math.round(x*256),73856093)^Math.imul(Math.round(y*256),19349663)^Math.imul(Math.round(z*256),83492791);
 h=Math.imul(h^h>>>16,0x7feb352d);return (h^h>>>15)>>>0;
}
type Sample={x:number;y:number;z:number;color:number[];hash:number;id:number};
const compare=(a:Sample,b:Sample)=>a.hash-b.hash||a.x-b.x||a.y-b.y||a.z-b.z||a.color[0]-b.color[0]||a.color[1]-b.color[1]||a.color[2]-b.color[2];
function sourceOwner(x:number,y:number,z:number,segments:WoodySegment[]){
 let owner='unanchored',best=Infinity;
 for(const s of segments){
  const ax=s.tip.x-s.root.x,ay=s.tip.y-s.root.y,az=s.tip.z-s.root.z;
  const t=Math.max(0,Math.min(1,((x-s.root.x)*ax+(y-s.root.y)*ay+(z-s.root.z)*az)/(ax*ax+ay*ay+az*az)));
  const dx=x-s.root.x-ax*t,dy=y-s.root.y-ay*t,dz=z-s.root.z-az*t,d=dx*dx+dy*dy+dz*dz;
  if(d<best){best=d;owner=s.owner;}
 }
 return owner;
}

/** Large measured crowns had the same spray size as saplings, leaving sparse
 * floating islands. Scale footprint from each source crown's sample density,
 * bounded by the original viewer's 1.65m maximum spray envelope. */
export function scaleLeafClusters(sources:THREE.Mesh[],segments:WoodySegment[]){
 const groups=new Map<string,{low:number[];high:number[];count:number}>();
 for(const mesh of sources){const p=mesh.geometry.getAttribute('position'),owners:string[]=[];
  for(let i=0;i<p.count;i++){
   const point=[p.getX(i),p.getY(i),p.getZ(i)],owner=sourceOwner(...point as [number,number,number],segments);owners.push(owner);
   let group=groups.get(owner);if(!group){group={low:[...point],high:[...point],count:0};groups.set(owner,group);}group.count++;
   for(let k=0;k<3;k++){group.low[k]=Math.min(group.low[k],point[k]);group.high[k]=Math.max(group.high[k],point[k]);}
  }
  mesh.userData.sourceLeafOwners=owners;
 }
 const scales=new Map<string,number>();for(const [owner,g] of groups){const volume=g.high.reduce((v,n,i)=>v*Math.max(.1,n-g.low[i]),1);scales.set(owner,owner==='unanchored'?1:Math.max(1,Math.min(1.45,Math.cbrt(volume/g.count)*1.05)));}
 for(const mesh of sources){const owners=mesh.userData.sourceLeafOwners as string[];mesh.geometry.setAttribute('foliageScale',new THREE.BufferAttribute(Float32Array.from(owners,o=>scales.get(o)??1),1));}
}

/** Coarse cards select existing source samples. Tree membership follows actual
 * source limbs; metric voxels, position hashes and extrema keep the result
 * stable when source material order or the viewer's arrival order changes. */
export function coarseLeafSources(sources:THREE.Mesh[],segments:WoodySegment[]){
 const voxels=new Map<string,Sample>(),extrema=new Map<string,Sample[]>(),sourceCounts=new Map<string,number>();let serial=0;
 for(const mesh of sources){
  const positions=mesh.geometry.getAttribute('position'),colors=mesh.geometry.getAttribute('color');
  for(let i=0;i<positions.count;i++){
   const x=positions.getX(i),y=positions.getY(i),z=positions.getZ(i),owner=(mesh.userData.sourceLeafOwners as string[]|undefined)?.[i]??sourceOwner(x,y,z,segments);
   sourceCounts.set(owner,(sourceCounts.get(owner)??0)+1);
   const sample={x,y,z,color:colors?[colors.getX(i),colors.getY(i),colors.getZ(i)]:[1,1,1],hash:leafSampleHash(x,y,z),id:serial++};
   const key=owner+'|'+Math.floor(x/1.55)+','+Math.floor(y/1.55)+','+Math.floor(z/1.55);
   if(!voxels.has(key)||compare(sample,voxels.get(key)!)<0)voxels.set(key,sample);
   let extremes=extrema.get(owner);if(!extremes){extremes=Array(6).fill(sample);extrema.set(owner,extremes);}
   for(let axis=0;axis<3;axis++){
    const prop=(['x','y','z'] as const)[axis],low=extremes[axis*2],high=extremes[axis*2+1];
    if(sample[prop]<low[prop]||sample[prop]===low[prop]&&compare(sample,low)<0)extremes[axis*2]=sample;
    if(sample[prop]>high[prop]||sample[prop]===high[prop]&&compare(sample,high)<0)extremes[axis*2+1]=sample;
   }
  }
 }
 const groups=new Map<string,Sample[]>();for(const [key,sample] of voxels){const owner=key.split('|')[0];if(!groups.has(owner))groups.set(owner,[]);groups.get(owner)!.push(sample);}
 const retained:Sample[]=[],compensation=new Map<number,number>();
 for(const [owner,samples] of groups){
  const selected=new Map<number,Sample>();for(const s of extrema.get(owner)??[])selected.set(s.id,s);
  for(const s of samples.sort(compare)){
   if(owner!=='unanchored'&&selected.size>=COARSE_LEAVES_PER_TREE)break;selected.set(s.id,s);
  }
  const density=Math.min(1.8,Math.sqrt((sourceCounts.get(owner)??selected.size)/Math.max(1,selected.size))*.95);
  for(const s of selected.values()){retained.push(s);compensation.set(s.id,density);}
 }
 // Spatial order is stable across material/run order and is also friendly to
 // the worker's later card construction and source-cell bounds.
 retained.sort((a,b)=>a.x-b.x||a.y-b.y||a.z-b.z);
 const p=new Float32Array(retained.length*3),c=new Uint16Array(retained.length*3),scale=new Float32Array(retained.length).fill(1);
 const scaleByPoint=new Map<string,number>();for(const mesh of sources){const p=mesh.geometry.getAttribute('position'),s=mesh.geometry.getAttribute('foliageScale');if(s)for(let i=0;i<p.count;i++)scaleByPoint.set(p.getX(i)+','+p.getY(i)+','+p.getZ(i),s.getX(i));}
 retained.forEach((s,i)=>{p.set([s.x,s.y,s.z],i*3);c.set(s.color.map(v=>Math.round(v*65535)),i*3);scale[i]=Math.min(2.1,(scaleByPoint.get(s.x+','+s.y+','+s.z)??1)*(compensation.get(s.id)??1));});
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(p,3));geometry.setAttribute('color',new THREE.BufferAttribute(c,3,true));geometry.setAttribute('foliageScale',new THREE.BufferAttribute(scale,1));
 const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xffffff}));mesh.userData.foliageFanCentres=true;
 return [mesh];
}
