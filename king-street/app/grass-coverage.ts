import * as THREE from 'three';

// A conservative plan of the source triangles. Used only to skip a blade draw
// when its entire grid misses grass. The GPU's lawn mask still resolves edges,
// tree wells, paths and anything covering the original surface.
export function lawnCoverage(lawns:THREE.Object3D[],bin=8){
 const bounds=new THREE.Box3(),part=new THREE.Box3();
 for(const lawn of lawns)bounds.union(part.setFromObject(lawn));
 if(bounds.isEmpty())return {bounds,touches:()=>false};
 const {x:x0,z:z0}=bounds.min;
 const nx=Math.floor((bounds.max.x-x0)/bin)+1,nz=Math.floor((bounds.max.z-z0)/bin)+1;
 const ix=(x:number)=>Math.max(0,Math.min(nx-1,Math.floor((x-x0)/bin)));
 const iz=(z:number)=>Math.max(0,Math.min(nz-1,Math.floor((z-z0)/bin)));
 const occupied=new Uint8Array(Math.ceil(nx*nz/8));
 function mark(x:number,z:number){const i=z*nx+x;occupied[i>>>3]|=1<<(i&7);}
 const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3();
 for(const lawn of lawns)lawn.traverse(object=>{
  if(!(object instanceof THREE.Mesh))return;
  const g=object.geometry,p=g.getAttribute('position');
  if(!p)return;
  if(object instanceof THREE.BatchedMesh||object instanceof THREE.InstancedMesh){
   part.setFromObject(object);
   for(let z=iz(part.min.z);z<=iz(part.max.z);z++)for(let x=ix(part.min.x);x<=ix(part.max.x);x++)mark(x,z);
   return;
  }
  const vertex=(i:number,out:THREE.Vector3)=>out.fromBufferAttribute(p,g.index?g.index.getX(i):i).applyMatrix4(object.matrixWorld);
  for(let i=0,n=g.index?.count??p.count;i+2<n;i+=3){
   vertex(i,a);vertex(i+1,b);vertex(i+2,c);
   const xMin=ix(Math.min(a.x,b.x,c.x)),xMax=ix(Math.max(a.x,b.x,c.x));
   const zMin=iz(Math.min(a.z,b.z,c.z)),zMax=iz(Math.max(a.z,b.z,c.z));
   const winding=Math.sign((b.x-a.x)*(c.z-a.z)-(b.z-a.z)*(c.x-a.x));
   if(!winding)continue;
   for(let z=zMin;z<=zMax;z++)for(let x=xMin;x<=xMax;x++){
    const cx=x0+(x+.5)*bin,cz=z0+(z+.5)*bin;
    // Maximum signed distance among the four corners against each edge.
    const misses=(u:THREE.Vector3,v:THREE.Vector3)=>{
     const dx=v.x-u.x,dz=v.z-u.z;
     return winding*(dx*(cz-u.z)-dz*(cx-u.x))+.5*bin*(Math.abs(dx)+Math.abs(dz)) < -1e-6;
    };
    if(!misses(a,b)&&!misses(b,c)&&!misses(c,a))mark(x,z);
   }
  }
 });
 function touches(x:number,z:number,r:number){
  if(x+r<bounds.min.x||x-r>bounds.max.x||z+r<bounds.min.z||z-r>bounds.max.z)return false;
  const xMax=ix(x+r),zMax=iz(z+r);
  for(let row=iz(z-r);row<=zMax;row++)for(let col=ix(x-r);col<=xMax;col++){
   const i=row*nx+col;if(occupied[i>>>3]&(1<<(i&7)))return true;
  }
  return false;
 }
 return {bounds,touches};
}
