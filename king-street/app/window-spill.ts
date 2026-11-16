import * as THREE from 'three';
import type {GlazingEmitter} from './glazing';

type Occupancy = {cell:number; origin:number; n:number} & ({occupiedCell(index:number):boolean}|{mask:Uint8Array});
export function buildWindowSpill(windows: GlazingEmitter[], groundAt:(x:number,z:number)=>number, field?:Occupancy|null) {
  const occupied=(x:number,z:number)=>{
    if(!field)return false;
    const i=Math.floor((x-field.origin)/field.cell),j=Math.floor((z-field.origin)/field.cell);
    if(i<0||j<0||i>=field.n||j>=field.n)return false;
    const index=j*field.n+i;
    return 'occupiedCell' in field?field.occupiedCell(index):!!field.mask[index];
  };
  const sources=[];
  let xmin=Infinity,zmin=Infinity,xmax=-Infinity,zmax=-Infinity;
  for(const window of windows){
    const [x,y,z]=window.center, height=y-groundAt(x,z);
    if(height<.3||height>4.5||window.intensity<=0)continue;
    let [nx,nz]=window.normal;
    if(field) {
      // Recessed panes can sit slightly inside the shell. Compare several
      // offsets against the same foundation mask used for car collisions.
      let orientation=0;
      for(const d of [.6,1.1,1.9]) orientation+=Number(occupied(x-d*nx,z-d*nz))-Number(occupied(x+d*nx,z+d*nz));
      if(orientation<0){nx=-nx;nz=-nz;}
      else if(orientation===0&&!window.normalKnown)continue;
    } else if(!window.normalKnown)continue;
    const range=Math.min(9,3+Math.sqrt(window.width*window.height)*1.6);
    sources.push({w:window,x,y,z,nx,nz,range});
    xmin=Math.min(xmin,x-range);zmin=Math.min(zmin,z-range);
    xmax=Math.max(xmax,x+range);zmax=Math.max(zmax,z+range);
  }
  if(!sources.length){xmin=0;zmin=0;xmax=1;zmax=1;}
  const cell=Math.max(.70,(xmax-xmin)/1024,(zmax-zmin)/1024);
  const width=Math.max(1,Math.ceil((xmax-xmin)/cell)),height=Math.max(1,Math.ceil((zmax-zmin)/cell));
  const values=new Float32Array(width*height*4);
  for(const {w,x,y,z,nx,nz,range} of sources){
    const area=w.width*w.height;
    for(let j=Math.max(0,Math.floor((z-range-zmin)/cell));j<=Math.min(height-1,Math.floor((z+range-zmin)/cell));j++){
      const pz=zmin+(j+.5)*cell,dz=pz-z;
      for(let i=Math.max(0,Math.floor((x-range-xmin)/cell));i<=Math.min(width-1,Math.floor((x+range-xmin)/cell));i++){
        const px=xmin+(i+.5)*cell,dx=px-x, depth=dx*nx+dz*nz, r2=dx*dx+dz*dz;
        if(depth<.02||r2>=range*range||occupied(px,pz))continue;
        const ground=groundAt(px,pz),dy=y-ground;
        if(dy<.10)continue;
        // Area emitter, projected solid angle at both window and pavement.
        // The finite-aperture term avoids a point singularity near the sill.
        const denominator=r2+dy*dy+.17*area, fade=Math.max(0,1-r2/(range*range));
        const energy=3.6*w.intensity*area*depth*dy/(denominator*denominator)*fade*fade;
        const k=(j*width+i)*4;
        for(let c=0;c<3;c++)values[k+c]+=energy*w.color[c];
        values[k+3]+=energy*w.color[0]*ground;
      }
    }
  }
  const packed=new Uint16Array(values.length);
  for(let i=0;i<values.length;i++)packed[i]=THREE.DataUtils.toHalfFloat(values[i]);
  const texture=new THREE.DataTexture(packed,width,height,THREE.RGBAFormat,THREE.HalfFloatType);
  texture.magFilter=texture.minFilter=THREE.LinearFilter;texture.generateMipmaps=false;texture.needsUpdate=true;
  return {texture,bounds:new THREE.Vector4(xmin,zmin,1/(width*cell),1/(height*cell)),count:sources.length};
}
