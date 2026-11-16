import * as THREE from 'three';
import {lazyBlockVariants} from './lazy-block-variants';
import {disposeOwnedTextures} from './material-resources';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';

export const variantLabels = {baseline:'Current mesh', photo:'Photo mesh', splats:'Gaussian splats'} as const;
export type BlockVariant = keyof typeof variantLabels;
export type BlockTrial = {label:string; manifest:string; routeEN:number[][]; lengthM:number; photo:string; splats:string; count:number; attribution?:string; sourceUrl?:string; comparisonUrl?:string};
export function blockVariant(value:string|null):BlockVariant {
  return value === 'photo' || value === 'splats' ? value : 'baseline';
}

// Distance along an intersection-to-intersection polyline, shared by the driver and the comparison replay.
export function blockPath(points:number[][]) {
  if(points.length<2 || points.some(p=>p.length<2 || !p.slice(0,2).every(Number.isFinite))) throw new Error('Invalid comparison route');
  const line=points.filter((p,i)=>i===0 || Math.hypot(p[0]-points[i-1][0],p[1]-points[i-1][1])>.001);
  const distances=[0];
  for(let i=1;i<line.length;i++) distances.push(distances[i-1]+Math.hypot(line[i][0]-line[i-1][0],line[i][1]-line[i-1][1]));
  const length=distances[distances.length-1];
  if(!length) throw new Error('Empty comparison route');
  function sample(m:number){
    const d=THREE.MathUtils.clamp(m,0,length);
    let i=1;while(i<distances.length-1 && distances[i]<d) i++;
    const p=line[i-1],q=line[i], t=(d-distances[i-1])/(distances[i]-distances[i-1]);
    return {x:THREE.MathUtils.lerp(p[0],q[0],t),z:-THREE.MathUtils.lerp(p[1],q[1],t),yaw:Math.atan2(p[0]-q[0],q[1]-p[1])};
  }
  function project(x:number,z:number){
    let best=Infinity,station=0;
    for(let i=1;i<line.length;i++){
      const p=line[i-1],q=line[i],dx=q[0]-p[0],dy=q[1]-p[1],l=distances[i]-distances[i-1];
      const f=THREE.MathUtils.clamp(((x-p[0])*dx+(-z-p[1])*dy)/(l*l),0,1);
      const error=Math.hypot(x-p[0]-f*dx,-z-p[1]-f*dy);
      if(error<best){best=error;station=distances[i-1]+l*f;}
    }
    return station;
  }
  return {length,sample,project};
}

// Each record describes a surface Gaussian with two independent tangent vectors in metres.
// Their perspective projection gives the ellipse its footprint and depth at every fragment.
function splatLayer(buffer:ArrayBuffer,count:number,textureLimit:number) {
  if(!Number.isSafeInteger(count) || count<1 || buffer.byteLength!==count*40) throw new Error('Incomplete Gaussian data');
  const stride=4;
  const width=Math.min(textureLimit,Math.ceil(Math.sqrt(count*stride)/stride)*stride);
  const height=Math.ceil(count*stride/width);
  if(height>textureLimit) throw new Error('Gaussian block exceeds the GPU texture limit');
  const texels=new Float32Array(width*height*4),f32=new Float32Array(buffer),bytes=new Uint8Array(buffer);
  const centres=new Float32Array(count*3), ids=new Float32Array(count);
  const linear=Float32Array.from({length:256},(_,i)=>THREE.ColorManagement.toWorkingColorSpace(new THREE.Color(i/255,0,0),THREE.SRGBColorSpace).r);
  for(let i=0;i<count;i++){
    const t=i*16,f=i*10,k=i*40;
    for(let j=0;j<3;j++){
      centres[i*3+j]=texels[t+j]=f32[f+j];
      texels[t+4+j]=f32[f+3+j];texels[t+8+j]=f32[f+6+j];
      texels[t+12+j]=linear[bytes[k+36+j]];
    }
    texels[t+15]=bytes[k+39]/255;ids[i]=i;
  }
  const texture=new THREE.DataTexture(texels,width,height,THREE.RGBAFormat,THREE.FloatType);
  texture.needsUpdate=true;
  const geometry=new THREE.InstancedBufferGeometry();
  geometry.setIndex([0,1,2,0,2,3]);
  geometry.setAttribute('position',new THREE.Float32BufferAttribute([-3,-3,0,3,-3,0,3,3,0,-3,3,0],3));
  const indices=new THREE.InstancedBufferAttribute(ids,1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('gaussianIndex',indices);geometry.instanceCount=count;
  const material=new THREE.ShaderMaterial({
    uniforms:{records:{value:texture},recordWidth:{value:width}},
    glslVersion:THREE.GLSL3,side:THREE.DoubleSide,forceSinglePass:true,transparent:true,depthWrite:false,depthTest:true,
    polygonOffset:true,polygonOffsetFactor:-1,polygonOffsetUnits:-1,
    vertexShader:`
      uniform sampler2D records;
      uniform int recordWidth;
      in float gaussianIndex;
      out vec2 localSigma;
      flat out vec4 rgba;
      vec4 record(int addr){return texelFetch(records,ivec2(addr%recordWidth,addr/recordWidth),0);}
      void main(){
        int base=int(gaussianIndex)*4;
        localSigma=position.xy;rgba=record(base+3);
        vec3 world=record(base).xyz+record(base+1).xyz*localSigma.x+record(base+2).xyz*localSigma.y;
        gl_Position=projectionMatrix*modelViewMatrix*vec4(world,1.);
      }`,
    fragmentShader:`
      in vec2 localSigma;
      flat in vec4 rgba;
      out vec4 fragColor;
      void main(){
        float r2=dot(localSigma,localSigma);
        if(r2>9.) discard;
        float alpha=rgba.a*exp(-.5*r2);
        if(alpha<.003) discard;
        fragColor=vec4(rgba.rgb,alpha);
      }`,
  });
  const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.name='block-trial-gaussians';
  const direction=new THREE.Vector3(),previous=new THREE.Vector3(Infinity,Infinity,Infinity);
  const bins=4096,counts=new Uint32Array(bins),binById=new Uint16Array(count),depths=new Float32Array(count);
  // Translating the camera shifts every depth equally. Only rotation can change the ordering.
  function sort(camera:THREE.Camera) {
    camera.getWorldDirection(direction);
    if(previous.distanceToSquared(direction)<.0001)return;
    previous.copy(direction);
    let lo=Infinity,hi=-Infinity;
    for(let i=0;i<count;i++){
      const j=i*3,d=centres[j]*direction.x+centres[j+1]*direction.y+centres[j+2]*direction.z;
      depths[i]=d;lo=Math.min(lo,d);hi=Math.max(hi,d);
    }
    counts.fill(0);
    const scale=(bins-1)/Math.max(hi-lo,1e-6);
    for(let i=0;i<count;i++){
      const b=Math.min(bins-1,Math.max(0,Math.floor((hi-depths[i])*scale)));
      counts[b]++;binById[i]=b;
    }
    let offset=0;
    for(let b=0;b<bins;b++){const n=counts[b];counts[b]=offset;offset+=n;}
    for(let i=0;i<count;i++)ids[counts[binById[i]]++]=i;
    indices.needsUpdate=true;
  }
  return {mesh,sort,dispose(){geometry.dispose();material.dispose();texture.dispose();}};
}

export function loadBlockLayers(trial:BlockTrial,renderer:THREE.WebGLRenderer) {
  const root=new THREE.Group();root.name='block-trial-appearance';
  const layers=lazyBlockVariants(async(variant,signal)=>{
    const response=await fetch(variant==='photo'?trial.photo:trial.splats,{signal});
    if(!response.ok)throw new Error('Could not load '+variantLabels[variant]+'.');
    const bytes=await response.arrayBuffer();
    if(variant==='splats'){
      const splats=splatLayer(bytes,trial.count,renderer.capabilities.maxTextureSize);
      splats.mesh.layers.set(1);splats.mesh.visible=false;root.add(splats.mesh);
      return {show(visible:boolean){splats.mesh.visible=visible;},update(camera:THREE.Camera){if(splats.mesh.visible)splats.sort(camera);},
        dispose(){splats.mesh.removeFromParent();splats.dispose();}};
    }
    const photo=await new GLTFLoader().parseAsync(bytes,trial.photo.slice(0,trial.photo.lastIndexOf('/')+1));
    const materials=new Set<THREE.Material>(),textures=new Set<THREE.Texture>();
    photo.scene.traverse(obj=>{
      obj.layers.set(1);
      if(!(obj instanceof THREE.Mesh))return;
      for(const m of Array.isArray(obj.material)?obj.material:[obj.material]){
        m.polygonOffset=true;m.polygonOffsetFactor=-2;m.polygonOffsetUnits=-2;materials.add(m);
        for(const value of Object.values(m))if(value instanceof THREE.Texture)textures.add(value);
      }
    });
    photo.scene.visible=false;root.add(photo.scene);
    return {show(visible:boolean){photo.scene.visible=visible;},update(_camera:THREE.Camera){},dispose(){
      photo.scene.removeFromParent();photo.scene.traverse(obj=>{if(obj instanceof THREE.Mesh)obj.geometry.dispose();});
      disposeOwnedTextures(textures);for(const material of materials)material.dispose();
    }};
  });
  // Overlays only join the driver's colour layer; the street keeps supplying
  // collisions, probes and shadows regardless of selection or loading state.
  return {root,set:layers.set,update(camera:THREE.Camera){layers.forEach(layer=>layer.update(camera));},
    dispose(){root.removeFromParent();layers.dispose();}};
}
