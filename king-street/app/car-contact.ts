import * as THREE from 'three';

// The car blocks the sky immediately underneath it. Screen-space occlusion
// loses this contact when the chase camera cannot see the underside.
export function carContact(body: THREE.Object3D, receivers: Set<THREE.Material>) {
  body.updateWorldMatrix(true,true);
  const inverse={value:body.matrixWorld.clone().invert()};
  const bounds=new THREE.Box3(), local=new THREE.Matrix4();
  body.traverse(o=>{
    if(!(o instanceof THREE.Mesh))return;
    o.geometry.computeBoundingBox();
    if(o.geometry.boundingBox)bounds.union(o.geometry.boundingBox.clone().applyMatrix4(local.multiplyMatrices(inverse.value,o.matrixWorld)));
  });
  const size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
  const footprint={value:new THREE.Vector4(center.x,center.z,size.x*.5,size.z*.5)};
  const enabled={value:0};
  for(const mat of receivers){
    if(!(mat instanceof THREE.MeshStandardMaterial))continue;
    const previous=mat.onBeforeCompile.bind(mat),key=mat.customProgramCacheKey.bind(mat);
    mat.onBeforeCompile=(shader,renderer)=>{
      previous(shader,renderer);
      Object.assign(shader.uniforms,{uUnderCar:enabled,uCarInverse:inverse,uCarFootprint:footprint,uCarHullHeight:{value:size.y}});
      shader.fragmentShader=shader.fragmentShader.replace('#include <common>',`#include <common>
        #define HAS_CAR_CONTACT
        uniform float uUnderCar,uCarHullHeight;
        uniform mat4 uCarInverse;uniform vec4 uCarFootprint;
        float carHullShadow(vec2 p,vec2 halfSize,float penumbra){
          // A rounded bumper contour, measured in car space.
          const float radius=.22;
          vec2 d=abs(p-uCarFootprint.xy)-halfSize+radius;
          float edge=length(max(d,vec2(0.)))+min(max(d.x,d.y),0.)-radius;
          return 1.-smoothstep(-penumbra,penumbra,edge);
        }
        float carLampVisibility(vec3 lightPosition,vec3 groundPosition){
          if(uUnderCar<.5)return 1.;
          vec3 p=(uCarInverse*vec4(groundPosition,1.)).xyz;
          if(abs(p.y)>.7 || dot(p.xz,p.xz)>225.)return 1.;
          vec3 delta=mat3(uCarInverse)*(lightPosition-groundPosition);
          if(delta.y<.6)return 1.;
          float low=(uCarHullHeight*.26-p.y)/delta.y,high=(uCarHullHeight*.56-p.y)/delta.y;
          vec2 base=p.xz+delta.xz*clamp(low,0.,1.);
          vec2 top=p.xz+delta.xz*clamp(high,0.,1.);
          float penumbra=.07+.09*min(3.,length(delta.xz)/delta.y);
          float cover=max(carHullShadow(base,uCarFootprint.zw*vec2(.96,.93),penumbra),
                          carHullShadow(top,uCarFootprint.zw*vec2(.84,.86),penumbra));
          return 1.-cover*.97;
        }`)
        .replace('#include <aomap_fragment>',`#include <aomap_fragment>
        if(uUnderCar>.5){
          vec3 world=cameraPosition-vViewPosition*mat3(viewMatrix);
          vec3 ground=(uCarInverse*vec4(world,1.)).xyz;
          vec2 d=abs(ground.xz-uCarFootprint.xy)-uCarFootprint.zw*vec2(.90,.84);
          float edge=length(max(d,vec2(0.)))+min(max(d.x,d.y),0.);
          float proximity=(1.-smoothstep(-.07,.60,edge))*(1.-smoothstep(.22,1.,abs(ground.y)));
          reflectedLight.indirectDiffuse*=1.-.83*proximity;
          reflectedLight.indirectSpecular*=1.-.78*proximity;
          reflectedLight.directDiffuse*=1.-.25*proximity;
        }`);
    };
    mat.customProgramCacheKey=()=>key()+'|car-contact';
    mat.needsUpdate=true;
  }
  return (on:boolean)=>{
    enabled.value=on?1:0;
    if(on){body.updateWorldMatrix(true,false);inverse.value.copy(body.matrixWorld).invert();}
  };
}
