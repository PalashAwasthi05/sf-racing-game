import * as THREE from 'three';

export function opticalCover(color:THREE.ColorRepresentation,absorption:number,roughness=.025){
 // Straight-alpha composition: R + A*tint + (1-F-A)*background.
 // Physical lighting already contains Fresnel R; dividing by coverage below
 // prevents the blend stage from attenuating that reflection a second time.
 const material=new THREE.MeshPhysicalMaterial({color,metalness:0,roughness,ior:1.5,
  transparent:true,opacity:1,depthWrite:false,side:THREE.DoubleSide});
 material.forceSinglePass=true;
 material.userData.roadsterOpticalAbsorption=absorption;
 material.onBeforeCompile=(shader)=>patchRoadsterOptics(material,shader);
 material.customProgramCacheKey=()=>'roadster-optical-v1';
 return material;
}

export function patchRoadsterOptics(material:THREE.Material,shader:Parameters<THREE.Material['onBeforeCompile']>[0]){
 const absorption=material.userData.roadsterOpticalAbsorption;
 if(typeof absorption!=='number')return;
  if(shader.fragmentShader.includes('roadsterOpticalCoverage'))return;
  shader.uniforms.roadsterOpticalAbsorption={value:absorption};
  shader.fragmentShader=shader.fragmentShader
   .replace('#include <common>','#include <common>\nuniform float roadsterOpticalAbsorption;')
   .replace('#include <opaque_fragment>',/* glsl */`
    float roadsterOpticalFresnel=max3(F_Schlick(material.specularColor,
     material.specularF90,saturate(abs(dot(normal,geometryViewDir)))));
    float roadsterOpticalCoverage=roadsterOpticalFresnel+
     (1.0-roadsterOpticalFresnel)*roadsterOpticalAbsorption;
    outgoingLight=(totalSpecular+totalDiffuse*(1.0-roadsterOpticalFresnel)*roadsterOpticalAbsorption+
     totalEmissiveRadiance)/max(roadsterOpticalCoverage,0.0001);
    diffuseColor.a=roadsterOpticalCoverage;
    #include <opaque_fragment>
   `);
 }
