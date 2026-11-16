import * as THREE from 'three';

const finished=new WeakSet<THREE.Material>();

/** Finish source-crown fallback cards after atmosphere/CSM setup.
 * Reuses the viewer's shared spray texture without cloning/uploading it.
 * Metric detail removes coverage only inside the baked source silhouette;
 * brown source bark keeps its atlas color and alpha unchanged.
 */
export function finishCanopyImpostor(material:THREE.MeshStandardMaterial,leafTexture:THREE.Texture):void{
 if(finished.has(material))return;
 finished.add(material);
 const previous=material.onBeforeCompile.bind(material),key=material.customProgramCacheKey.bind(material);
 material.onBeforeCompile=(shader,renderer)=>{
  previous(shader,renderer);
  if(shader.fragmentShader.includes('canopyLeafShare'))return;
  shader.uniforms.uCanopyLeafSpray={value:leafTexture};
  shader.vertexShader=shader.vertexShader
   .replace('#include <common>','#include <common>\nvarying vec3 vCanopyMetricPosition;')
   .replace('#include <project_vertex>',/* glsl */`
    #include <project_vertex>
    vec4 canopyMetricPosition=vec4(transformed,1.0);
    #ifdef USE_BATCHING
     canopyMetricPosition=batchingMatrix*canopyMetricPosition;
    #endif
    #ifdef USE_INSTANCING
     canopyMetricPosition=instanceMatrix*canopyMetricPosition;
    #endif
    vCanopyMetricPosition=(modelMatrix*canopyMetricPosition).xyz;
   `);
  shader.fragmentShader=shader.fragmentShader
   .replace('#include <common>',/* glsl */`
    #include <common>
    varying vec3 vCanopyMetricPosition;
    uniform sampler2D uCanopyLeafSpray;
   `)
   .replace('#include <map_fragment>',/* glsl */`
    #include <map_fragment>
    // Test the original linear atlas albedo before lighting. Warm bark never
    // acquires a green texture or a new cutout, even within a mixed crown card.
    float canopyLeafShare=smoothstep(0.006,0.045,
     diffuseColor.g-max(diffuseColor.r,diffuseColor.b)*1.08);
    // The atlas stores raw source leaf albedo. Match the source-spray finish's
    // dark olive response even when derivative fading removes metric detail.
    float canopyLuma=dot(diffuseColor.rgb,vec3(0.2126,0.7152,0.0722));
    vec3 canopyOlive=mix(diffuseColor.rgb,canopyLuma*vec3(1.08,1.0,0.78),0.18)*0.72;
    diffuseColor.rgb=mix(diffuseColor.rgb,canopyOlive,canopyLeafShare);
    vec2 canopyMetricUv=vec2((vCanopyMetricPosition.x+vCanopyMetricPosition.z)*0.70710678,
     vCanopyMetricPosition.y)/1.45;
    float canopyFootprint=max(length(dFdx(canopyMetricUv)),length(dFdy(canopyMetricUv)));
    float canopyDetail=1.0-smoothstep(0.018,0.075,canopyFootprint);
    if(canopyLeafShare*canopyDetail>0.001){
     // Two overlapping sprays avoid a checker of isolated square clusters.
     // Derivative fading restores the original atlas at small screen sizes.
     vec4 canopySprayA=texture2D(uCanopyLeafSpray,fract(canopyMetricUv));
     vec2 canopyUvB=mat2(0.362,-0.932,0.932,0.362)*canopyMetricUv+vec2(0.43,0.61);
     vec4 canopySprayB=texture2D(uCanopyLeafSpray,fract(canopyUvB));
     float canopyCoverage=max(canopySprayA.a,canopySprayB.a);
     float canopyCut=smoothstep(0.12,0.70,canopyCoverage);
     diffuseColor.a*=mix(1.0,canopyCut,canopyLeafShare*canopyDetail);
     vec3 canopyRelative=mix(canopySprayB.rgb,canopySprayA.rgb,
      canopySprayA.a/max(0.001,canopySprayA.a+canopySprayB.a));
     diffuseColor.rgb*=mix(vec3(1.0),clamp(canopyRelative*0.90,vec3(0.27),vec3(0.96)),
      canopyLeafShare*canopyDetail);
    }
   `)
   .replace('#include <normal_fragment_maps>',/* glsl */`
    #include <normal_fragment_maps>
    // A crown receives the sky over a hemisphere, rather than lighting as two
    // intersecting flat walls. The existing CSM/direct/indirect paths remain.
    vec3 canopySkyNormal=normalize(mat3(viewMatrix)*vec3(0.0,1.0,0.0));
    vec3 canopySoftNormal=normalize(canopySkyNormal*0.90+normalize(vViewPosition)*0.35);
    // Use the same crown normal on either card. Retaining ten percent of the
    // two plane normals left a vertical dark stripe through each intersection.
    normal=normalize(mix(normal,canopySoftNormal,canopyLeafShare));
   `);
 };
 material.customProgramCacheKey=()=>key()+'|source-canopy-leaf-detail-v2';
 material.alphaToCoverage=true;
 material.needsUpdate=true;
}
