import * as THREE from 'three';
import {STREET_PROBE_GLSL,type StreetProbeState} from './street-probe';
import type {TemporalResolve} from './temporal-resolve';

// The cube probe covers the unseen street. For nearby, glossy surfaces use the
// actual depth and radiance of visible objects wherever a reflected ray hits.
const TRACE=/* glsl */`
uniform mat4 uPreviousStreetClip;
uniform sampler2D uPreviousStreet;
uniform float uStreetHistory;
bool validStreetCoordinate(vec4 clip){
 if(!(clip.w>.25))return false;
 return all(lessThan(abs(clip.xy),vec2(clip.w*.996)));
}
bool validStreetDepth(float depth){return depth>0. && depth<65504.;}
bool validStreetRadiance(vec3 color){return all(greaterThanEqual(color,vec3(0.))) && all(lessThan(color,vec3(65504.)));}
vec4 traceStreet(vec3 origin,vec3 ray,float rough){
 vec4 start=uPreviousStreetClip*vec4(origin,1.), stepClip=uPreviousStreetClip*vec4(ray,0.);
 ivec2 dims=textureSize(uPreviousStreet,0);
 float lo=.11, distance=.20;
 for(int i=0;i<15;i++){
  vec4 q=start+distance*stepClip;
  if(!validStreetCoordinate(q))break;
  vec2 uv=q.xy/q.w*.5+.5;

  float z=texelFetch(uPreviousStreet,ivec2(uv*vec2(dims)),0).a;
  if(validStreetDepth(z) && q.w>=z){
   float hi=distance;
   for(int j=0;j<4;j++){
    float mid=(hi+lo)*.5;
    vec4 m=start+mid*stepClip;
    // The previous ray step can be behind the camera or outside the history.
    // A clamped edge texel is not evidence of a surface hit there.
    if(!validStreetCoordinate(m)){lo=mid;continue;}
    vec2 p=m.xy/m.w*.5+.5;
    float d=texelFetch(uPreviousStreet,clamp(ivec2(p*vec2(dims)),ivec2(0),dims-1),0).a;
    if(validStreetDepth(d) && m.w>d)hi=mid;else lo=mid;
   }
   q=start+hi*stepClip;if(!validStreetCoordinate(q))return vec4(0.);uv=q.xy/q.w*.5+.5;
   float actual=texelFetch(uPreviousStreet,ivec2(uv*vec2(dims)),0).a;
   if(!validStreetDepth(actual))return vec4(0.);
   float gap=q.w-actual;
   float confidence=(1.-smoothstep(.07,.22+q.w*.012,abs(gap)))*smoothstep(.18,.6,hi);
   vec2 edge=min(uv,1.-uv);
   confidence*=smoothstep(.003,.055,min(edge.x,edge.y))*(1.-smoothstep(20.,45.,hi));
   // Small roughness footprint, rejecting radiance from unrelated depths at silhouettes.
   vec4 center=textureLod(uPreviousStreet,uv,0.);
   if(!validStreetRadiance(center.rgb))return vec4(0.);
   vec3 sum=center.rgb;
   float weight=1.,spread=clamp(rough*rough*hi/max(q.w,.5)*float(dims.y)*.35,.5,9.);
   for(int j=0;j<4;j++){
    vec2 delta=j==0?vec2(.78,.33):(j==1?vec2(-.34,.88):(j==2?vec2(-.82,-.39):vec2(.43,-.76)));
    vec4 tap=textureLod(uPreviousStreet,uv+delta*spread/vec2(dims),0.);
    if(!validStreetDepth(tap.a) || !validStreetRadiance(tap.rgb))continue;
    float w=1.-smoothstep(.16,.32+.03*q.w,abs(tap.a-actual));sum+=w*tap.rgb;weight+=w;
   }
   return vec4(sum/weight,confidence);
  }
  lo=distance;distance=distance*1.34+.18;
 }
 return vec4(0.);
}`;

export function reflectStreet(materials:Iterable<THREE.MeshStandardMaterial>, temporal:TemporalResolve, probe?:StreetProbeState) {
 for(const material of new Set(materials)){
  const wet=material.userData.groundDetail?.startsWith('asphalt')===true;
  const previous=material.onBeforeCompile.bind(material),key=material.customProgramCacheKey.bind(material);
  material.onBeforeCompile=(shader,renderer)=>{
   previous(shader,renderer);
   Object.assign(shader.uniforms,temporal.reflectionUniforms);
   if(probe){
    Object.assign(shader.uniforms,{uStreetProbeOrigin:probe.origin,uStreetProbeEnabled:probe.enabled,uStreetProbeCapture:probe.capture});
    shader.defines={...shader.defines,STREET_DEPTH_PROBE:1};
    shader.fragmentShader=shader.fragmentShader.replace('#include <cube_uv_reflection_fragment>', '#include <cube_uv_reflection_fragment>\n'+STREET_PROBE_GLSL);
    if(!wet)shader.fragmentShader=shader.fragmentShader.replace('#include <envmap_physical_pars_fragment>',
     THREE.ShaderChunk.envmap_physical_pars_fragment.replace('vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );',
      'reflectVec=correctStreetDirection(envMap,cameraPosition-vViewPosition*mat3(viewMatrix),reflectVec,roughness);\nvec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );'));
   }
   shader.fragmentShader=shader.fragmentShader
     .replace('#include <common>','#include <common>\n'+TRACE)
     .replace('vec3 totalSpecular =', /* glsl */`
       float streetRough=${wet?'mix(.24,.13,gdFilm)':'material.roughness'};
       #ifdef USE_CLEARCOAT
       if(material.clearcoat>.015)streetRough=min(streetRough,material.clearcoatRoughness);
       #endif
       if(${probe?'uStreetProbeCapture<.5 && ':''}uStreetHistory>.5 && streetRough<.43 && vViewPosition.z<60. ${wet?'&& gdFilm>.40':''}){
        vec3 streetNormal=normal;
        #ifdef USE_CLEARCOAT
        if(material.clearcoat>.015)streetNormal=geometryClearcoatNormal;
        #endif
        vec3 N=inverseTransformDirection(streetNormal,viewMatrix);
        vec3 world=cameraPosition-vViewPosition*mat3(viewMatrix);
        vec3 R=reflect(normalize(world-cameraPosition),N);
        vec4 hit=traceStreet(world+N*.075,R,streetRough);
        float baseWeight=hit.a*${wet?'gdFilm':'(1.-smoothstep(.20,.43,material.roughness))'};
        vec3 brdf=EnvironmentBRDF(geometryNormal,geometryViewDir,${wet?'vec3(.0204),1.,streetRough':'material.specularColor,material.specularF90,material.roughness'});
        reflectedLight.indirectSpecular=mix(reflectedLight.indirectSpecular,hit.rgb*brdf,baseWeight*.90);
        #ifdef USE_CLEARCOAT
        vec3 coat=EnvironmentBRDF(geometryClearcoatNormal,geometryViewDir,material.clearcoatF0,material.clearcoatF90,material.clearcoatRoughness);
        clearcoatSpecularIndirect=mix(clearcoatSpecularIndirect,hit.rgb*coat,hit.a*.88);
        #endif
       }
       vec3 totalSpecular =`);
  };
  material.customProgramCacheKey=()=>key()+(probe?'|local-depth':'')+(wet?'|street-ray-wet':'|street-ray');
  material.needsUpdate=true;
 }
}
