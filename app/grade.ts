import * as THREE from 'three';
import {ShaderPass} from 'three/addons/postprocessing/ShaderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import type {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';

// The final look: a restrained bloom on the brightest highlights, then a film-style grade in linear light before
// tone mapping (contrast about middle grey in log space so highlights are not crushed, vibrance that lifts muted
// colours more than saturated ones, a warm/cool split between light and shade, a soft vignette). With filmic tone
// mapping this is what gives sunlit surfaces their punch and shaded ones their depth.
const GRADE={
 uniforms:{tDiffuse:{value:null},uContrast:{value:1.12},uVibrance:{value:.22},uSaturation:{value:.04},uWarm:{value:1},uVignette:{value:.22}},
 vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
 fragmentShader:`uniform sampler2D tDiffuse;uniform float uContrast,uVibrance,uSaturation,uWarm,uVignette;varying vec2 vUv;
 void main(){
  vec4 c=texture2D(tDiffuse,vUv);vec3 col=max(c.rgb,vec3(0.));
  vec3 lg=log2(col+vec3(1e-4));lg=(lg-log2(.18))*uContrast+log2(.18);col=exp2(lg)-vec3(1e-4);
  float lum=dot(col,vec3(.2126,.7152,.0722));float sat=max(max(col.r,col.g),col.b)-min(min(col.r,col.g),col.b);
  float boost=1.+uSaturation+uVibrance*(1.-clamp(sat/(lum+1e-3),0.,1.));col=max(mix(vec3(lum),col,boost),vec3(0.));
  float t=clamp(lum/(lum+.6),0.,1.);col*=mix(vec3(1.-.03*uWarm,1.,1.+.05*uWarm),vec3(1.+.035*uWarm,1.+.008*uWarm,1.-.05*uWarm),t);
  float d=length((vUv-.5)*vec2(1.,.9));col*=1.-uVignette*smoothstep(.42,.95,d);
  gl_FragColor=vec4(col,c.a);
 }`};
type Look={contrast:number;vibrance:number;saturation:number;warm:number;vignette:number;bloom:number;threshold:number};
const LOOKS:Record<string,Look>={
 afternoon:{contrast:1.12,vibrance:.22,saturation:.04,warm:1,vignette:.22,bloom:.28,threshold:1.15},
 golden:{contrast:1.08,vibrance:.2,saturation:.06,warm:1.5,vignette:.26,bloom:.36,threshold:1},
 overcast:{contrast:1.05,vibrance:.18,saturation:.02,warm:.25,vignette:.18,bloom:.18,threshold:1.3},
};
export function createGrade(composer:EffectComposer){
 const bloom=new UnrealBloomPass(new THREE.Vector2(512,512),.28,.45,1.15);
 const grade=new ShaderPass(GRADE);
 composer.addPass(bloom);composer.addPass(grade);
 function preset(name:string){
  const l=LOOKS[name]||LOOKS.afternoon;const u=grade.uniforms;
  u.uContrast.value=l.contrast;u.uVibrance.value=l.vibrance;u.uSaturation.value=l.saturation;u.uWarm.value=l.warm;u.uVignette.value=l.vignette;
  bloom.strength=l.bloom;bloom.threshold=l.threshold;
 }
 preset('afternoon');
 return {bloom,grade,preset,dispose(){bloom.dispose();grade.dispose();}};
}
