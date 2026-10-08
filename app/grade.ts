import * as THREE from 'three';
import {ShaderPass} from 'three/addons/postprocessing/ShaderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {Pass, FullScreenQuad} from 'three/addons/postprocessing/Pass.js';
import type {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';

// The final look, between the lit image and the tone-mapped output:
//  exposure  the eye's adaptation: the scene's average log luminance (centre-weighted) is metered on the GPU through a
//            mip chain every frame, with no readback, and the exposure eases toward the value that brings it to a fixed
//            key within a limited range, about a second to adapt. A street in its buildings' shade was rendered at the
//            exposure of an open sunlit plaza and read dull and dark; a camera, an eye and the reference game all adapt.
//  bloom     a restrained glow on the brightest highlights.
//  grade     in linear light: contrast about middle grey in log space, vibrance that lifts muted colours more than
//            saturated ones, a warm/cool split between light and shade, a soft vignette, a light luminance sharpening
//            (texture detail softened by filtering and the multisampled resolve reads crisper) and a faint film grain.
const GRADE={
 uniforms:{tDiffuse:{value:null},uContrast:{value:1.12},uVibrance:{value:.22},uSaturation:{value:.04},uWarm:{value:1},uVignette:{value:.22},uSharpen:{value:.22},uGrain:{value:.03},uTime:{value:0}},
 vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
 fragmentShader:`uniform sampler2D tDiffuse;uniform float uContrast,uVibrance,uSaturation,uWarm,uVignette,uSharpen,uGrain,uTime;varying vec2 vUv;
 float luma(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
 void main(){
  vec2 px=1./vec2(textureSize(tDiffuse,0));
  vec4 c=texture2D(tDiffuse,vUv);vec3 col=max(c.rgb,vec3(0.));
  // luminance sharpening (unsharp mask on a 4-tap cross, in log space so it is even across light and shade)
  float l0=luma(col),la=luma(texture2D(tDiffuse,vUv+vec2(px.x,0.)).rgb)+luma(texture2D(tDiffuse,vUv-vec2(px.x,0.)).rgb)+luma(texture2D(tDiffuse,vUv+vec2(0.,px.y)).rgb)+luma(texture2D(tDiffuse,vUv-vec2(0.,px.y)).rgb);
  float d=log2(l0+1e-3)-log2(la*.25+1e-3);col*=exp2(clamp(d*uSharpen,-.25,.25));
  vec3 lg=log2(col+vec3(1e-4));lg=(lg-log2(.18))*uContrast+log2(.18);col=exp2(lg)-vec3(1e-4);
  float lum=luma(col);float sat=max(max(col.r,col.g),col.b)-min(min(col.r,col.g),col.b);
  float boost=1.+uSaturation+uVibrance*(1.-clamp(sat/(lum+1e-3),0.,1.));col=max(mix(vec3(lum),col,boost),vec3(0.));
  float t=clamp(lum/(lum+.6),0.,1.);col*=mix(vec3(1.-.03*uWarm,1.,1.+.05*uWarm),vec3(1.+.035*uWarm,1.+.008*uWarm,1.-.05*uWarm),t);
  float dv=length((vUv-.5)*vec2(1.,.9));col*=1.-uVignette*smoothstep(.42,.95,dv);
  // grain: multiplicative, strongest in the mid-tones, a new pattern every frame
  float n=fract(sin(dot(gl_FragCoord.xy+fract(uTime*.37)*vec2(61.7,23.3),vec2(12.9898,78.233)))*43758.5453)-.5;
  col*=1.+n*uGrain*(1.-abs(t-.45));
  gl_FragColor=vec4(col,c.a);
 }`};

const METER=`uniform sampler2D tDiffuse;varying vec2 vUv;
 void main(){vec3 c=texture2D(tDiffuse,vUv).rgb;float l=dot(max(c,vec3(0.)),vec3(.2126,.7152,.0722));
  // centre-weighted, and the sky's brightness counts only up to a sunlit wall's: a frame of sky must not darken the street
  vec2 d=(vUv-.5)*vec2(1.,.85);float w=exp(-dot(d,d)*3.5);float lg=log2(clamp(l,1e-4,.6));
  gl_FragColor=vec4(lg*w,w,0.,1.);}`;
// Partial adaptation, as an eye's: a view metered k stops darker than the key gets 0.4 k stops more exposure (at most
// 1.35x), so shade still reads darker than sun and the sky over a shaded street keeps its colour (at 1.7x it washed white).
const ADAPT=`uniform sampler2D tLum;uniform sampler2D tPrev;uniform float uKey,uComp,uMin,uMax,uRate;varying vec2 vUv;
 void main(){vec4 m=textureLod(tLum,vec2(.5),12.);float avg=m.x/max(m.y,1e-6);float target=clamp(exp2((uKey-avg)*uComp),uMin,uMax);
  float prev=texture2D(tPrev,vec2(.5)).r;float e=prev<=0.?target:mix(prev,target,uRate);gl_FragColor=vec4(e,avg,target,1.);}`;
const EXPOSE=`uniform sampler2D tDiffuse;uniform sampler2D tExp;varying vec2 vUv;
 void main(){vec4 c=texture2D(tDiffuse,vUv);gl_FragColor=vec4(c.rgb*texture2D(tExp,vec2(.5)).r,c.a);}`;
const VERT='varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}';

/** Eye adaptation: meters the HDR image on the GPU and scales it toward a fixed key, easing over about a second. */
class AutoExposurePass extends Pass {
 lum=new THREE.WebGLRenderTarget(256,128,{type:THREE.HalfFloatType,format:THREE.RGBAFormat,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false});
 adapt=[0,1].map(()=>new THREE.WebGLRenderTarget(1,1,{type:THREE.FloatType,format:THREE.RGBAFormat,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:false}));
 cur=0;snapNext=true;speed=1.6;
 meter=new THREE.ShaderMaterial({uniforms:{tDiffuse:{value:null}},vertexShader:VERT,fragmentShader:METER,depthTest:false,depthWrite:false});
 adaptMat=new THREE.ShaderMaterial({uniforms:{tLum:{value:null},tPrev:{value:null},uKey:{value:-2.5},uComp:{value:.4},uMin:{value:.75},uMax:{value:1.35},uRate:{value:1}},vertexShader:VERT,fragmentShader:ADAPT,depthTest:false,depthWrite:false});
 expose=new THREE.ShaderMaterial({uniforms:{tDiffuse:{value:null},tExp:{value:null}},vertexShader:VERT,fragmentShader:EXPOSE,depthTest:false,depthWrite:false});
 quad=new FullScreenQuad();
 render(renderer:THREE.WebGLRenderer,writeBuffer:THREE.WebGLRenderTarget,readBuffer:THREE.WebGLRenderTarget,deltaTime:number){
  const next=1-this.cur;
  this.meter.uniforms.tDiffuse.value=readBuffer.texture;this.quad.material=this.meter;renderer.setRenderTarget(this.lum);this.quad.render(renderer);
  const u=this.adaptMat.uniforms;u.tLum.value=this.lum.texture;u.tPrev.value=this.adapt[this.cur].texture;
  u.uRate.value=this.snapNext?1:1-Math.exp(-Math.max(0,Math.min(deltaTime||0,.25))*this.speed);this.snapNext=false;
  this.quad.material=this.adaptMat;renderer.setRenderTarget(this.adapt[next]);this.quad.render(renderer);
  this.expose.uniforms.tDiffuse.value=readBuffer.texture;this.expose.uniforms.tExp.value=this.adapt[next].texture;this.quad.material=this.expose;
  renderer.setRenderTarget(this.renderToScreen?null:writeBuffer);this.quad.render(renderer);
  this.cur=next;
 }
 /** Reads the current exposure, the metered average log2 luminance and the target (debugging; stalls the GPU). */
 read(renderer:THREE.WebGLRenderer){const b=new Float32Array(4);renderer.readRenderTargetPixels(this.adapt[this.cur],0,0,1,1,b);return {exposure:b[0],avgLog2:b[1],target:b[2]};}
 dispose(){this.lum.dispose();this.adapt.forEach(t=>t.dispose());this.meter.dispose();this.adaptMat.dispose();this.expose.dispose();this.quad.dispose();}
}

// key: the metered log2 luminance that keeps the base exposure (a sunlit street view)
type Look={contrast:number;vibrance:number;saturation:number;warm:number;vignette:number;bloom:number;threshold:number;key:number};
const LOOKS:Record<string,Look>={
 afternoon:{contrast:1.12,vibrance:.26,saturation:.06,warm:1.35,vignette:.2,bloom:.3,threshold:1.1,key:-2.5},
 golden:{contrast:1.08,vibrance:.2,saturation:.06,warm:1.6,vignette:.24,bloom:.38,threshold:1,key:-2.8},
 overcast:{contrast:1.08,vibrance:.2,saturation:.04,warm:.35,vignette:.18,bloom:.18,threshold:1.3,key:-2.6},
};
export function createGrade(composer:EffectComposer){
 const exposure=new AutoExposurePass();
 const bloom=new UnrealBloomPass(new THREE.Vector2(512,512),.28,.45,1.15);
 const grade=new ShaderPass(GRADE);
 composer.addPass(exposure);composer.addPass(bloom);composer.addPass(grade);
 const render=grade.render.bind(grade);grade.render=(...a:Parameters<typeof render>)=>{grade.uniforms.uTime.value=performance.now()/1000;render(...a);};
 function preset(name:string){
  const l=LOOKS[name]||LOOKS.afternoon;const u=grade.uniforms;
  u.uContrast.value=l.contrast;u.uVibrance.value=l.vibrance;u.uSaturation.value=l.saturation;u.uWarm.value=l.warm;u.uVignette.value=l.vignette;
  bloom.strength=l.bloom;bloom.threshold=l.threshold;exposure.adaptMat.uniforms.uKey.value=l.key;exposure.snapNext=true;
 }
 preset('afternoon');
 return {bloom,grade,exposure,preset,snap:()=>{exposure.snapNext=true;},dispose(){exposure.dispose();bloom.dispose();grade.dispose();}};
}
