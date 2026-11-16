import * as THREE from 'three';
import {renderPostQuad} from './postprocessing-utils.ts';
import {Pass, FullScreenQuad} from 'three/addons/postprocessing/Pass.js';
import type {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import type {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';

// Everything between the lit scene and the screen, in two full-screen passes (at a 5K window every full-screen pass of a
// half-float image costs about half a millisecond, and the chain had seven):
//  exposure  the eye's adaptation: the scene's average log luminance (centre-weighted) is metered on the GPU through a
//            mip chain every frame, with no readback, and the exposure eases toward the value that brings it to a fixed
//            key within a limited range, about a second to adapt. A street in its buildings' shade was rendered at the
//            exposure of an open sunlit plaza and read dull and dark; a camera, an eye and the reference game all adapt.
//  composite the detail-scale ambient occlusion (GTAO) on the indirect share of each pixel's light (occlusion.ts writes
//            that share into the scene's alpha), and the exposure: the first full-screen pass.
//  bloom     a restrained glow on the brightest highlights, at a quarter of the resolution and below (a progressive
//            down- and upsampled mip chain); the last pass adds it.
//  grade     in linear light: contrast about middle grey in log space, vibrance that lifts muted colours more than
//            saturated ones, a warm/cool split between light and shade, a soft vignette, a light luminance sharpening
//            (texture detail softened by filtering and the multisampled resolve reads crisper) and a faint film grain;
//            then the tone mapping (ACES filmic) and the output transfer: the second full-screen pass, to the screen.
const VERT = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}';

const METER = `uniform sampler2D tDiffuse;varying vec2 vUv;
 void main(){vec3 c=texture2D(tDiffuse,vUv).rgb;
  // A corrupt sample must carry zero metering weight before mip reduction.
  if(any(isnan(c))||any(isinf(c))){gl_FragColor=vec4(0.);return;}
  float l=dot(max(c,vec3(0.)),vec3(.2126,.7152,.0722));
  // centre-weighted, and the sky's brightness counts only up to a sunlit wall's: a frame of sky must not darken the street
  vec2 d=(vUv-.5)*vec2(1.,.85);float w=exp(-dot(d,d)*3.5);float lg=log2(clamp(l,1e-4,.6));
  gl_FragColor=vec4(lg*w,w,0.,1.);}`;
// Partial adaptation, as an eye's: a view metered k stops darker than the key gets 0.5 k stops more exposure (at most
// 1.8x), so shade still reads darker than sun and the sky over a shaded street keeps its colour. Measured 2026-11-12 on a
// street in building shade (metered 2.8 stops under the key): at the old 1.35x cap the road sat at sRGB 67; at 1.8x it is
// 81 with the sky at (158,174,208), still blue; at 2.4x the sky reaches (180,194,221) and starts to wash out.
const ADAPT = `uniform sampler2D tLum;uniform sampler2D tPrev;uniform float uKey,uComp,uMin,uMax,uRate;varying vec2 vUv;
 void main(){vec4 m=textureLod(tLum,vec2(.5),12.);float prev=texture2D(tPrev,vec2(.5)).r;
  bool metered=m.y>1e-6&&!any(isnan(m.xy))&&!any(isinf(m.xy));
  float avg=metered?m.x/m.y:uKey;
  bool hasPrevious=prev>=uMin&&prev<=uMax;
  float target=metered?clamp(exp2((uKey-avg)*uComp),uMin,uMax):(hasPrevious?prev:1.);
  float e=hasPrevious?mix(prev,target,uRate):target;gl_FragColor=vec4(e,avg,target,1.);}`;

/** Eye adaptation: meters the HDR scene on the GPU and eases an exposure toward a fixed key; the exposure stays on the GPU
 *  (a 1x1 texture) for the composite to apply. Draws nothing full-screen. */
class ExposurePass extends Pass {
 private source?: THREE.Texture;
 lum = new THREE.WebGLRenderTarget(256, 128, {type: THREE.HalfFloatType, format: THREE.RGFormat, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
 adapt = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false}));
 cur = 0; snapNext = true; speed = 1.6;
 private sinceMeter = 0;
 meter = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}}, vertexShader: VERT, fragmentShader: METER, depthTest: false, depthWrite: false});
 adaptMat = new THREE.ShaderMaterial({uniforms: {tLum: {value: null}, tPrev: {value: null}, uKey: {value: -2.5}, uComp: {value: .5}, uMin: {value: .75}, uMax: {value: 1.8}, uRate: {value: 1}}, vertexShader: VERT, fragmentShader: ADAPT, depthTest: false, depthWrite: false});
 quad = new FullScreenQuad();
 /** `source`: the image to meter; without one, the composer's current image */
 constructor(source?: THREE.Texture) { super(); this.source=source; this.needsSwap = false; }
 get texture() { return this.adapt[this.cur].texture; }
 render(renderer: THREE.WebGLRenderer, _w: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget, deltaTime: number) {
  const next = 1 - this.cur;
  // The meter measures a slowly adapting eye. Keep adaptation continuous at
  // display rate while reusing a recent log-luminance pyramid between readings.
  this.sinceMeter += Math.max(0,deltaTime || 0);
  if (this.snapNext || this.sinceMeter >= 1 / 15) {
    this.meter.uniforms.tDiffuse.value = this.source ?? readBuffer.texture;
    this.quad.material = this.meter;
    renderPostQuad(renderer,this.lum,this.quad);
    this.sinceMeter = 0;
  }
  const u = this.adaptMat.uniforms; u.tLum.value = this.lum.texture; u.tPrev.value = this.adapt[this.cur].texture;
  u.uRate.value = this.snapNext ? 1 : 1 - Math.exp(-Math.max(0, Math.min(deltaTime || 0, .25)) * this.speed); this.snapNext = false;
  this.quad.material = this.adaptMat; renderPostQuad(renderer,this.adapt[next],this.quad);
  this.cur = next;
 }
 /** Reads the current exposure, the metered average log2 luminance and the target (debugging; stalls the GPU). */
 read(renderer: THREE.WebGLRenderer) { const b = new Float32Array(4); renderer.readRenderTargetPixels(this.adapt[this.cur], 0, 0, 1, 1, b); return {exposure: b[0], avgLog2: b[1], target: b[2]}; }
 dispose() { this.lum.dispose(); this.adapt.forEach(t => t.dispose()); this.meter.dispose(); this.adaptMat.dispose(); this.quad.dispose(); }
}

/** The scene (resolved from its multisampled target, read directly: no copy), times the detail-scale occlusion on its
 *  indirect share, times the exposure. Without a scene texture it reads the composer's current image, and without an
 *  occlusion pass it applies the exposure alone. */
const WHITE = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1); WHITE.needsUpdate = true;
const ZERO_OFFSET = new THREE.Vector2();
// Reproject static street surfaces through the previous driving camera. The
// occlusion pass already contains the camera-space depth. Use that to reject
// taps across depth discontinuities and to leave the player's moving body sharp.
const MOTION = /* glsl */`
 uniform sampler2D tMotionDepth;
 uniform mat4 uPreviousClip, uClipToCar;
 uniform vec3 uCarMin,uCarMax;
 uniform float uShutter;
 vec3 motionColor(vec3 centre) {
  if(uShutter<=0.) return centre;
  float depth=texture2D(tMotionDepth,vUv+uSourceOffset).r;
  if(depth>=.999999) return centre;
  vec4 clip=vec4(vUv*2.-1.,depth*2.-1.,1.);
  vec4 previous=uPreviousClip*clip;
  if(previous.w<=0.) return centre;
  vec2 trail=(previous.xy/previous.w*.5+.5-vUv)*uShutter;
  vec2 dimensions=vec2(textureSize(tScene,0));
  float pixels=length(trail*dimensions);
  if(pixels<.75) return centre;
  vec4 body=uClipToCar*clip;
  vec3 point=body.xyz/body.w;
  vec3 outside=max(uCarMin-point,point-uCarMax);
  float clearance=max(outside.x,max(outside.y,outside.z));
  if(clearance<=.08) return centre;
  // Limit the angular smear across output sizes so street detail stays
  // readable at speed, including in the smaller driving preview.
  trail*=min(1.,(dimensions.y/180.)/pixels)*smoothstep(.08,.40,clearance);
  vec3 sum=centre;
  float weight=1.;
  // For perspective depth the distance to the far-depth endpoint is proportional
  // to reciprocal range. Compare relative range so foreground edges stay crisp.
  float tolerance=max(1e-6,(1.-depth)*.14);
  for(int i=1;i<=4;i++){
   vec2 sampleUV=vUv+trail*(float(i)*.25);
   float other=textureLod(tMotionDepth,sampleUV+uSourceOffset,0.).r;
   float tap=(1.-smoothstep(.25,1.,abs(other-depth)/tolerance))*(1.-.08*float(i));
   sum+=textureLod(tScene,sampleUV,0.).rgb*tap;
   weight+=tap;
  }
  return sum/weight;
 }`;

/** Static local body bounds survive suspension and wheel rotation. Cache the
 * hierarchy walk once, then update its world transform with the driving camera. */
function motionControl(pass: CompositePass, ao: GTAOPass | undefined, sceneDepth?:THREE.Texture) {
 const uniforms=pass.material.uniforms;
 const lastClip=new THREE.Matrix4(),currentClip=new THREE.Matrix4(),clipInverse=new THREE.Matrix4();
 const previousEye=new THREE.Vector3(),previousOrientation=new THREE.Quaternion();
 const localBounds=new WeakMap<THREE.Object3D,THREE.Box3>();
 let tracked: THREE.Object3D|null=null;
 function reset(){tracked=null;uniforms.uShutter.value=0;}
 function update(camera:THREE.PerspectiveCamera,car:THREE.Object3D|null,speedKmh:number,dt:number) {
  uniforms.uShutter.value=0;
  if((!sceneDepth && !ao?.enabled) || !car || speedKmh<12 || dt<=0 || dt>.075){tracked=null;return;}
  camera.updateMatrixWorld();
  currentClip.multiplyMatrices(camera.projectionMatrix,camera.matrixWorldInverse);
  const continuous=tracked===car && previousEye.distanceToSquared(camera.position)<16 &&
   previousOrientation.angleTo(camera.quaternion)<.23;
  if(continuous) {
   let bounds=localBounds.get(car);
   if(!bounds) {
    bounds=new THREE.Box3();
    car.updateWorldMatrix(true,true);
    const worldToCar=car.matrixWorld.clone().invert(),localBox=new THREE.Box3(),toCar=new THREE.Matrix4();
    car.traverse(child=>{
     if(!(child instanceof THREE.Mesh) || !child.geometry?.attributes.position)return;
     const geometry=child.geometry;
     if(!geometry.boundingBox)geometry.computeBoundingBox();
     if(geometry.boundingBox)bounds!.union(localBox.copy(geometry.boundingBox).applyMatrix4(toCar.multiplyMatrices(worldToCar,child.matrixWorld)));
    });
    localBounds.set(car,bounds);
   }
   if(!bounds.isEmpty()){
    clipInverse.copy(currentClip).invert();
    (uniforms.uPreviousClip.value as THREE.Matrix4).multiplyMatrices(lastClip,clipInverse);
    car.updateWorldMatrix(true,false);
    (uniforms.uClipToCar.value as THREE.Matrix4).copy(car.matrixWorld).invert().multiply(clipInverse);
    (uniforms.uCarMin.value as THREE.Vector3).copy(bounds.min);
    (uniforms.uCarMax.value as THREE.Vector3).copy(bounds.max);
    uniforms.tMotionDepth.value=sceneDepth??ao!.depthTexture;
    uniforms.uShutter.value=Math.min(1,1/(180*dt))*THREE.MathUtils.smoothstep(speedKmh,12,60);
   }
  }
  tracked=car;lastClip.copy(currentClip);previousEye.copy(camera.position);previousOrientation.copy(camera.quaternion);
 }
 return {update,reset};
}

// Exposure and bloom need positive RGB only. The beauty and temporal targets
// retain full RGBA because alpha holds indirect lighting and depth respectively.
function gradedColorTarget() {
 return new THREE.WebGLRenderTarget(1,1,{format:THREE.RGBFormat,type:THREE.UnsignedInt101111Type,
   internalFormat:'R11F_G11F_B10F',minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false});
}
class CompositePass extends Pass {
 private scene: THREE.Texture | undefined;
 private ao: GTAOPass | undefined;
 private exposure: ExposurePass;
 private resolved?:{value:THREE.Texture|null;sampleOffset?:THREE.Vector2};
 readonly output=gradedColorTarget();
 setSize(w:number,h:number){this.output.setSize(w,h);}
 material: THREE.ShaderMaterial; quad: FullScreenQuad;
 constructor(scene: THREE.Texture | undefined, ao: GTAOPass | undefined, exposure: ExposurePass, resolved?:{value:THREE.Texture|null;sampleOffset?:THREE.Vector2}) {
  super();this.needsSwap=false;
  this.scene=scene;this.ao=ao;this.exposure=exposure;this.resolved=resolved;
  this.material = new THREE.ShaderMaterial({
   uniforms: {tRawScene:{value:scene},tScene: {value: scene}, tAO: {value: null}, tExp: {value: null}, uStrength: {value: 1},uSourceOffset:{value:new THREE.Vector2()},
    tMotionDepth:{value:WHITE},uShutter:{value:0},uPreviousClip:{value:new THREE.Matrix4()},uClipToCar:{value:new THREE.Matrix4()},
    uCarMin:{value:new THREE.Vector3()},uCarMax:{value:new THREE.Vector3()}},
   vertexShader: VERT,
   fragmentShader: /* glsl */`uniform sampler2D tRawScene;uniform sampler2D tScene;uniform sampler2D tAO;uniform sampler2D tExp;uniform float uStrength;uniform vec2 uSourceOffset;varying vec2 vUv;
    ${MOTION}
    void main(){vec4 raw=texture2D(tRawScene,vUv+uSourceOffset);vec4 c=texture2D(tScene,vUv);
     c.rgb=motionColor(c.rgb);if(any(isnan(c.rgb))||any(isinf(c.rgb)))c.rgb=raw.rgb;
     float ao=texture2D(tAO,vUv+uSourceOffset).r;if(!(ao>=0.&&ao<=1.))ao=1.;
     float exposure=texture2D(tExp,vec2(.5)).r;if(!(exposure>0.)||isinf(exposure))exposure=1.;
     c.rgb*=mix(1.,ao,clamp(raw.a,0.,1.)*uStrength)*exposure;
     // one NaN or infinite pixel here would spread through the bloom's mip chain into a block
     if(!(c.r==c.r&&c.g==c.g&&c.b==c.b))c.rgb=vec3(0.);gl_FragColor=vec4(min(c.rgb,vec3(6e4)),1.);}`,
   depthTest: false, depthWrite: false});
  this.quad = new FullScreenQuad(this.material);
 }
 render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget) {
  const u = this.material.uniforms;
  u.tRawScene.value=this.scene??readBuffer.texture;u.tScene.value=this.resolved?.value??u.tRawScene.value;
  (u.uSourceOffset.value as THREE.Vector2).copy(this.resolved?.sampleOffset??ZERO_OFFSET);
  u.tAO.value = this.ao?.enabled ? (this.ao as unknown as {pdRenderTarget: THREE.WebGLRenderTarget}).pdRenderTarget.texture : WHITE; u.uStrength.value = this.ao?.enabled ? .65 : 0; u.tExp.value = this.exposure.texture;
  renderPostQuad(renderer,this.output,this.quad);
 }
 dispose() { this.output.dispose();this.material.dispose(); this.quad.dispose(); }
}

// The bloom: a soft-thresholded, firefly-suppressed downsample to a quarter of the resolution, four more halvings, then
// back up, each level adding a tent-filtered upsample of the one below to its own (the dual-filter chain of mobile
// renderers): a wide, smooth glow from about a dozen small passes.
const DOWN0 = /* glsl */`uniform sampler2D tDiffuse;uniform vec2 uTexel;uniform float uThreshold,uScatterLimit;varying vec2 vUv;
 vec3 tap(vec2 o){vec3 c=min(texture2D(tDiffuse,vUv+o*uTexel).rgb,vec3(6e4));return c/(1.+max(max(c.r,c.g),c.b)*.25);}
 void main(){
  // four bilinear taps cover the 4x4 source texels of this quarter-resolution texel; each is weighted down by its own
  // brightness (a lone bright texel cannot flash a whole bloom tile on and off as it moves)
  vec3 c=(tap(vec2(-1.,-1.))+tap(vec2(1.,-1.))+tap(vec2(-1.,1.))+tap(vec2(1.,1.)))*.25;c/=max(1.-max(max(c.r,c.g),c.b)*.25,1e-3);
  float b=dot(c,vec3(.2126,.7152,.0722)),k=uThreshold*.5,s=clamp(b-uThreshold+k,0.,2.*k);s=s*s/(4.*k+1e-4);
  vec3 spill=c*max(s,b-uThreshold)/max(b,1e-4);
  // A solar reflection can be hundreds of times brighter than the body paint.
  // Roll off the light fed to the wide halo while preserving the specular core.
  float peak=max(spill.r,max(spill.g,spill.b));
  float excess=max(peak-uScatterLimit,0.);
  float shoulder=min(peak,uScatterLimit)+excess/(1.+excess/uScatterLimit);
  gl_FragColor=vec4(spill*(shoulder/max(peak,1e-5)),1.);}`;
const DOWN = /* glsl */`uniform sampler2D tDiffuse;uniform vec2 uTexel;varying vec2 vUv;
 void main(){vec2 h=uTexel*.5;vec3 c=texture2D(tDiffuse,vUv).rgb*4.+texture2D(tDiffuse,vUv-h).rgb+texture2D(tDiffuse,vUv+h).rgb+texture2D(tDiffuse,vUv+vec2(h.x,-h.y)).rgb+texture2D(tDiffuse,vUv-vec2(h.x,-h.y)).rgb;gl_FragColor=vec4(c*.125,1.);}`;
const UP = /* glsl */`uniform sampler2D tDiffuse;uniform sampler2D tLevel;uniform vec2 uTexel;uniform float uSpread;varying vec2 vUv;
 void main(){vec2 h=uTexel*.5;
  vec3 c=texture2D(tDiffuse,vUv+vec2(-h.x*2.,0.)).rgb+texture2D(tDiffuse,vUv+vec2(-h.x,h.y)).rgb*2.+texture2D(tDiffuse,vUv+vec2(0.,h.y*2.)).rgb+texture2D(tDiffuse,vUv+vec2(h.x,h.y)).rgb*2.
   +texture2D(tDiffuse,vUv+vec2(h.x*2.,0.)).rgb+texture2D(tDiffuse,vUv+vec2(h.x,-h.y)).rgb*2.+texture2D(tDiffuse,vUv+vec2(0.,-h.y*2.)).rgb+texture2D(tDiffuse,vUv+vec2(-h.x,-h.y)).rgb*2.;
  gl_FragColor=vec4(texture2D(tLevel,vUv).rgb+c/12.*uSpread,1.);}`;

class BloomPass extends Pass {
 private source:THREE.Texture;
 levels: THREE.WebGLRenderTarget[] = []; ups: THREE.WebGLRenderTarget[] = [];
 down0 = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, uTexel: {value: new THREE.Vector2()}, uThreshold: {value: 1.1}, uScatterLimit: {value: 4}}, vertexShader: VERT, fragmentShader: DOWN0, depthTest: false, depthWrite: false});
 down = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, uTexel: {value: new THREE.Vector2()}}, vertexShader: VERT, fragmentShader: DOWN, depthTest: false, depthWrite: false});
 up = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, tLevel: {value: null}, uTexel: {value: new THREE.Vector2()}, uSpread: {value: 1}}, vertexShader: VERT, fragmentShader: UP, depthTest: false, depthWrite: false});
 quad = new FullScreenQuad(); strength = .3; threshold = 1.1; radius = .45; full = new THREE.Vector2(1, 1);
 constructor(source:THREE.Texture,count = 5) {
  super(); this.needsSwap = false;
  this.source=source;
  const rt = gradedColorTarget;
  for (let i = 0; i < count; i++) { this.levels.push(rt()); if (i < count - 1) this.ups.push(rt()); }
 }
 /** The glow at a quarter of the resolution, to add (times `strength`) in the last pass. */
 get texture() { return (this.ups[0] ?? this.levels[0]).texture; }
 setSize(width: number, height: number) {
  this.full.set(width, height);
  let w = Math.max(1, Math.round(width / 4)), h = Math.max(1, Math.round(height / 4));
  for (let i = 0; i < this.levels.length; i++) { this.levels[i].setSize(w, h); if (i < this.ups.length) this.ups[i].setSize(w, h); w = Math.max(1, Math.round(w / 2)); h = Math.max(1, Math.round(h / 2)); }
 }
 render(renderer: THREE.WebGLRenderer, _w: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget) {
  const q = this.quad, n = this.levels.length;
  q.material = this.down0; this.down0.uniforms.tDiffuse.value = this.source; this.down0.uniforms.uTexel.value.set(1 / this.full.x, 1 / this.full.y); this.down0.uniforms.uThreshold.value = this.threshold;
  renderPostQuad(renderer,this.levels[0],q);
  q.material = this.down;
  for (let i = 1; i < n; i++) { const s = this.levels[i - 1]; this.down.uniforms.tDiffuse.value = s.texture; this.down.uniforms.uTexel.value.set(1 / s.width, 1 / s.height); renderPostQuad(renderer,this.levels[i],q); }
  // up: level i takes its own downsample plus the upsampled sum below; the radius spreads the glow toward the wide levels
  q.material = this.up; this.up.uniforms.uSpread.value = .6 + .8 * this.radius;
  for (let i = n - 2; i >= 0; i--) { const below = i === n - 2 ? this.levels[n - 1] : this.ups[i + 1]; this.up.uniforms.tDiffuse.value = below.texture; this.up.uniforms.tLevel.value = this.levels[i].texture; this.up.uniforms.uTexel.value.set(1 / below.width, 1 / below.height); renderPostQuad(renderer,this.ups[i],q); }
 }
 dispose() { [...this.levels, ...this.ups].forEach(t => t.dispose()); this.down0.dispose(); this.down.dispose(); this.up.dispose(); this.quad.dispose(); }
}

const FINISH = /* glsl */`uniform sampler2D tDiffuse;uniform sampler2D tBloom;uniform sampler2D tSun;uniform float uSun;uniform float uBloom,uContrast,uVibrance,uSaturation,uWarm,uVignette,uSharpen,uGrain,uTime,uToneExposure,uHighlightChroma;varying vec2 vUv;
 float luma(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
 // three.js's ACES filmic tone mapping, as its output pass applies it
 vec3 pfRRT(vec3 v){vec3 a=v*(v+.0245786)-.000090537;vec3 b=v*(.983729*v+.4329510)+.238081;return a/b;}
 vec3 pfACES(vec3 color){
  const mat3 I=mat3(vec3(.59719,.07600,.02840),vec3(.35458,.90834,.13383),vec3(.04823,.01566,.83777));
  const mat3 O=mat3(vec3(1.60475,-.10208,-.00327),vec3(-.53108,1.10813,-.07276),vec3(-.07367,-.00605,1.07602));
  color*=uToneExposure/.6;color=I*color;color=pfRRT(color);color=O*color;return clamp(color,0.,1.);}
 void main(){
  vec2 px=1./vec2(textureSize(tDiffuse,0));
  vec4 c=texture2D(tDiffuse,vUv);vec3 col=max(c.rgb,vec3(0.));
  // luminance sharpening (unsharp mask on a 4-tap cross, in log space so it is even across light and shade)
  float l0=luma(col),la=luma(texture2D(tDiffuse,vUv+vec2(px.x,0.)).rgb)+luma(texture2D(tDiffuse,vUv-vec2(px.x,0.)).rgb)+luma(texture2D(tDiffuse,vUv+vec2(0.,px.y)).rgb)+luma(texture2D(tDiffuse,vUv-vec2(0.,px.y)).rgb);
  float d=log2(l0+1e-3)-log2(la*.25+1e-3);col*=exp2(clamp(d*uSharpen,-.25,.25));
  // the sun's shafts and glare (sun-effects.ts), already exposed
  if(uSun>0.)col+=texture2D(tSun,vUv).rgb*uSun;
  col+=texture2D(tBloom,vUv).rgb*uBloom;
  vec3 lg=log2(col+vec3(1e-4));lg=(lg-log2(.18))*uContrast+log2(.18);col=exp2(lg)-vec3(1e-4);
  float lum=luma(col);float sat=max(max(col.r,col.g),col.b)-min(min(col.r,col.g),col.b);
  float boost=1.+uSaturation+uVibrance*(1.-clamp(sat/(lum+1e-3),0.,1.));col=max(mix(vec3(lum),col,boost),vec3(0.));
  float t=clamp(lum/(lum+.6),0.,1.);col*=mix(vec3(1.-.03*uWarm,1.,1.+.05*uWarm),vec3(1.+.035*uWarm,1.+.008*uWarm,1.-.05*uWarm),t);
  float dv=length((vUv-.5)*vec2(1.,.9));col*=1.-uVignette*smoothstep(.42,.95,dv);
  // grain: multiplicative, strongest in the mid-tones, a new pattern every frame
  float n=fract(sin(dot(gl_FragCoord.xy+fract(uTime*.37)*vec2(61.7,23.3),vec2(12.9898,78.233)))*43758.5453)-.5;
  col*=1.+n*uGrain*(1.-abs(t-.45));
  vec3 display=pfACES(col);
  // At night, keep the colour in bright narrowband luminaires as exposure
  // compresses their highlights. Neutral lights follow the normal filmic curve.
  float peak=max(col.r,max(col.g,col.b));
  float chroma=1.-min(col.r,min(col.g,col.b))/max(peak,1e-5);
  float preserve=uHighlightChroma*chroma*smoothstep(.45,2.5,peak);
  vec3 hue=col/max(peak,1e-5)*max(display.r,max(display.g,display.b));
  gl_FragColor=linearToOutputTexel(vec4(mix(display,hue,preserve),1.));
 }`;

class FinishPass extends Pass {
 sun?: Pass&{texture:THREE.Texture};
 private bloom: BloomPass;
 private source:THREE.Texture;
 material: THREE.ShaderMaterial; quad: FullScreenQuad;
 constructor(bloom: BloomPass, source:THREE.Texture) {
  super(); this.needsSwap = false;
  this.bloom=bloom;this.source=source;
  this.material = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, tBloom: {value: null}, tSun: {value: null}, uSun: {value: 0}, uBloom: {value: 0}, uContrast: {value: 1.12}, uVibrance: {value: .22}, uSaturation: {value: .04}, uWarm: {value: 1}, uVignette: {value: .22}, uSharpen: {value: .22}, uGrain: {value: .03}, uTime: {value: 0}, uToneExposure: {value: 1}, uHighlightChroma: {value: 0}},
   vertexShader: VERT, fragmentShader: FINISH, depthTest: false, depthWrite: false, toneMapped: false});
  this.quad = new FullScreenQuad(this.material);
 }
 get uniforms() { return this.material.uniforms; }
 render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget) {
  const u = this.material.uniforms;
  u.tDiffuse.value = this.source; u.tBloom.value = this.bloom.texture; u.uBloom.value = this.bloom.enabled ? this.bloom.strength : 0;
  u.tSun.value = this.sun?.texture ?? null; u.uSun.value = this.sun?.enabled ? 1 : 0;
  u.uTime.value = performance.now() / 1000; u.uToneExposure.value = renderer.toneMappingExposure;
  renderPostQuad(renderer,this.renderToScreen ? null : writeBuffer,this.quad);
 }
 dispose() { this.material.dispose(); this.quad.dispose(); }
}

// key: the metered log2 luminance that keeps the base exposure (a sunlit street view)
// Vibrance and extra saturation are off and the afternoon's warm/cool split is low: measured against 41 street
// photographs from registered cameras (work/areas/ring/colour_check.py), vibrance .26 + saturation .06 + warm 1.35 rendered
// walls 1.8x as saturated as the photographs (grey concrete with the right source colour at 2x); these values bring
// shaded walls to the photographs' saturation (1.0x) and sunlit ones to 1.47x. ACES stays: a neutral tone map (Khronos)
// raised sunlit walls to 2x, as did lowering the environment light; a neutral sun and lower contrast barely moved them.
// 10-15, over 150 area views (Palash: "overcorrecting on sun"): walls still rendered 1.35x the photographs' chroma and
// shade 20-25 L* too bright: afternoon saturation -.2, image fill (atmosphere env) 1.8 -> .8 and sun 3 -> 4.4 (less fill
// darkens lit walls too, so the sun carries what the fill gave them).
// 10-22 (Palash: "darker than it really is"): afternoon base exposure .9 -> 1.08 and fill .8 -> .95 (6 photo views: mean brightness 96.6 -> ~108, photos 108.6).
type Look = {contrast: number; vibrance: number; saturation: number; warm: number; vignette: number; bloom: number; threshold: number; key: number};
const LOOKS: Record<string, Look> = {
 afternoon: {contrast: 1.12, vibrance: 0, saturation: -.2, warm: .4, vignette: .2, bloom: .3, threshold: 1.1, key: -2.5},
 golden: {contrast: 1.08, vibrance: 0, saturation: 0, warm: 1.6, vignette: .24, bloom: .38, threshold: 1, key: -2.8},
 night: {contrast: 1.035, vibrance: .02, saturation: 0, warm: .18, vignette: .1, bloom: .17, threshold: 1.15, key: -4.8},
 overcast: {contrast: 1.08, vibrance: 0, saturation: 0, warm: .35, vignette: .18, bloom: .18, threshold: 1.3, key: -2.6},
};
/** Adds the exposure, occlusion composite, bloom and grade passes after the scene and occlusion passes. `scene` is the
 *  scene pass's resolved texture (its alpha the indirect share of each pixel's light), `ao` the occlusion pass whose
 *  result the composite applies; without them the passes work on the composer's own image. The last pass tone maps and
 *  draws to the screen (no output pass after it). */
export function createGrade(composer: EffectComposer, scene?: THREE.Texture, ao?: GTAOPass, resolved?:{value:THREE.Texture|null;sampleOffset?:THREE.Vector2}, sceneDepth?:THREE.Texture, sun?:(exposure:()=>THREE.Texture)=>Pass&{texture:THREE.Texture}) {
 const exposure = new ExposurePass(scene);
 const composite = new CompositePass(scene, ao, exposure,resolved);
 const motion = motionControl(composite,ao,sceneDepth);
 const bloom = new BloomPass(composite.output.texture);
 const finish = new FinishPass(bloom,composite.output.texture);
 // the sun's shafts and glare are drawn once the exposure is known; the last pass adds them
 const sunPass = sun?.(() => exposure.texture);
 composer.addPass(exposure); composer.addPass(composite); if (sunPass) composer.addPass(sunPass); composer.addPass(bloom); composer.addPass(finish);
 if (sunPass) finish.sun = sunPass;
 // the glow's strength: the dual-filter chain sums about five levels where the old one weighted five mips by ~0.6
 const BLOOM_SCALE = .62;
 function preset(name: string) {
  motion.reset();
  const l = LOOKS[name] || LOOKS.afternoon; const u = finish.uniforms;
  u.uContrast.value = l.contrast; u.uVibrance.value = l.vibrance; u.uSaturation.value = l.saturation; u.uWarm.value = l.warm; u.uVignette.value = l.vignette;
  const night = name === 'night';
  bloom.down0.uniforms.uScatterLimit.value = night ? 18 : 4;
  u.uHighlightChroma.value = night ? .52 : 0;
  const meter = exposure.adaptMat.uniforms;
  meter.uMin.value = night ? .8 : .75;
  meter.uMax.value = night ? 1.38 : 1.8; meter.uComp.value = night ? .4 : .5;   // daytime shade adapts further (see ADAPT); night keeps its calibration
  bloom.strength = l.bloom * BLOOM_SCALE; bloom.threshold = l.threshold; exposure.adaptMat.uniforms.uKey.value = l.key; exposure.snapNext = true;
 }
 preset('afternoon');
 return {bloom, sun: sunPass, grade: finish, exposure, composite, preset, motion:motion.update, resetMotion:motion.reset, snap: () => { exposure.snapNext = true; motion.reset(); }, dispose() { exposure.dispose(); composite.dispose(); bloom.dispose(); finish.dispose(); }};
}
