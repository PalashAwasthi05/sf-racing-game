import * as THREE from 'three';
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
 void main(){vec3 c=texture2D(tDiffuse,vUv).rgb;float l=dot(max(c,vec3(0.)),vec3(.2126,.7152,.0722));
  // centre-weighted, and the sky's brightness counts only up to a sunlit wall's: a frame of sky must not darken the street
  vec2 d=(vUv-.5)*vec2(1.,.85);float w=exp(-dot(d,d)*3.5);float lg=log2(clamp(l,1e-4,.6));
  gl_FragColor=vec4(lg*w,w,0.,1.);}`;
// Partial adaptation, as an eye's: a view metered k stops darker than the key gets 0.4 k stops more exposure (at most
// 1.35x), so shade still reads darker than sun and the sky over a shaded street keeps its colour (at 1.7x it washed white).
const ADAPT = `uniform sampler2D tLum;uniform sampler2D tPrev;uniform float uKey,uComp,uMin,uMax,uRate;varying vec2 vUv;
 void main(){vec4 m=textureLod(tLum,vec2(.5),12.);float avg=m.x/max(m.y,1e-6);float target=clamp(exp2((uKey-avg)*uComp),uMin,uMax);
  float prev=texture2D(tPrev,vec2(.5)).r;float e=prev<=0.?target:mix(prev,target,uRate);gl_FragColor=vec4(e,avg,target,1.);}`;

/** Eye adaptation: meters the HDR scene on the GPU and eases an exposure toward a fixed key; the exposure stays on the GPU
 *  (a 1x1 texture) for the composite to apply. Draws nothing full-screen. */
class ExposurePass extends Pass {
 lum = new THREE.WebGLRenderTarget(256, 128, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
 adapt = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false}));
 cur = 0; snapNext = true; speed = 1.6;
 meter = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}}, vertexShader: VERT, fragmentShader: METER, depthTest: false, depthWrite: false});
 adaptMat = new THREE.ShaderMaterial({uniforms: {tLum: {value: null}, tPrev: {value: null}, uKey: {value: -2.5}, uComp: {value: .4}, uMin: {value: .75}, uMax: {value: 1.35}, uRate: {value: 1}}, vertexShader: VERT, fragmentShader: ADAPT, depthTest: false, depthWrite: false});
 quad = new FullScreenQuad();
 /** `source`: the image to meter; without one, the composer's current image */
 constructor(private source?: THREE.Texture) { super(); this.needsSwap = false; }
 get texture() { return this.adapt[this.cur].texture; }
 render(renderer: THREE.WebGLRenderer, _w: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget, deltaTime: number) {
  const next = 1 - this.cur;
  this.meter.uniforms.tDiffuse.value = this.source ?? readBuffer.texture; this.quad.material = this.meter; renderer.setRenderTarget(this.lum); this.quad.render(renderer);
  const u = this.adaptMat.uniforms; u.tLum.value = this.lum.texture; u.tPrev.value = this.adapt[this.cur].texture;
  u.uRate.value = this.snapNext ? 1 : 1 - Math.exp(-Math.max(0, Math.min(deltaTime || 0, .25)) * this.speed); this.snapNext = false;
  this.quad.material = this.adaptMat; renderer.setRenderTarget(this.adapt[next]); this.quad.render(renderer);
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
class CompositePass extends Pass {
 material: THREE.ShaderMaterial; quad: FullScreenQuad;
 constructor(private scene: THREE.Texture | undefined, private ao: GTAOPass | undefined, private exposure: ExposurePass) {
  super();
  this.material = new THREE.ShaderMaterial({
   uniforms: {tScene: {value: scene}, tAO: {value: null}, tExp: {value: null}, uStrength: {value: 1}},
   vertexShader: VERT,
   fragmentShader: /* glsl */`uniform sampler2D tScene;uniform sampler2D tAO;uniform sampler2D tExp;uniform float uStrength;varying vec2 vUv;
    void main(){vec4 c=texture2D(tScene,vUv);float ao=texture2D(tAO,vUv).r;c.rgb*=mix(1.,ao,clamp(c.a,0.,1.)*uStrength)*texture2D(tExp,vec2(.5)).r;
     // one NaN or infinite pixel here would spread through the bloom's mip chain into a block
     if(!(c.r==c.r&&c.g==c.g&&c.b==c.b))c.rgb=vec3(0.);gl_FragColor=vec4(min(c.rgb,vec3(6e4)),1.);}`,
   depthTest: false, depthWrite: false});
  this.quad = new FullScreenQuad(this.material);
 }
 render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget) {
  const u = this.material.uniforms;
  u.tScene.value = this.scene ?? readBuffer.texture;
  u.tAO.value = this.ao ? (this.ao as unknown as {pdRenderTarget: THREE.WebGLRenderTarget}).pdRenderTarget.texture : WHITE; u.uStrength.value = this.ao?.enabled ? 1 : 0; u.tExp.value = this.exposure.texture;
  renderer.setRenderTarget(writeBuffer); this.quad.render(renderer);
 }
 dispose() { this.material.dispose(); this.quad.dispose(); }
}

// The bloom: a soft-thresholded, firefly-suppressed downsample to a quarter of the resolution, four more halvings, then
// back up, each level adding a tent-filtered upsample of the one below to its own (the dual-filter chain of mobile
// renderers): a wide, smooth glow from about a dozen small passes.
const DOWN0 = /* glsl */`uniform sampler2D tDiffuse;uniform vec2 uTexel;uniform float uThreshold;varying vec2 vUv;
 vec3 tap(vec2 o){vec3 c=min(texture2D(tDiffuse,vUv+o*uTexel).rgb,vec3(6e4));return c/(1.+max(max(c.r,c.g),c.b)*.25);}
 void main(){
  // four bilinear taps cover the 4x4 source texels of this quarter-resolution texel; each is weighted down by its own
  // brightness (a lone bright texel cannot flash a whole bloom tile on and off as it moves)
  vec3 c=(tap(vec2(-1.,-1.))+tap(vec2(1.,-1.))+tap(vec2(-1.,1.))+tap(vec2(1.,1.)))*.25;c/=max(1.-max(max(c.r,c.g),c.b)*.25,1e-3);
  float b=dot(c,vec3(.2126,.7152,.0722)),k=uThreshold*.5,s=clamp(b-uThreshold+k,0.,2.*k);s=s*s/(4.*k+1e-4);
  gl_FragColor=vec4(c*max(s,b-uThreshold)/max(b,1e-4),1.);}`;
const DOWN = /* glsl */`uniform sampler2D tDiffuse;uniform vec2 uTexel;varying vec2 vUv;
 void main(){vec2 h=uTexel*.5;vec3 c=texture2D(tDiffuse,vUv).rgb*4.+texture2D(tDiffuse,vUv-h).rgb+texture2D(tDiffuse,vUv+h).rgb+texture2D(tDiffuse,vUv+vec2(h.x,-h.y)).rgb+texture2D(tDiffuse,vUv-vec2(h.x,-h.y)).rgb;gl_FragColor=vec4(c*.125,1.);}`;
const UP = /* glsl */`uniform sampler2D tDiffuse;uniform sampler2D tLevel;uniform vec2 uTexel;uniform float uSpread;varying vec2 vUv;
 void main(){vec2 h=uTexel*.5;
  vec3 c=texture2D(tDiffuse,vUv+vec2(-h.x*2.,0.)).rgb+texture2D(tDiffuse,vUv+vec2(-h.x,h.y)).rgb*2.+texture2D(tDiffuse,vUv+vec2(0.,h.y*2.)).rgb+texture2D(tDiffuse,vUv+vec2(h.x,h.y)).rgb*2.
   +texture2D(tDiffuse,vUv+vec2(h.x*2.,0.)).rgb+texture2D(tDiffuse,vUv+vec2(h.x,-h.y)).rgb*2.+texture2D(tDiffuse,vUv+vec2(0.,-h.y*2.)).rgb+texture2D(tDiffuse,vUv+vec2(-h.x,-h.y)).rgb*2.;
  gl_FragColor=vec4(texture2D(tLevel,vUv).rgb+c/12.*uSpread,1.);}`;

class BloomPass extends Pass {
 levels: THREE.WebGLRenderTarget[] = []; ups: THREE.WebGLRenderTarget[] = [];
 down0 = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, uTexel: {value: new THREE.Vector2()}, uThreshold: {value: 1.1}}, vertexShader: VERT, fragmentShader: DOWN0, depthTest: false, depthWrite: false});
 down = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, uTexel: {value: new THREE.Vector2()}}, vertexShader: VERT, fragmentShader: DOWN, depthTest: false, depthWrite: false});
 up = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, tLevel: {value: null}, uTexel: {value: new THREE.Vector2()}, uSpread: {value: 1}}, vertexShader: VERT, fragmentShader: UP, depthTest: false, depthWrite: false});
 quad = new FullScreenQuad(); strength = .3; threshold = 1.1; radius = .45; full = new THREE.Vector2(1, 1);
 constructor(count = 5) {
  super(); this.needsSwap = false;
  const rt = () => new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
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
  q.material = this.down0; this.down0.uniforms.tDiffuse.value = readBuffer.texture; this.down0.uniforms.uTexel.value.set(1 / this.full.x, 1 / this.full.y); this.down0.uniforms.uThreshold.value = this.threshold;
  renderer.setRenderTarget(this.levels[0]); q.render(renderer);
  q.material = this.down;
  for (let i = 1; i < n; i++) { const s = this.levels[i - 1]; this.down.uniforms.tDiffuse.value = s.texture; this.down.uniforms.uTexel.value.set(1 / s.width, 1 / s.height); renderer.setRenderTarget(this.levels[i]); q.render(renderer); }
  // up: level i takes its own downsample plus the upsampled sum below; the radius spreads the glow toward the wide levels
  q.material = this.up; this.up.uniforms.uSpread.value = .6 + .8 * this.radius;
  for (let i = n - 2; i >= 0; i--) { const below = i === n - 2 ? this.levels[n - 1] : this.ups[i + 1]; this.up.uniforms.tDiffuse.value = below.texture; this.up.uniforms.tLevel.value = this.levels[i].texture; this.up.uniforms.uTexel.value.set(1 / below.width, 1 / below.height); renderer.setRenderTarget(this.ups[i]); q.render(renderer); }
 }
 dispose() { [...this.levels, ...this.ups].forEach(t => t.dispose()); this.down0.dispose(); this.down.dispose(); this.up.dispose(); this.quad.dispose(); }
}

const FINISH = /* glsl */`uniform sampler2D tDiffuse;uniform sampler2D tBloom;uniform float uBloom,uContrast,uVibrance,uSaturation,uWarm,uVignette,uSharpen,uGrain,uTime,uToneExposure;varying vec2 vUv;
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
  col+=texture2D(tBloom,vUv).rgb*uBloom;
  vec3 lg=log2(col+vec3(1e-4));lg=(lg-log2(.18))*uContrast+log2(.18);col=exp2(lg)-vec3(1e-4);
  float lum=luma(col);float sat=max(max(col.r,col.g),col.b)-min(min(col.r,col.g),col.b);
  float boost=1.+uSaturation+uVibrance*(1.-clamp(sat/(lum+1e-3),0.,1.));col=max(mix(vec3(lum),col,boost),vec3(0.));
  float t=clamp(lum/(lum+.6),0.,1.);col*=mix(vec3(1.-.03*uWarm,1.,1.+.05*uWarm),vec3(1.+.035*uWarm,1.+.008*uWarm,1.-.05*uWarm),t);
  float dv=length((vUv-.5)*vec2(1.,.9));col*=1.-uVignette*smoothstep(.42,.95,dv);
  // grain: multiplicative, strongest in the mid-tones, a new pattern every frame
  float n=fract(sin(dot(gl_FragCoord.xy+fract(uTime*.37)*vec2(61.7,23.3),vec2(12.9898,78.233)))*43758.5453)-.5;
  col*=1.+n*uGrain*(1.-abs(t-.45));
  gl_FragColor=linearToOutputTexel(vec4(pfACES(col),1.));
 }`;

class FinishPass extends Pass {
 material: THREE.ShaderMaterial; quad: FullScreenQuad;
 constructor(private bloom: BloomPass) {
  super(); this.needsSwap = false;
  this.material = new THREE.ShaderMaterial({uniforms: {tDiffuse: {value: null}, tBloom: {value: null}, uBloom: {value: 0}, uContrast: {value: 1.12}, uVibrance: {value: .22}, uSaturation: {value: .04}, uWarm: {value: 1}, uVignette: {value: .22}, uSharpen: {value: .22}, uGrain: {value: .03}, uTime: {value: 0}, uToneExposure: {value: 1}},
   vertexShader: VERT, fragmentShader: FINISH, depthTest: false, depthWrite: false, toneMapped: false});
  this.quad = new FullScreenQuad(this.material);
 }
 get uniforms() { return this.material.uniforms; }
 render(renderer: THREE.WebGLRenderer, writeBuffer: THREE.WebGLRenderTarget, readBuffer: THREE.WebGLRenderTarget) {
  const u = this.material.uniforms;
  u.tDiffuse.value = readBuffer.texture; u.tBloom.value = this.bloom.texture; u.uBloom.value = this.bloom.enabled ? this.bloom.strength : 0;
  u.uTime.value = performance.now() / 1000; u.uToneExposure.value = renderer.toneMappingExposure;
  renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer); this.quad.render(renderer);
 }
 dispose() { this.material.dispose(); this.quad.dispose(); }
}

// key: the metered log2 luminance that keeps the base exposure (a sunlit street view)
// Vibrance and extra saturation are off and the afternoon's warm/cool split is low: measured against 41 street
// photographs from registered cameras (work/areas/ring/colour_check.py), vibrance .26 + saturation .06 + warm 1.35 rendered
// walls 1.8x as saturated as the photographs (grey concrete with the right source colour at 2x); these values bring
// shaded walls to the photographs' saturation (1.0x) and sunlit ones to 1.47x. ACES stays: a neutral tone map (Khronos)
// raised sunlit walls to 2x, as did lowering the environment light; a neutral sun and lower contrast barely moved them.
type Look = {contrast: number; vibrance: number; saturation: number; warm: number; vignette: number; bloom: number; threshold: number; key: number};
const LOOKS: Record<string, Look> = {
 afternoon: {contrast: 1.12, vibrance: 0, saturation: 0, warm: .4, vignette: .2, bloom: .3, threshold: 1.1, key: -2.5},
 golden: {contrast: 1.08, vibrance: 0, saturation: 0, warm: 1.6, vignette: .24, bloom: .38, threshold: 1, key: -2.8},
 overcast: {contrast: 1.08, vibrance: 0, saturation: 0, warm: .35, vignette: .18, bloom: .18, threshold: 1.3, key: -2.6},
};
/** Adds the exposure, occlusion composite, bloom and grade passes after the scene and occlusion passes. `scene` is the
 *  scene pass's resolved texture (its alpha the indirect share of each pixel's light), `ao` the occlusion pass whose
 *  result the composite applies; without them the passes work on the composer's own image. The last pass tone maps and
 *  draws to the screen (no output pass after it). */
export function createGrade(composer: EffectComposer, scene?: THREE.Texture, ao?: GTAOPass) {
 const exposure = new ExposurePass(scene);
 const composite = new CompositePass(scene, ao, exposure);
 const bloom = new BloomPass();
 const finish = new FinishPass(bloom);
 composer.addPass(exposure); composer.addPass(composite); composer.addPass(bloom); composer.addPass(finish);
 // the glow's strength: the dual-filter chain sums about five levels where the old one weighted five mips by ~0.6
 const BLOOM_SCALE = .62;
 function preset(name: string) {
  const l = LOOKS[name] || LOOKS.afternoon; const u = finish.uniforms;
  u.uContrast.value = l.contrast; u.uVibrance.value = l.vibrance; u.uSaturation.value = l.saturation; u.uWarm.value = l.warm; u.uVignette.value = l.vignette;
  bloom.strength = l.bloom * BLOOM_SCALE; bloom.threshold = l.threshold; exposure.adaptMat.uniforms.uKey.value = l.key; exposure.snapNext = true;
 }
 preset('afternoon');
 return {bloom, grade: finish, exposure, composite, preset, snap: () => { exposure.snapNext = true; }, dispose() { exposure.dispose(); composite.dispose(); bloom.dispose(); finish.dispose(); }};
}
