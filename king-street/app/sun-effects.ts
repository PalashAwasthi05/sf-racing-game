import * as THREE from 'three';
import {Pass, FullScreenQuad} from 'three/addons/postprocessing/Pass.js';
import {renderPostQuad} from './postprocessing-utils.ts';

// What the sun does to the air and to the lens, which no surface shader draws:
//  shafts  the haze between the eye and each pixel is lit by the sun except where a building or a tree shadows it. Every
//          pixel's view ray is marched through the sun's own cascaded shadow maps (no extra scene render), so beams
//          stand between buildings and under canopies and shadowed air stays clear. Scattering is strongly forward
//          (a two-lobe Henyey-Greenstein phase): looking toward the sun the street fills with light, looking away the
//          air is nearly clear, as haze behaves.
//  glare   the solar disc seen directly: a wide veil, a tight corona, diffraction spikes and faint lens ghosts, all
//          scaled by the share of the disc not hidden behind geometry (found from the depth buffer, eased over time so
//          the sun passing behind a tree flickers softly, not in single frames).
// Marched and blurred at half resolution; the grade adds the result before bloom and tone mapping.
const VERT = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}';
const STEPS = 24;
const MARCH = /* glsl */`
#include <packing>
uniform sampler2D tDepth,tShadowNear,tShadowFar,tExp;
uniform mat4 uClipToWorld,uShadowNear,uShadowFar;
uniform vec3 uCam,uForward,uSunDir,uSunColor;
uniform float uSplit,uReach,uDensity,uHeight,uEnabled;
varying vec2 vUv;
float lit(sampler2D map,mat4 m,vec3 p){
 vec4 c=m*vec4(p,1.);c.xyz/=c.w;
 if(c.x<=0.||c.y<=0.||c.x>=1.||c.y>=1.||c.z>=1.)return 1.;
 return step(c.z-.0015,unpackRGBAToDepth(textureLod(map,c.xy,0.)));
}
float hg(float c,float g){float k=1.+g*g-2.*g*c;return (1.-g*g)/(12.566371*k*sqrt(k));}
void main(){
 float depth=textureLod(tDepth,vUv,0.).r;
 vec4 w=uClipToWorld*vec4(vUv*2.-1.,depth*2.-1.,1.);vec3 ray=w.xyz/w.w-uCam;float dist=length(ray);vec3 dir=ray/max(dist,1e-4);
 bool sky=depth>=.999999;float reach=sky?uReach:min(dist,uReach);
 if(uEnabled<.5){gl_FragColor=vec4(0.,0.,0.,sky?1.:clamp(dist/uReach,0.,1.));return;}
 float jitter=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))));
 float along=max(dot(dir,uForward),1e-3),sum=0.,dt=reach/${STEPS}.;
 for(int i=0;i<${STEPS};i++){
  float t=(float(i)+jitter)*dt;vec3 p=uCam+dir*t;
  float l=t*along<uSplit?lit(tShadowNear,uShadowNear,p):lit(tShadowFar,uShadowFar,p);
  // the haze thins with height above the eye's street, and what is scattered far away is itself dimmed on the way back
  sum+=l*exp(-max(p.y-uCam.y,0.)/uHeight)*exp(-t*uDensity*.6);
 }
 float c=dot(dir,uSunDir);
 float phase=.6*hg(c,.78)+.4*hg(c,.1);
 float exposure=texture2D(tExp,vec2(.5)).r;if(!(exposure>0.)||isinf(exposure))exposure=1.;
 vec3 scattered=uSunColor*(phase*uDensity*sum*dt*exposure);
 gl_FragColor=vec4(min(scattered,vec3(40.)),sky?1.:clamp(dist/uReach,0.,1.));
}`;
// Separable 9-tap blur that does not cross depth edges (alpha holds the normalised distance): the march's per-pixel
// jitter becomes smooth beams, and a beam behind a near wall does not bleed onto the wall.
const BLUR = /* glsl */`
uniform sampler2D tDiffuse;uniform vec2 uStep;varying vec2 vUv;
void main(){
 vec4 c=texture2D(tDiffuse,vUv);vec3 sum=c.rgb*.2270;float weight=.2270;
 float W[4];W[0]=.1945;W[1]=.1216;W[2]=.0541;W[3]=.0162;
 for(int i=0;i<4;i++){for(int s=-1;s<=1;s+=2){
  vec4 o=texture2D(tDiffuse,vUv+uStep*float((i+1)*s));
  float k=W[i]*(1.-smoothstep(.02,.08,abs(o.a-c.a)/max(c.a,.02)));sum+=o.rgb*k;weight+=k;}}
 gl_FragColor=vec4(sum/weight,c.a);
}`;
// The share of the solar disc with nothing in front of it: 1x1, eased toward this frame's reading.
const SEEN = /* glsl */`
#include <packing>
uniform sampler2D tDepth,tPrev,tShadowNear;uniform mat4 uShadowNear;uniform vec3 uEye;uniform vec2 uSun,uAspect;uniform float uFront,uRate,uShadowed;varying vec2 vUv;
void main(){
 // a sun outside the frame has no depth to test: there it counts as open when the eye itself stands in sunlight
 float eye=1.;
 if(uShadowed>.5){vec4 c=uShadowNear*vec4(uEye,1.);c.xyz/=c.w;if(c.x>0.&&c.y>0.&&c.x<1.&&c.y<1.&&c.z<1.)eye=step(c.z-.0015,unpackRGBAToDepth(textureLod(tShadowNear,c.xy,0.)));}
 float open=0.;
 for(int i=0;i<32;i++){float a=float(i)*2.3999632,r=sqrt((float(i)+.5)/32.)*.012;vec2 uv=uSun+vec2(cos(a),sin(a))*r*uAspect;
  open+=uv.x>0.&&uv.y>0.&&uv.x<1.&&uv.y<1.?step(.999999,textureLod(tDepth,uv,0.).r):eye;}
 vec2 out2=max(max(-uSun,uSun-1.),vec2(0.))/uAspect;                // how far outside the frame, in frame heights
 float now=uFront*open/32.*(1.-smoothstep(.1,1.5,length(out2))),prev=texture2D(tPrev,vec2(.5)).r;if(!(prev>=0.&&prev<=1.))prev=now;
 gl_FragColor=vec4(mix(prev,now,uRate),0.,0.,1.);
}`;
const GLARE = /* glsl */`
uniform sampler2D tDiffuse,tSeen,tExp;uniform vec2 uSun,uAspect;uniform vec3 uSunColor;uniform float uGlare,uFront;varying vec2 vUv;
float ring(vec2 p,vec2 at,float radius,float width){float d=length(p-at);return smoothstep(radius,radius-width,d);}
void main(){
 vec3 col=texture2D(tDiffuse,vUv).rgb;
 float seen=texture2D(tSeen,vec2(.5)).r*uGlare;
 if(seen>1e-4){
  float exposure=texture2D(tExp,vec2(.5)).r;if(!(exposure>0.)||isinf(exposure))exposure=1.;
  // screen positions in units of the frame's height, so the flare keeps its shape at any aspect
  vec2 p=(vUv-.5)/uAspect,s=(uSun-.5)/uAspect,d=p-s;float r=length(d);
  float veil=.03/(1.+r*r*4.);                                   // light scattered inside the lens, over the whole frame
  float corona=.9*exp(-r*38.)+.22*exp(-r*9.)+.1*exp(-r*2.6);
  float a=atan(d.y,d.x);
  float spikes=(pow(abs(cos(a*3.)),48.)+.6*pow(abs(cos(a*3.+.52)),90.))*.5*exp(-r*5.5);
  float streak=.35*exp(-abs(d.y)*150.)*exp(-abs(d.x)*3.2);        // the horizontal smear of a wide lens
  vec3 light=uSunColor*(veil+corona+spikes+streak);
  // ghosts: reflections between lens elements lie on the line through the frame's centre, opposite the sun
  vec3 ghosts=vec3(.05,.09,.07)*ring(p,-s*.42,.035,.012)+vec3(.09,.06,.04)*ring(p,-s*.9,.055,.03)+vec3(.04,.05,.09)*ring(p,s*.35,.02,.01)+vec3(.05,.07,.03)*ring(p,-s*1.45,.08,.05);
  col+=(light+ghosts*.22*dot(uSunColor,vec3(.333)))*seen*exposure;
 }
 gl_FragColor=vec4(col,1.);
}`;

export type SunLook = {shafts: number; glare: number};
// shafts: the haze the beams are drawn in, relative to a clear afternoon; glare: the strength of the lens effects
export const SUN_LOOKS: Record<string, SunLook> = {
 afternoon: {shafts: 1, glare: 1},
 golden: {shafts: 1.5, glare: 1.2},
 overcast: {shafts: 0, glare: 0},
 night: {shafts: 0, glare: 0},
};
type Cascade = {shadow: THREE.DirectionalLightShadow; color: THREE.Color; intensity: number};

/** A pass that produces `texture` (half resolution, in exposed scene light): light shafts plus the sun's glare. */
export class SunEffectsPass extends Pass {
 private camera: THREE.PerspectiveCamera;
 private lights: Cascade[];
 private sunDirection: () => THREE.Vector3;
 private target = () => new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
 private a = this.target(); private b = this.target();
 private seen = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false}));
 private cur = 0; private quad = new FullScreenQuad();
 private march: THREE.ShaderMaterial; private blur: THREE.ShaderMaterial; private seenMat: THREE.ShaderMaterial; private glare: THREE.ShaderMaterial;
 private projectionInverse: THREE.Matrix4; private exposure: () => THREE.Texture;
 private clip = new THREE.Matrix4(); private sunClip = new THREE.Vector4(); private dir = new THREE.Vector3();
 look: SunLook = {shafts: 1, glare: 1};
 /** The frame budget may drop the light shafts (the ray march and its blur); the glare stays. */
 shafts = true; private blank = false;
 /** `depth`: the scene's depth; `projectionInverse`: the inverse of the projection that depth was drawn with;
  *  `exposure`: the eye adaptation's 1x1 texture; `split`: where the near shadow cascade hands over (m). */
 constructor(camera: THREE.PerspectiveCamera, depth: THREE.Texture, projectionInverse: THREE.Matrix4, exposure: () => THREE.Texture, lights: Cascade[], sunDirection: () => THREE.Vector3, split: number) {
  super(); this.needsSwap = false; this.camera = camera; this.lights = lights; this.sunDirection = sunDirection; this.projectionInverse = projectionInverse; this.exposure = exposure;
  const mat = (fragmentShader: string, uniforms: Record<string, THREE.IUniform>) => new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader, uniforms, depthTest: false, depthWrite: false, toneMapped: false});
  this.march = mat(MARCH, {tDepth: {value: depth}, tShadowNear: {value: null}, tShadowFar: {value: null}, tExp: {value: null}, uClipToWorld: {value: new THREE.Matrix4()}, uShadowNear: {value: new THREE.Matrix4()}, uShadowFar: {value: new THREE.Matrix4()},
   uCam: {value: new THREE.Vector3()}, uForward: {value: new THREE.Vector3()}, uSunDir: {value: new THREE.Vector3()}, uSunColor: {value: new THREE.Color()}, uSplit: {value: split}, uReach: {value: 240}, uDensity: {value: .002}, uHeight: {value: 70}, uEnabled: {value: 1}});
  this.blur = mat(BLUR, {tDiffuse: {value: null}, uStep: {value: new THREE.Vector2()}});
  this.seenMat = mat(SEEN, {tDepth: {value: depth}, tPrev: {value: null}, tShadowNear: {value: null}, uShadowNear: {value: new THREE.Matrix4()}, uEye: {value: new THREE.Vector3()}, uShadowed: {value: 0}, uSun: {value: new THREE.Vector2()}, uAspect: {value: new THREE.Vector2(1, 1)}, uFront: {value: 0}, uRate: {value: 1}});
  this.glare = mat(GLARE, {tDiffuse: {value: null}, tSeen: {value: null}, tExp: {value: null}, uSun: {value: new THREE.Vector2()}, uAspect: {value: new THREE.Vector2(1, 1)}, uSunColor: {value: new THREE.Color()}, uGlare: {value: 1}, uFront: {value: 0}});
 }
 get texture() { return this.b.texture; }
 setSize(width: number, height: number) { this.blank = false; const w = Math.max(1, Math.round(width / 2)), h = Math.max(1, Math.round(height / 2)); this.a.setSize(w, h); this.b.setSize(w, h); }
 render(renderer: THREE.WebGLRenderer, _w: THREE.WebGLRenderTarget, _r: THREE.WebGLRenderTarget, deltaTime: number) {
  const camera = this.camera, [near, far] = this.lights, sun = this.sunDirection(), m = this.march.uniforms, exposure = this.exposure();
  const maps = near.shadow.map && far.shadow.map, on = !!maps && this.shafts && this.look.shafts > 0 && sun.y > .02;
  m.uEnabled.value = on ? 1 : 0;
  if (on) {
   m.tShadowNear.value = near.shadow.map!.texture; m.tShadowFar.value = far.shadow.map!.texture;
   (m.uShadowNear.value as THREE.Matrix4).copy(near.shadow.matrix); (m.uShadowFar.value as THREE.Matrix4).copy(far.shadow.matrix);
  }
  (m.uClipToWorld.value as THREE.Matrix4).multiplyMatrices(camera.matrixWorld, this.projectionInverse);
  (m.uCam.value as THREE.Vector3).setFromMatrixPosition(camera.matrixWorld); camera.getWorldDirection(m.uForward.value as THREE.Vector3);
  (m.uSunDir.value as THREE.Vector3).copy(sun); (m.uSunColor.value as THREE.Color).copy(near.color).multiplyScalar(near.intensity * this.look.shafts);
  m.tExp.value = exposure;
  const b = this.blur.uniforms;
  if (on || !this.blank) {   // once the shafts are off and their target is black, skip the march and both blurs
  this.blank = !on;
  this.quad.material = this.march; renderPostQuad(renderer, this.a, this.quad);
  b.tDiffuse.value = this.a.texture; (b.uStep.value as THREE.Vector2).set(1.5 / this.a.width, 0); this.quad.material = this.blur; renderPostQuad(renderer, this.b, this.quad);
  b.tDiffuse.value = this.b.texture; (b.uStep.value as THREE.Vector2).set(0, 1.5 / this.a.height); renderPostQuad(renderer, this.a, this.quad);
  }
  // where the sun is in the frame, and how much of its disc is open
  this.clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  this.sunClip.set(sun.x, sun.y, sun.z, 0).applyMatrix4(this.clip);
  const front = this.sunClip.w > 1e-4 ? 1 : 0, sx = front ? this.sunClip.x / this.sunClip.w * .5 + .5 : -9, sy = front ? this.sunClip.y / this.sunClip.w * .5 + .5 : -9;
  const aspect = this.a.height / Math.max(this.a.width, 1), next = 1 - this.cur, s = this.seenMat.uniforms, g = this.glare.uniforms;
  (s.uSun.value as THREE.Vector2).set(sx, sy); (s.uAspect.value as THREE.Vector2).set(aspect, 1); s.uShadowed.value = maps ? 1 : 0; if (maps) { s.tShadowNear.value = near.shadow.map!.texture; (s.uShadowNear.value as THREE.Matrix4).copy(near.shadow.matrix); (s.uEye.value as THREE.Vector3).copy(m.uCam.value as THREE.Vector3).addScaledVector(m.uForward.value as THREE.Vector3, 3); }
  s.uFront.value = front; s.tPrev.value = this.seen[this.cur].texture;
  s.uRate.value = deltaTime > 0 ? 1 - Math.exp(-Math.min(deltaTime, .25) * 14) : 1;
  this.quad.material = this.seenMat; renderPostQuad(renderer, this.seen[next], this.quad); this.cur = next;
  g.tDiffuse.value = this.a.texture; g.tSeen.value = this.seen[next].texture; g.tExp.value = exposure; (g.uSun.value as THREE.Vector2).set(sx, sy); (g.uAspect.value as THREE.Vector2).set(aspect, 1);
  (g.uSunColor.value as THREE.Color).copy(near.color).multiplyScalar(near.intensity); g.uGlare.value = sun.y > 0 ? this.look.glare : 0;
  this.quad.material = this.glare; renderPostQuad(renderer, this.b, this.quad);
 }
 dispose() { this.a.dispose(); this.b.dispose(); this.seen.forEach(t => t.dispose()); this.march.dispose(); this.blur.dispose(); this.seenMat.dispose(); this.glare.dispose(); this.quad.dispose(); }
}
