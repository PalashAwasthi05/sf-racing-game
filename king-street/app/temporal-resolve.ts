import * as THREE from 'three';
import {renderPostQuad} from './postprocessing-utils.ts';
import {Pass, FullScreenQuad} from 'three/addons/postprocessing/Pass.js';

// RGB is scene-linear radiance, alpha is metric view depth. Keep this before
// exposure, bloom and shutter blur so none of those effects feeds back.
const RESOLVE = /* glsl */`
uniform sampler2D currentColor, currentDepth, lastFrame;
uniform mat4 previousFromClip, previousStableClip, localFromClip, previousBodyFromLocal, worldFromClip;
uniform vec3 boxMin, boxMax;
uniform vec3 sourceEye;
uniform vec2 sourceOffset;
uniform vec2 clipRange;
uniform float validHistory, movingBody, staticFrame;
varying vec2 uvScreen;
bool validHistoryColor(vec3 c){return all(greaterThanEqual(c,vec3(0.))) && all(lessThan(c,vec3(65504.)));}
vec3 compressHDR(vec3 c){return c/(1.+dot(c,vec3(.2126,.7152,.0722)));}
vec3 expandHDR(vec3 c){return c/max(.0001,1.-dot(c,vec3(.2126,.7152,.0722)));}
void main(){
 vec2 px=1./vec2(textureSize(currentColor,0));
 // With a fixed camera and body, the eight raster jitters are already samples
 // of the same display pixel's footprint. Accumulate those samples directly.
 // Undoing each jitter with bilinear filtering first convolves every stationary
 // edge with a tent; reprojecting that filtered result compounds the softness.
 if(staticFrame>.5){
   ivec2 pixel=ivec2(gl_FragCoord.xy);
   vec3 sampleColor=max(texelFetch(currentColor,pixel,0).rgb,vec3(0.));
   float sampleDepth=texelFetch(currentDepth,pixel,0).r;
   float sampleZ=clipRange.x*clipRange.y/(clipRange.y*(1.-sampleDepth)+clipRange.x*sampleDepth);
   vec3 history=texelFetch(lastFrame,pixel,0).rgb;
   if(validHistory>.5&&validHistoryColor(history))
     sampleColor=expandHDR(mix(compressHDR(sampleColor),compressHDR(history),.875));
   gl_FragColor=vec4(sampleColor,sampleZ);return;
 }
 vec2 sourceUv=clamp(uvScreen+sourceOffset,px*.5,1.-px*.5);
 vec3 center=max(texture2D(currentColor,sourceUv).rgb,vec3(0.));
 float depth=texture2D(currentDepth,sourceUv).r;
 float z=clipRange.x*clipRange.y/(clipRange.y*(1.-depth)+clipRange.x*depth);
 gl_FragColor=vec4(center,z);
 if(validHistory<.5)return;
 vec4 clip=vec4(sourceUv*2.-1.,2.*depth-1.,1.);
 bool sky=depth>=.999999;
 vec4 prev;
 if(sky){
   vec4 farWorld=worldFromClip*clip;
   vec3 direction=farWorld.xyz-sourceEye*farWorld.w;
   prev=previousStableClip*vec4(direction,0.);
 }else prev=previousFromClip*clip;
 if(movingBody>.5){
   vec4 body=localFromClip*clip;
   vec3 p=body.xyz/body.w;
   if(all(greaterThanEqual(p,boxMin)) && all(lessThanEqual(p,boxMax)))prev=previousBodyFromLocal*body;
 }
 if(prev.w<=0.)return;
 vec2 uv=prev.xy/prev.w*.5+.5;
 if(any(lessThan(uv,px))||any(greaterThan(uv,1.-px)))return;
 float lastZ=texelFetch(lastFrame,ivec2(uv*vec2(textureSize(lastFrame,0))),0).a;
 // Homogeneous clip.w for the source point is 1/z. Undo that scale before testing disocclusion.
 float expected=prev.w*z, tolerance=max(.085,expected*.018);
 float depthMatch=sky?step(clipRange.y*.995,lastZ):1.-smoothstep(tolerance,2.*tolerance,abs(lastZ-expected));
 if(!(depthMatch>=.001))return;
 vec3 c=compressHDR(center), lo=c,hi=c;
 for(int i=0;i<4;i++){
   vec2 offset=i==0?vec2(px.x,0.):(i==1?vec2(-px.x,0.):(i==2?vec2(0.,px.y):vec2(0.,-px.y)));
   vec3 neighbor=textureLod(currentColor,clamp(sourceUv+offset,px*.5,1.-px*.5),0.).rgb;
   if(!validHistoryColor(neighbor))continue;
   vec3 sampleColor=compressHDR(neighbor);
   lo=min(lo,sampleColor);hi=max(hi,sampleColor);
 }
 vec3 history=textureLod(lastFrame,uv,0.).rgb;
 if(!validHistoryColor(history))return;
 vec3 raw=compressHDR(history);
 vec3 bounded=clamp(raw,lo-.004,hi+.004);
 float disagreement=dot(abs(raw-bounded),vec3(.2126,.7152,.0722));
 float speed=length((uv-uvScreen)/px);
 float keep=mix(.84,.57,smoothstep(1.,10.,speed))*depthMatch*(1.-smoothstep(.025,.15,disagreement));
 gl_FragColor.rgb=expandHDR(mix(c,bounded,keep));
}`;

const JITTER = [[0,-1/6],[-.25,1/6],[.25,-7/18],[-.375,-1/18],[.125,5/18],[-.125,-5/18],[.375,1/18],[0,7/18]];
export class TemporalResolve extends Pass {
  private readonly frames=[0,1].map(()=>new THREE.WebGLRenderTarget(1,1,{
    type:THREE.HalfFloatType,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false,
  }));
  readonly source={value:null as THREE.Texture|null,sampleOffset:new THREE.Vector2()};
  readonly reflectionUniforms={
    uPreviousStreet:{value:null as THREE.Texture|null},uPreviousStreetClip:{value:new THREE.Matrix4()},uStreetHistory:{value:0},
  };
  private readonly material=new THREE.ShaderMaterial({
    uniforms:{
      currentColor:{value:null as THREE.Texture|null},currentDepth:{value:null as THREE.Texture|null},lastFrame:{value:null},
      previousFromClip:{value:new THREE.Matrix4()},previousStableClip:{value:new THREE.Matrix4()},localFromClip:{value:new THREE.Matrix4()},previousBodyFromLocal:{value:new THREE.Matrix4()},worldFromClip:{value:new THREE.Matrix4()},
      boxMin:{value:new THREE.Vector3()},boxMax:{value:new THREE.Vector3()},sourceEye:{value:new THREE.Vector3()},sourceOffset:{value:new THREE.Vector2()},clipRange:{value:new THREE.Vector2()},validHistory:{value:0},movingBody:{value:0},staticFrame:{value:0},
    },
    vertexShader:'varying vec2 uvScreen;void main(){uvScreen=uv;gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:RESOLVE,depthTest:false,depthWrite:false,
  });
  private readonly quad=new FullScreenQuad(this.material);
  readonly depthProjectionInverse=new THREE.Matrix4();
  readonly depthProjection=new THREE.Matrix4();
  private readonly stableClip=new THREE.Matrix4();
  private readonly clip=new THREE.Matrix4();
  private readonly inverse=new THREE.Matrix4();
  private readonly originalProjection=new THREE.Matrix4();
  private readonly previousBody=new THREE.Matrix4();
  private readonly eye=new THREE.Vector3();
  private readonly orientation=new THREE.Quaternion();
  private readonly bounds=new WeakMap<THREE.Object3D,THREE.Box3>();
  private body:THREE.Object3D|null=null;
  private frame=0;
  private previous=0;
  private valid=false;
  private ready=false;
  reflections=true;
  get stationary(){return this.valid && this.material.uniforms.staticFrame.value===1;}
  private readonly input:THREE.WebGLRenderTarget;
  private readonly camera:THREE.PerspectiveCamera;
  constructor(input:THREE.WebGLRenderTarget,camera:THREE.PerspectiveCamera){
    super();this.input=input;this.camera=camera;this.needsSwap=false;this.source.value=input.texture;
    this.material.uniforms.currentColor.value=input.texture;this.material.uniforms.currentDepth.value=input.depthTexture;
  }
  track(body:THREE.Object3D|null){
    if(body!==this.body){this.body=body;this.reset();}
  }
  reset(){
    this.valid=false;this.ready=false;this.frame=0;this.reflectionUniforms.uStreetHistory.value=0;
    this.source.value=this.input.texture;
  }
  setSize(w:number,h:number){
    w=Math.max(1,Math.floor(w));h=Math.max(1,Math.floor(h));
    if(this.frames[0].width===w&&this.frames[0].height===h)return;
    this.frames.forEach(f=>f.setSize(w,h));this.reset();
  }
  begin(){
    const c=this.camera,u=this.material.uniforms;
    if(this.eye.distanceToSquared(c.position)>16||this.orientation.angleTo(c.quaternion)>.25)this.reset();
    c.updateMatrixWorld();this.originalProjection.copy(c.projectionMatrix);
    this.stableClip.multiplyMatrices(this.originalProjection,c.matrixWorldInverse);
    const [x,y]=JITTER[this.frame++%JITTER.length];
    // Projection offsets move geometry by (-x,-y) raster pixels, so a stable output
    // pixel samples the jittered raster at (-x,-y) from its own coordinate.
    this.source.sampleOffset.set(-x/this.input.width,-y/this.input.height);
    // Only the beauty pass is jittered; restore before orbit picking, shadows and other cameras read this projection.
    c.projectionMatrix.elements[8]+=2*x/this.input.width;
    c.projectionMatrix.elements[9]+=2*y/this.input.height;
    c.projectionMatrixInverse.copy(c.projectionMatrix).invert();
    this.depthProjection.copy(c.projectionMatrix);
    this.depthProjectionInverse.copy(c.projectionMatrixInverse);
    this.clip.multiplyMatrices(c.projectionMatrix,c.matrixWorldInverse);
    this.inverse.copy(this.clip).invert();
    u.previousFromClip.value.multiplyMatrices(this.reflectionUniforms.uPreviousStreetClip.value,this.inverse);
    u.previousStableClip.value.copy(this.reflectionUniforms.uPreviousStreetClip.value);
    u.worldFromClip.value.copy(this.inverse);
    u.sourceEye.value.copy(c.position);u.sourceOffset.value.copy(this.source.sampleOffset);
    u.clipRange.value.set(c.near,c.far);
    u.lastFrame.value=this.frames[this.previous].texture;u.validHistory.value=Number(this.valid);
    u.movingBody.value=Number(!!this.body);
    // Use exact transforms so even a slow orbit/reverse resumes depth-tested
    // reprojection immediately. A cut/resize/reset cannot enter this path.
    u.staticFrame.value=Number(this.valid&&this.stableClip.equals(this.reflectionUniforms.uPreviousStreetClip.value));
    if(this.body){
      const body=this.body;body.updateWorldMatrix(true,false);
      if(!body.matrixWorld.equals(this.previousBody))u.staticFrame.value=0;
      const toLocal=u.localFromClip.value.copy(body.matrixWorld).invert();
      let bounds=this.bounds.get(body);
      if(!bounds){
        body.updateWorldMatrix(true,true);bounds=new THREE.Box3();const localBox=new THREE.Box3(),matrix=new THREE.Matrix4();
        body.traverse(child=>{if(child instanceof THREE.Mesh&&child.geometry.attributes.position){
          if(!child.geometry.boundingBox)child.geometry.computeBoundingBox();
          bounds!.union(localBox.copy(child.geometry.boundingBox!).applyMatrix4(matrix.multiplyMatrices(toLocal,child.matrixWorld)));
        }});
        this.bounds.set(body,bounds);
      }
      u.boxMin.value.copy(bounds.min);u.boxMax.value.copy(bounds.max);
      toLocal.multiply(this.inverse);
      u.previousBodyFromLocal.value.multiplyMatrices(this.reflectionUniforms.uPreviousStreetClip.value,this.previousBody);
    }
    this.reflectionUniforms.uPreviousStreet.value=this.frames[this.previous].texture;
    this.reflectionUniforms.uStreetHistory.value=Number(this.valid&&this.reflections);
    this.ready=true;
  }
  restore(){
    this.camera.projectionMatrix.copy(this.originalProjection);this.camera.projectionMatrixInverse.copy(this.originalProjection).invert();
    // Other scene draws (including reflection probes) must never trace the driving camera's buffer.
    this.reflectionUniforms.uStreetHistory.value=0;
  }
  render(renderer:THREE.WebGLRenderer){
    if(!this.ready){this.source.value=this.input.texture;return;}
    const next=1-this.previous;
    renderPostQuad(renderer,this.frames[next],this.quad);
    this.previous=next;this.valid=true;this.ready=false;
    this.source.value=this.frames[next].texture;
    this.reflectionUniforms.uPreviousStreetClip.value.copy(this.stableClip);
    if(this.body)this.previousBody.copy(this.body.matrixWorld);
    this.eye.copy(this.camera.position);this.orientation.copy(this.camera.quaternion);
  }
  dispose(){this.frames.forEach(f=>f.dispose());this.material.dispose();this.quad.dispose();}
}
