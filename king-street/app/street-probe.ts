import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';

export type StreetProbeState = {
  origin: THREE.IUniform<THREE.Vector3>;
  enabled: THREE.IUniform<number>;
  capture: THREE.IUniform<number>;
};

// Keep radial distance in the sharpest level of the radiance atlas. The
// roughness levels retain Three's ordinary filtered RGB. Both are addressed
// by the same direction, using one texture unit even on the wet-road shader.
export class DepthProbeCapture {
  readonly cube: THREE.WebGLCubeRenderTarget;
  private face: THREE.WebGLRenderTarget;
  private pack: THREE.ShaderMaterial;
  private quad: FullScreenQuad;

  constructor(size = 512) {
    this.cube = new THREE.WebGLCubeRenderTarget(size, {
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
    });
    this.face = new THREE.WebGLRenderTarget(size, size, {
      type: THREE.HalfFloatType,
      depthTexture: new THREE.DepthTexture(size, size, THREE.UnsignedIntType),
    });
    this.pack = new THREE.ShaderMaterial({
      uniforms: {
        beauty: { value: this.face.texture },
        depthMap: { value: this.face.depthTexture },
        planes: { value: new THREE.Vector2() },
      },
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
      vertexShader:
        'varying vec2 coord;void main(){coord=uv;gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader: `varying vec2 coord;
    uniform sampler2D beauty,depthMap;
    uniform vec2 planes;
    bool usableRadiance(vec3 c){return !any(isnan(c))&&!any(isinf(c));}
    void main(){
     ivec2 size=textureSize(beauty,0),p=clamp(ivec2(coord*vec2(size)),ivec2(0),size-1);
     float depth=texelFetch(depthMap,p,0).x;
     float viewZ=planes.x*planes.y/mix(planes.y,planes.x,depth);
     float radial=viewZ*length(vec3(2.*coord-1.,1.));
     vec3 radiance=texelFetch(beauty,p,0).rgb;
     // A single nonfinite texel spreads to whole roughness levels during
     // spherical convolution. Repair it in the source face before filtering.
     if(!usableRadiance(radiance)){
      vec3 sum=vec3(0.);float count=0.;
      for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
       vec3 neighbor=texelFetch(beauty,clamp(p+ivec2(x,y),ivec2(0),size-1),0).rgb;
       if(usableRadiance(neighbor)){sum+=neighbor;count+=1.;}
      }
      radiance=sum/max(count,1.);
     }
     gl_FragColor=vec4(clamp(radiance,0.,60000.),depth>=0.&&depth<.999999?min(radial,60000.):60000.);
    }`,
    });
    this.quad = new FullScreenQuad(this.pack);
  }

  renderFace(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
    face: number,
  ) {
    const prev = renderer.getRenderTarget(),
      prevFace = renderer.getActiveCubeFace(),
      prevMip = renderer.getActiveMipmapLevel();
    try {
      this.pack.uniforms.planes.value.set(camera.near, camera.far);
      renderer.setRenderTarget(this.face);
      renderer.render(scene, camera);
      renderer.setRenderTarget(this.cube, face);
      this.quad.render(renderer);
    } finally {
      renderer.setRenderTarget(prev, prevFace, prevMip);
    }
  }

  dispose() {
    this.quad.dispose();
    this.pack.dispose();
    this.face.dispose();
    this.cube.dispose();
  }
}

export const STREET_PROBE_GLSL = /* glsl */ `
uniform vec3 uStreetProbeOrigin;
uniform float uStreetProbeEnabled,uStreetProbeCapture;
#ifdef ENVMAP_TYPE_CUBE_UV

// PMREM packs the six top-level faces in two rows of three, with a one-texel gutter.
float streetRange(sampler2D atlas,vec3 d) {
 float f=getFace(d),size=float(textureSize(atlas,0).y)*.25;
 vec2 tile=vec2(mod(f,3.),floor(f/3.));
 vec2 uv=tile*size+1.+getUV(d,f)*(size-2.);
 return texelFetch(atlas,ivec2(uv),0).a;
}
bool solveStreetHit(sampler2D atlas,vec3 offset,vec3 ray,float along,float across2,float range,float seedTravel,out vec3 direction,out float travel){
 for(int i=0;i<5;i++){
  if(range*range<=across2 || range<.36 || range>350.)break;
  float halfChord=sqrt(range*range-across2);
  float nearTravel=-along-halfChord,farTravel=-along+halfChord;
  travel=(nearTravel>.12 && abs(nearTravel-seedTravel)<abs(farTravel-seedTravel))?nearTravel:farTravel;
  if(travel<=.12)break;
  direction=normalize(offset+ray*travel);
  float next=streetRange(atlas,direction);
  if(abs(next-range)<max(.10,.008*range))return true;
  range=next;seedTravel=travel;
 }
 return false;
}
vec3 correctStreetDirection(sampler2D atlas,vec3 world,vec3 ray,float roughness){
 if(uStreetProbeEnabled<.5 || uStreetProbeCapture>.5 || roughness>.43)return ray;
 vec3 offset=world-uStreetProbeOrigin;
 float along=dot(offset,ray),across2=max(0.,dot(offset,offset)-along*along);
 if(dot(offset,offset)>8100.)return ray;
 vec3 direction=ray,farDirection=ray;
 float range=streetRange(atlas,ray),farTravel=0.;
 bool farHit=solveStreetHit(atlas,offset,ray,along,across2,range,60000.,farDirection,farTravel);
 // A street behind the receiver can lie in a different part of the cube.
 // The rotation-only ray may see sky, or a wall behind a nearer parked car.
 // Search in increasing order from the receiver, bounded by the far solution.
 float limit=farHit?farTravel:110.;
 float travel=.30,previous=.12;
 for(int i=0;i<17;i++){
  if(travel>=limit)break;
  vec3 point=offset+travel*ray;
  float radial=length(point);
  float surface=streetRange(atlas,point/max(.001,radial));
  if(surface<350. && abs(surface-radial)<max(.48,1.45*(travel-previous))){
   float hitTravel;
   if(solveStreetHit(atlas,offset,ray,along,across2,surface,travel,direction,hitTravel) && hitTravel<=limit &&
      abs(hitTravel-travel)<max(.65,2.*(travel-previous)))return direction;
  }
  previous=travel;travel=travel*1.29+.34;
 }
 return farHit?farDirection:ray;
}
#endif`;
