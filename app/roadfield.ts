import * as THREE from 'three';

// The road field: what a road surface needs to know to wear as a road does. A road's texture is projected from above,
// so a fragment knows nothing of which way the road runs, where its lanes are or where its kerbs stand. This draws the
// road surfaces once from above at load, finds every road pixel's nearest kerb edge by a jump flood on the GPU, and keeps
// per pixel: the unit vector from that kerb across the road, the distance from it, and half the road's width there
// (the largest kerb distance along that line, i.e. the centre line's). From those a shader places wheel paths, oil
// drips down the lane centres, gutter grime, and cracks across and along the road. Nothing here is location specific.

const VERT = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
const MASK_VERT = 'void main(){gl_Position=projectionMatrix*viewMatrix*modelMatrix*vec4(position,1.);}';
const MASK_FRAG = 'void main(){gl_FragColor=vec4(1.);}';
// a road pixel with a non-road neighbour is a kerb seed (its own coordinate); every other pixel starts empty
const SEED = /* glsl */`
precision highp float;uniform sampler2D tMask;uniform vec2 uPx;varying vec2 vUv;
void main(){
 float r=texture2D(tMask,vUv).r;
 float n=min(min(texture2D(tMask,vUv+vec2(uPx.x,0.)).r,texture2D(tMask,vUv-vec2(uPx.x,0.)).r),min(texture2D(tMask,vUv+vec2(0.,uPx.y)).r,texture2D(tMask,vUv-vec2(0.,uPx.y)).r));
 gl_FragColor=(r>.5&&n<.5)?vec4(vUv,0.,1.):vec4(-1.,-1.,0.,0.);
}`;
const FLOOD = /* glsl */`
precision highp float;uniform sampler2D tSeeds;uniform float uStep;uniform vec2 uPx;varying vec2 vUv;
void main(){
 vec4 best=texture2D(tSeeds,vUv);float bd=best.x<0.?1e9:distance(best.xy,vUv);
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  vec2 q=vUv+vec2(float(x),float(y))*uStep*uPx;if(q.x<0.||q.y<0.||q.x>1.||q.y>1.)continue;
  vec4 s=texture2D(tSeeds,q);if(s.x<0.)continue;float d=distance(s.xy,vUv);if(d<bd){bd=d;best=s;}}
 gl_FragColor=best;
}`;
// per road pixel: the kerb distance (m), the half width (m): the
// largest kerb distance met walking on across the road, and how straight the road is here: whether the kerb distance
// holds along the road for a few metres either way (it does on a street, not in an intersection or round a corner)
const PACK = /* glsl */`
precision highp float;uniform sampler2D tSeeds;uniform sampler2D tMask;uniform float uSize;varying vec2 vUv;
float kerb(vec2 uv){vec4 s=texture2D(tSeeds,uv);return s.x<0.?0.:distance(s.xy,uv)*uSize;}
void main(){
 if(texture2D(tMask,vUv).r<.5){gl_FragColor=vec4(0.);return;}
 vec4 s=texture2D(tSeeds,vUv);if(s.x<0.){gl_FragColor=vec4(0.);return;}
 vec2 v=(vUv-s.xy)*uSize;float d=length(v);vec2 g=d>1e-4?v/d:vec2(1.,0.);
 float half_=max(d,.25);
 for(int k=1;k<=48;k++){vec2 q=vUv+g*(float(k)*.5)/uSize;if(texture2D(tMask,q).r<.5)break;half_=max(half_,kerb(q));}
 vec2 t=vec2(-g.y,g.x);float dev=0.;
 for(int k=-2;k<=2;k++){if(k==0)continue;vec2 q=vUv+t*(float(k)*3.)/uSize;dev=max(dev,texture2D(tMask,q).r<.5?9.:abs(kerb(q)-d));}
 gl_FragColor=vec4(d,half_,1.-smoothstep(.35,1.4,dev),1.);
}`;

export function buildRoadField(renderer: THREE.WebGLRenderer, scene: THREE.Scene, roads: THREE.Object3D[], reach: number, size = 2048) {
  const R = reach, N = size, span = 2 * R;
  const prevTarget = renderer.getRenderTarget(), prevAuto = renderer.shadowMap.autoUpdate, prevOverride = scene.overrideMaterial, prevBg = scene.background, prevFog = scene.fog;
  const clear = renderer.getClearColor(new THREE.Color()), clearA = renderer.getClearAlpha();
  // 1. the road surfaces from above (layer 7 holds only them)
  const LAYER = 7; roads.forEach(o => o.traverse(x => x.layers.enable(LAYER)));
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, 1, 4000); cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.layers.set(LAYER); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const mask = new THREE.WebGLRenderTarget(N, N, {type: THREE.UnsignedByteType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false});
  const maskMat = new THREE.ShaderMaterial({vertexShader: MASK_VERT, fragmentShader: MASK_FRAG, side: THREE.DoubleSide, depthTest: false, depthWrite: false});
  scene.overrideMaterial = maskMat; scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false; renderer.setClearColor(0x000000, 0);
  renderer.setRenderTarget(mask); renderer.clear(); renderer.render(scene, cam);
  scene.overrideMaterial = prevOverride; scene.background = prevBg; scene.fog = prevFog; renderer.shadowMap.autoUpdate = prevAuto;
  roads.forEach(o => o.traverse(x => x.layers.disable(LAYER)));
  // 2-4. seeds, the jump flood, the pack
  const ft = () => new THREE.WebGLRenderTarget(N, N, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false});
  const a = ft(), b = ft();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); const qs = new THREE.Scene(); qs.add(quad); const qc = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const px = new THREE.Vector2(1 / N, 1 / N);
  const seed = new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader: SEED, uniforms: {tMask: {value: mask.texture}, uPx: {value: px}}, depthTest: false, depthWrite: false});
  quad.material = seed; renderer.setRenderTarget(a); renderer.render(qs, qc);
  const flood = new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader: FLOOD, uniforms: {tSeeds: {value: null}, uStep: {value: 1}, uPx: {value: px}}, depthTest: false, depthWrite: false});
  quad.material = flood; let src = a, dst = b;
  for (let step = N / 2; step >= 1; step /= 2) { flood.uniforms.tSeeds.value = src.texture; flood.uniforms.uStep.value = step; renderer.setRenderTarget(dst); renderer.render(qs, qc); [src, dst] = [dst, src]; }
  flood.uniforms.tSeeds.value = src.texture; flood.uniforms.uStep.value = 1; renderer.setRenderTarget(dst); renderer.render(qs, qc); [src, dst] = [dst, src];   // JFA+1
  // filtered linearly: every stored value is continuous (the direction is taken from the kerb distance's gradient, so
  // nothing flips across the centre line); a 0.3 m texel sampled nearest drew stair-steps along every band
  const field = new THREE.WebGLRenderTarget(N, N, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
  const pack = new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader: PACK, uniforms: {tSeeds: {value: src.texture}, tMask: {value: mask.texture}, uSize: {value: span}}, depthTest: false, depthWrite: false});
  quad.material = pack; renderer.setRenderTarget(field); renderer.render(qs, qc);
  renderer.setRenderTarget(prevTarget); renderer.setClearColor(clear, clearA);
  [a, b, mask].forEach(t => t.dispose()); [maskMat, seed, flood, pack].forEach(m => m.dispose()); quad.geometry.dispose();
  const uniforms = {uRfTex: {value: field.texture}, uRfR: {value: R}};
  return {uniforms, texture: field.texture, target: field, dispose() { field.dispose(); }};
}
export type RoadField = ReturnType<typeof buildRoadField>;

/** GLSL: the road field at a world position. rfAt(P) returns the filtered texel: .x kerb distance (m), .y half width
 *  (m, zero off the road), .z straightness (1 along a street, 0 in an intersection). rfAcross(P) is the unit direction
 *  away from the kerb (east, north), from the kerb distance's gradient. */
export const ROADFIELD_GLSL = /* glsl */`
uniform sampler2D uRfTex;uniform float uRfR;
vec4 rfAt(vec3 P){vec2 uv=vec2(P.x,-P.z)/(2.*uRfR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return vec4(0.);return texture2D(uRfTex,uv);}
vec2 rfAcross(vec3 P,float d){float e=.6;vec2 gr=vec2(rfAt(P+vec3(e,0.,0.)).x-d,rfAt(P+vec3(0.,0.,-e)).x-d);float l=length(gr);return l>1e-4?gr/l:vec2(1.,0.);}
`;
