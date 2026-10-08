import * as THREE from 'three';

// The road field: what a road surface needs to know to wear as a road does. A road's texture is projected from above,
// so a fragment knows nothing of which way the road runs, where its lanes are or where its kerbs stand. This draws the
// road surfaces once from above at load, finds every road pixel's nearest kerb edge by a jump flood on the GPU, and keeps
// per pixel: the unit vector from that kerb across the road, the distance from it, and half the road's width there
// (the largest kerb distance along that line, i.e. the centre line's). From those a shader places wheel paths, oil
// drips down the lane centres, gutter grime, and cracks across and along the road. The same flood measures, everywhere
// off the road, how far the kerb is (a sidewalk's distance from its kerb) and, from the buildings drawn alongside, how
// far the nearest building wall is (a sidewalk's distance from its frontage). Nothing here is location specific.

const VERT = 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}';
const MASK_VERT = 'void main(){gl_Position=projectionMatrix*viewMatrix*modelMatrix*vec4(position,1.);}';
const MASK_FRAG = 'uniform vec4 uColor;void main(){gl_FragColor=uColor;}';
// mask: red where road, green where building. A road pixel with a non-road neighbour is a kerb seed, a building pixel
// with a non-building neighbour a wall seed (each its own coordinate, in .xy and .zw); every other pixel starts empty
const SEED = /* glsl */`
precision highp float;uniform sampler2D tMask;uniform vec2 uPx;varying vec2 vUv;
void main(){
 vec4 m=texture2D(tMask,vUv);
 vec4 n=min(min(texture2D(tMask,vUv+vec2(uPx.x,0.)),texture2D(tMask,vUv-vec2(uPx.x,0.))),min(texture2D(tMask,vUv+vec2(0.,uPx.y)),texture2D(tMask,vUv-vec2(0.,uPx.y))));
 gl_FragColor=vec4((m.r>.5&&n.r<.5)?vUv:vec2(-1.),(m.g>.5&&n.g<.5)?vUv:vec2(-1.));
}`;
const FLOOD = /* glsl */`
precision highp float;uniform sampler2D tSeeds;uniform float uStep;uniform vec2 uPx;varying vec2 vUv;
void main(){
 vec4 best=texture2D(tSeeds,vUv);float bd=best.x<0.?1e9:distance(best.xy,vUv),be=best.z<0.?1e9:distance(best.zw,vUv);
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
  vec2 q=vUv+vec2(float(x),float(y))*uStep*uPx;if(q.x<0.||q.y<0.||q.x>1.||q.y>1.)continue;
  vec4 s=texture2D(tSeeds,q);
  if(s.x>=0.){float d=distance(s.xy,vUv);if(d<bd){bd=d;best.xy=s.xy;}}
  if(s.z>=0.){float d=distance(s.zw,vUv);if(d<be){be=d;best.zw=s.zw;}}}
 gl_FragColor=best;
}`;
// per pixel: the kerb distance (m; signed: positive on the road, negative off it, zero at the road's edge), the half
// width (m, on the road): the largest kerb distance met walking on across the road, how straight the road is here:
// whether the kerb distance holds along the road for a few metres either way (it does on a street, not in an
// intersection or round a corner), and the distance from the nearest building wall (m; negative inside a building)
const PACK = /* glsl */`
precision highp float;uniform sampler2D tSeeds;uniform sampler2D tMask;uniform float uSize;uniform vec2 uPx;varying vec2 vUv;
float kerb(vec2 uv){vec4 s=texture2D(tSeeds,uv);return s.x<0.?0.:distance(s.xy,uv)*uSize;}
void main(){
 vec4 m=texture2D(tMask,vUv);vec4 s=texture2D(tSeeds,vUv);float texel=uSize*uPx.x;
 // a seed is a pixel at the edge, half a pixel from the edge itself: off the road (or a building) the distance to the
 // seed's centre less half a pixel, on it the distance plus half
 float wall=s.z<0.?99.:distance(s.zw,vUv)*uSize;wall=m.g>.5?-(wall+.5*texel):max(wall-.5*texel,0.);
 if(m.r<.5){gl_FragColor=vec4(s.x<0.?-99.:-max(distance(s.xy,vUv)*uSize-.5*texel,0.),0.,0.,wall);return;}
 if(s.x<0.){gl_FragColor=vec4(0.,0.,0.,wall);return;}
 vec2 v=(vUv-s.xy)*uSize;float d=length(v);vec2 g=d>1e-4?v/d:vec2(1.,0.);
 float half_=max(d,.25);
 for(int k=1;k<=48;k++){vec2 q=vUv+g*(float(k)*.5)/uSize;if(texture2D(tMask,q).r<.5)break;half_=max(half_,kerb(q));}
 vec2 t=vec2(-g.y,g.x);float dev=0.;
 for(int k=-2;k<=2;k++){if(k==0)continue;vec2 q=vUv+t*(float(k)*3.)/uSize;dev=max(dev,texture2D(tMask,q).r<.5?9.:abs(kerb(q)-d));}
 gl_FragColor=vec4(d,half_,1.-smoothstep(.35,1.4,dev),wall);
}`;

/** `roads`: the road surfaces; `buildings`: what stands on each building's footprint seen from above (its roofs and
 *  walls), for the distance from the nearest frontage. */
export function buildRoadField(renderer: THREE.WebGLRenderer, scene: THREE.Scene, roads: THREE.Object3D[], reach: number, size = 2048, buildings: THREE.Object3D[] = []) {
  const R = reach, N = size, span = 2 * R;
  const prevTarget = renderer.getRenderTarget(), prevAuto = renderer.shadowMap.autoUpdate, prevOverride = scene.overrideMaterial, prevBg = scene.background, prevFog = scene.fog;
  const clear = renderer.getClearColor(new THREE.Color()), clearA = renderer.getClearAlpha();
  // 1. the road surfaces (red) and the buildings (green) from above (layer 7 holds only the ones being drawn)
  const LAYER = 7;
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, 1, 4000); cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.layers.set(LAYER); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const mask = new THREE.WebGLRenderTarget(N, N, {type: THREE.UnsignedByteType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false});
  const maskMat = new THREE.ShaderMaterial({vertexShader: MASK_VERT, fragmentShader: MASK_FRAG, uniforms: {uColor: {value: new THREE.Vector4()}}, side: THREE.DoubleSide, depthTest: false, depthWrite: false,
    blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor});
  scene.overrideMaterial = maskMat; scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false; renderer.setClearColor(0x000000, 0);
  // both sets into one target: the renderer must not clear it between them
  const prevClear = renderer.autoClear; renderer.autoClear = false;
  renderer.setRenderTarget(mask); renderer.clear();
  for (const [set, color] of [[roads, [1, 0, 0, 1]], [buildings, [0, 1, 0, 1]]] as const) {
    if (!set.length) continue;
    set.forEach(o => o.traverse(x => x.layers.enable(LAYER))); maskMat.uniforms.uColor.value.set(...color);
    renderer.render(scene, cam);
    set.forEach(o => o.traverse(x => x.layers.disable(LAYER)));
  }
  renderer.autoClear = prevClear;
  scene.overrideMaterial = prevOverride; scene.background = prevBg; scene.fog = prevFog; renderer.shadowMap.autoUpdate = prevAuto;
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
  // nothing flips across the centre line; the kerb distance changes sign, not slope, at the road's edge); a 0.3 m texel
  // sampled nearest drew stair-steps along every band
  const field = new THREE.WebGLRenderTarget(N, N, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
  const pack = new THREE.ShaderMaterial({vertexShader: VERT, fragmentShader: PACK, uniforms: {tSeeds: {value: src.texture}, tMask: {value: mask.texture}, uSize: {value: span}, uPx: {value: px}}, depthTest: false, depthWrite: false});
  quad.material = pack; renderer.setRenderTarget(field); renderer.render(qs, qc);
  renderer.setRenderTarget(prevTarget); renderer.setClearColor(clear, clearA);
  [a, b, mask].forEach(t => t.dispose()); [maskMat, seed, flood, pack].forEach(m => m.dispose()); quad.geometry.dispose();
  const uniforms = {uRfTex: {value: field.texture}, uRfR: {value: R}};
  return {uniforms, texture: field.texture, target: field, dispose() { field.dispose(); }};
}
export type RoadField = ReturnType<typeof buildRoadField>;

/** GLSL: the road field at a world position. rfAt(P) returns the filtered texel: .x kerb distance (m; positive on the
 *  road, negative off it), .y half width (m, zero off the road), .z straightness (1 along a street, 0 in an
 *  intersection), .w distance from the nearest building wall (m; negative inside one). rfAcross(P) is the unit
 *  direction away from the kerb into the road (east, north), from the kerb distance's gradient; off the road it points
 *  back toward the kerb. The field has no mipmaps and is read at level 0: safe inside a branch that differs between
 *  neighbouring pixels (an implicit-derivative lookup there is undefined, and Metal returns NaN). */
export const ROADFIELD_GLSL = /* glsl */`
uniform sampler2D uRfTex;uniform float uRfR;
vec4 rfAt(vec3 P){vec2 uv=vec2(P.x,-P.z)/(2.*uRfR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return vec4(0.,0.,0.,99.);return textureLod(uRfTex,uv,0.);}
vec2 rfAcross(vec3 P,float d){float e=.6;vec2 gr=vec2(rfAt(P+vec3(e,0.,0.)).x-d,rfAt(P+vec3(0.,0.,-e)).x-d);float l=length(gr);return l>1e-4?gr/l:vec2(1.,0.);}
`;
