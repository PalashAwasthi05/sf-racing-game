import * as THREE from 'three';

// Surface finish: scanned material detail for the surfaces that carry a colour and nothing else: the mid-distance
// facade tier, the streamed roofs, the near walls whose material is a measured colour without a texture map, tree bark
// and lawns. A plane of one colour reads as plastic at any distance.
//
// Each kind takes a photo-scanned CC0 material (work/material-assets, packed by pack_detail.py into one texture: normal
// in RG, roughness in B, tone relative to the scan's average in A) projected in world space, so it needs no UVs and is
// continuous across a building:
//   walls   plaster (Plaster001), sampled at two sizes so the tile does not show; roof and ledge tops, gravel (Ground110)
//   bark    Bark016;   lawns   Grass004
// The scan gives relief (its normal map), roughness variation and tonal detail. Its hue is not used: the surface's own
// colour is multiplied by the scan's tone relative to its average, so a wall keeps the colour it was given (the colours
// come from photographs and are checked against them). Over that, walls keep two procedural terms no tile can give:
// tone drifting over metres, and faint vertical streaking where rain runs down. Nothing here is location specific, and
// no structure is invented: no joints, boards, bricks or panels, which need evidence (WoodSiding008 is in the library
// for walls the pipeline identifies as siding; nothing assigns it yet).

type Kind = 'wall' | 'masonry' | 'bark' | 'grass';
// side: the projection on vertical faces; top: on horizontal ones. size: metres per tile; tone: how much of the scan's
// tonal detail is used; bump: normal strength; rough: the scan's mean roughness (its variation about that is added).
const SETS: Record<Kind, {side: string; top: string; size: [number, number]; tone: [number, number]; bump: [number, number]; rough: [number, number]}> = {
  // walls and masonry take a generated wall finish (wall-layers.png, one layer chosen per material) instead of `side`
  wall: {side: 'Plaster001', top: 'Ground110', size: [2.4, 2.1], tone: [.65, .3], bump: [.8, .7], rough: [.79, .54]},
  masonry: {side: 'Plaster001', top: 'Ground110', size: [1.7, 2.1], tone: [.6, .3], bump: [.5, .7], rough: [.79, .54]},
  bark: {side: 'Bark016', top: 'Bark016', size: [.8, .8], tone: [.55, .55], bump: [1.4, 1.4], rough: [.6, .6]},
  grass: {side: 'Grass004', top: 'Grass004', size: [1.4, 1.4], tone: [.5, .5], bump: [.8, .8], rough: [.26, .26]},
};

const HEADER = /* glsl */`
#ifdef SF_WALL
uniform highp sampler2DArray uSfWall;uniform float uSfLayer;
#else
uniform sampler2D uSfSide;
#endif
#ifndef SF_ONE
uniform sampler2D uSfTop;
#else
#ifndef SF_WALL
#define uSfTop uSfSide
#endif
#endif
uniform vec2 uSfSize,uSfTone,uSfBumpK,uSfRoughMean;uniform float uSfWeather;
varying vec3 vSfPos;varying vec3 vSfNrm;
vec3 sfNormal;float sfRough;
float sfHash3(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}
float sfNoise3(vec3 p){
 vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 float a=sfHash3(i),b=sfHash3(i+vec3(1.,0.,0.)),c=sfHash3(i+vec3(0.,1.,0.)),d=sfHash3(i+vec3(1.,1.,0.));
 float e=sfHash3(i+vec3(0.,0.,1.)),f0=sfHash3(i+vec3(1.,0.,1.)),g=sfHash3(i+vec3(0.,1.,1.)),h=sfHash3(i+vec3(1.,1.,1.));
 return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),mix(mix(e,f0,f.x),mix(g,h,f.x),f.y),f.z);
}
vec3 sfFinish(vec3 albedo){
 vec3 P=vSfPos;vec3 N=normalize(vSfNrm);
 float wall=smoothstep(.3,.7,1.-abs(N.y));
 sfNormal=vec3(0.);sfRough=0.;float tone=1.;
 if(wall>.01){
  // two world projections, weighted by which way the wall faces, so the scan runs unbroken round a curved bay
  vec2 w=pow(abs(N.xz),vec2(4.));w/=max(w.x+w.y,1e-4);
  vec2 ua=vec2(P.z,P.y)/uSfSize.x,ub=vec2(P.x,P.y)/uSfSize.x;
#ifdef SF_WALL
  // One generated finish per material, read at its own size and again 4.3 times larger and turned a quarter (the
  // wall's slow unevenness, and no visible repeat). A flat wall faces one axis, so only that projection is read:
  // two lookups a pixel; the gradients are taken outside the branches so each still picks the right mip.
  bool first=w.x>w.y;vec2 uv=first?ua:ub,dx=first?dFdx(ua):dFdx(ub),dy=first?dFdy(ua):dFdy(ub);float blend=min(w.x,w.y);
  vec3 a=textureGrad(uSfWall,vec3(uv,uSfLayer),dx,dy).rgb;
#ifdef SF_MASONRY
  vec3 m=vec3(.5);   // brick and stone keep their own pattern: the face grain alone, one lookup
#else
  vec3 m=textureGrad(uSfWall,vec3(uv.yx*.233+.37,uSfLayer),dx.yx*.233,dy.yx*.233).rgb;
#endif
  vec2 na=a.xy*2.-1.+(vec2(m.y,m.x)*2.-1.)*.45;float t=a.z*2.*(.9+.2*m.z);
  if(blend>.03){vec2 u2=first?ub:ua;vec3 b=textureGrad(uSfWall,vec3(u2,uSfLayer),first?dFdx(ub):dFdx(ua),first?dFdy(ub):dFdy(ua)).rgb;
   // round a corner or a curved bay the other projection fades in; its slope is dropped, its tone kept
   na*=1.-blend;t=mix(t,b.z*2.,blend);}
  tone*=mix(1.,t,wall*uSfTone.x);
  sfNormal+=wall*uSfBumpK.x*(first?vec3(0.,na.y,na.x):vec3(na.x,na.y,0.));
  sfRough+=wall*(1.-t)*.35;
#else
  vec4 a=texture2D(uSfSide,ua),b=texture2D(uSfSide,ub);
  // the same scan 3.7 times larger and offset: its tone alone, so a wall is not one tile repeated
  float far=texture2D(uSfSide,ua.yx*.27+.37).a*w.x+texture2D(uSfSide,ub.yx*.27+.37).a*w.y;
  vec4 s=a*w.x+b*w.y;
  tone*=mix(1.,(s.a*2.)*(.55+.9*far),wall*uSfTone.x);
  vec2 na=a.xy*2.-1.,nb=b.xy*2.-1.;
  sfNormal+=wall*uSfBumpK.x*(w.x*vec3(0.,na.y,na.x)+w.y*vec3(nb.x,nb.y,0.));
  sfRough+=wall*(s.b-uSfRoughMean.x);
#endif
  if(uSfWeather>.5){
   // weather no tile carries: tone drifting over metres, and rain streaking, narrow across the wall and long down it
   tone*=1.+(sfNoise3(P*.31+vec3(4.1))-.5)*.14+(sfNoise3(P*.93+vec3(9.7))-.5)*.08;
   float px=max(length(fwidth(P))*.7,1e-4),keep=1.-smoothstep(.3,.6,px*7.);
   float streak=sfNoise3(P*vec3(7.,.35,7.)+vec3(2.3))-.5,run=sfNoise3(P*vec3(1.9,.12,1.9)+vec3(6.1));
   tone*=1.+wall*streak*.2*smoothstep(.3,.65,run)*keep;
  }
 }
 if(wall<.99){
  float flatness=1.-wall;
#if defined(SF_WALL) && defined(SF_ONE)
  vec3 g=texture(uSfWall,vec3(P.xz/uSfSize.y,uSfLayer)).rgb;vec4 t=vec4(g.xy,uSfRoughMean.y,g.z);
#else
  vec4 t=texture2D(uSfTop,P.xz/uSfSize.y);
#endif
  tone*=mix(1.,t.a*2.,flatness*uSfTone.y);
  vec2 nt=t.xy*2.-1.;sfNormal+=flatness*uSfBumpK.y*vec3(nt.x,0.,nt.y);
  sfRough+=flatness*(t.b-uSfRoughMean.y);
  if(uSfWeather>.5)tone*=1.+flatness*((sfNoise3(P*vec3(.55,.2,.55)+vec3(8.8))-.5)*.16+(sfNoise3(P*vec3(1.7,.3,1.7)+vec3(1.4))-.5)*.08);
 }
 return albedo*tone;
}
`;

// A flat, mid-rough, average-tone texel until a scan has loaded.
const placeholder = () => { const t = new THREE.DataTexture(new Uint8Array([128, 128, 160, 128]), 1, 1, THREE.RGBAFormat); t.needsUpdate = true; return t; };
const scans = new Map<string, {value: THREE.Texture}>();
/** The shared uniform holding a packed scan; it starts as the placeholder and takes the image when it arrives. */
function scan(name: string) {
  let u = scans.get(name); if (u) return u;
  u = {value: placeholder()}; scans.set(name, u);
  if (typeof Image !== 'undefined' && typeof document !== 'undefined' && typeof (document as Document).createElementNS === 'function') {
    const holder = u;
    new THREE.TextureLoader().load(`/materials/${name}.png`, texture => {
      texture.colorSpace = THREE.NoColorSpace; texture.wrapS = texture.wrapT = THREE.RepeatWrapping; texture.anisotropy = 8;
      texture.minFilter = THREE.LinearMipmapLinearFilter; texture.magFilter = THREE.LinearFilter; texture.generateMipmaps = true; texture.premultiplyAlpha = false;
      holder.value.dispose(); holder.value = texture;
    }, undefined, () => { /* a missing scan leaves the placeholder: the surface keeps its plain colour */ });
  }
  return u;
}

const WALL_LAYERS = 5;
const wallLayers: {value: THREE.DataArrayTexture} = {value: (() => { const data = new Uint8Array(4 * WALL_LAYERS); for (let i = 0; i < WALL_LAYERS; i++) data.set([128, 128, 128, 255], 4 * i);
  const t = new THREE.DataArrayTexture(data, 1, 1, WALL_LAYERS); t.needsUpdate = true; return t; })()};
let wallsRequested = false;
/** The generated wall finishes (work/material-assets/generate_walls.py): one array texture, so every wall material
 *  spends a single texture unit whichever finish it takes. */
function loadWallLayers() {
  if (wallsRequested || typeof Image === 'undefined' || typeof document === 'undefined' || typeof (document as Document).createElementNS !== 'function') return; wallsRequested = true;
  const image = new Image();
  image.onload = () => {
    const size = image.width, canvas = document.createElement('canvas'); canvas.width = size; canvas.height = size * WALL_LAYERS;
    const context = canvas.getContext('2d', {willReadFrequently: true}); if (!context || image.height !== size * WALL_LAYERS) return;
    context.drawImage(image, 0, 0);
    const t = new THREE.DataArrayTexture(new Uint8Array(context.getImageData(0, 0, size, size * WALL_LAYERS).data.buffer), size, size, WALL_LAYERS);
    t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType; t.colorSpace = THREE.NoColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter; t.generateMipmaps = true; t.anisotropy = 4; t.needsUpdate = true;
    wallLayers.value.dispose(); wallLayers.value = t;
  };
  image.src = '/materials/wall-layers.png';
}
/** Which finish a wall material takes. The pipeline's class name decides where it names one; a wall known only by its
 *  colour takes one of the plain finishes by that colour, so neighbouring buildings differ. 0 lace stucco, 1 floated
 *  stucco, 2 cast stone, 3 paint, 4 masonry face. */
function wallLayer(m: THREE.Material, kind: Kind) {
  if (kind === 'masonry') return 4;
  let h = 0; for (let i = 0; i < m.name.length; i++) h = Math.imul(h ^ m.name.charCodeAt(i), 16777619) >>> 0;
  if (/^stucco/.test(m.name)) return h % 3 === 0 ? 1 : 0;
  if (/^(render|plaster)/.test(m.name)) return 1;
  if (/^cast_stone/.test(m.name) || m.userData.layer === 'roofs') return 2;
  if (/^paint/.test(m.name)) return 3;
  return m.userData.class === 'facade_middle' ? 1 : [1, 3, 0, 1][h % 4];
}

function kindOf(m: THREE.Material): Kind | null {
  if (!(m instanceof THREE.MeshStandardMaterial) || m.userData.surfaceFinish || m.userData.glazed) return null;
  if (m.transparent || (m.metalness > .3 && !m.metalnessMap)) return null;       // a metal; with a metalness map the factor is not the value
  if (m.userData.bark || /bark/.test(m.name)) return m.map ? null : 'bark';
  // Plaster-like classes get the plaster scan over whatever flat tint map they carry; other mapped materials (brick,
  // stone, timber, the impostor pages) already have a real texture.
  const plaster = /^(stucco|render|plaster|cast_stone|paint)/.test(m.name);       // the pipeline's own class names
  // Brick and stone carry their own pattern and normal map; they take the masonry face grain over it.
  if (m.map && !plaster) return m.userData.weathered && !m.userData.groundDetail && m.normalMap && /^(auto_)?(brick|stone|masonry|terracotta)/.test(m.name) ? 'masonry' : null;
  if (m.name === 'grass' && String(m.userData.groundDetail ?? '').startsWith('grass')) return 'grass';
  if (m.userData.groundDetail || m.roughness < .5 || /glazing|glass|canopy|foliage/.test(m.name)) return null;
  return plaster || m.userData.weathered || m.userData.class === 'facade_middle' || (m.userData.class === 'distant_exterior' && m.userData.layer === 'roofs') ? 'wall' : null;
}
/** Whether a material is one of the plain-coloured surfaces that get a scanned finish. */
export function wantsFinish(m: THREE.Material) { return kindOf(m) !== null; }

/** Adds the scanned finish to the plain-coloured materials among those given, and sky reflections to building glass.
 *  Chains after any existing shader hook. */
export function finishSurfaces(materials: Iterable<THREE.Material>) {
  for (const m of materials) {
    if (m.userData.glazed) { reflectSky(m); continue; }
    const kind = kindOf(m); if (!kind) continue;
    const std = m as THREE.MeshStandardMaterial, set = SETS[kind]; std.userData.surfaceFinish = kind;
    // A material that already carries its own maps is near the 16 texture units a fragment shader may use: it takes the
    // side scan alone, on its tops too.
    const one = !!std.map, walled = kind === 'wall' || kind === 'masonry';
    if (walled) loadWallLayers();
    // Even so, a plaster wall with weathering, window lights and its own four maps came to 17 (found 2026-11-13: the
    // warning was logged for every such draw). Plaster, stucco and paint are not metal, and their metalness map is the
    // roughness texture bound a second time, so it goes and its unit pays for the scan.
    if (one && std.metalnessMap) { std.metalnessMap = null; std.metalness = 0; }
    if (kind === 'masonry') std.normalScale.multiplyScalar(1.7);   // the pipeline's joints were too shallow to catch the sun
    const uniforms: Record<string, THREE.IUniform> = {...(walled ? {uSfWall: wallLayers, uSfLayer: {value: wallLayer(std, kind)}} : {uSfSide: scan(set.side)}), ...(one ? {} : {uSfTop: scan(set.top)}), uSfSize: {value: new THREE.Vector2(...set.size)}, uSfTone: {value: new THREE.Vector2(...set.tone)},
      uSfBumpK: {value: new THREE.Vector2(...set.bump)}, uSfRoughMean: {value: new THREE.Vector2(...set.rough)}, uSfWeather: {value: kind === 'wall' ? 1 : 0}};
    const previous = std.onBeforeCompile.bind(std);
    std.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vSfPos;varying vec3 vSfNrm;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvSfPos=(modelMatrix*vec4(transformed,1.)).xyz;vSfNrm=normalize(mat3(modelMatrix)*objectNormal);');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + (one ? '#define SF_ONE\n' : '') + (walled ? '#define SF_WALL\n' : '') + (kind === 'masonry' ? '#define SF_MASONRY\n' : '') + HEADER)
        .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb=sfFinish(diffuseColor.rgb);')
        // the scan's normal, from world axes into view space, bends the surface normal
        .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nnormal=normalize(normal+mat3(viewMatrix)*sfNormal);')
        .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor=clamp(roughnessFactor+sfRough*.8,.08,1.);');
    };
    const key = std.customProgramCacheKey.bind(std);
    std.customProgramCacheKey = () => key() + '|finish5-' + kind + (one ? '-one' : '');
    std.needsUpdate = true;
  }
}

/** Building glass reflects the sky environment, not the street reflection probe. The probe is captured at the car for
 *  its paint and the wet road; seen in a window from the street it is close to black (measured 2026-11-12: panes at sRGB
 *  (5,7,15) on the probe, a blue-grey sky reflection with the room behind it on the sky environment), and a window with
 *  no reflection is a hole in the wall. The viewer assigns the probe to every reflective material from several places,
 *  so the material itself declines it: its envMap stays empty and three.js falls back to the scene's environment. */
export function reflectSky(m: THREE.Material) {
  if (!(m instanceof THREE.MeshStandardMaterial) || m.userData.skyReflected) return;
  m.userData.skyReflected = true; m.envMap = null;
  Object.defineProperty(m, 'envMap', {configurable: true, get: () => null, set: () => {}});
  m.needsUpdate = true;
}

/** The mid-distance tier's window glass. An insulated unit has four glass-air surfaces and reflects about 14 % of what is
 *  in front of it head-on (a single surface: 4 %), rising to a mirror at a glancing angle; from the street that reflection
 *  of sky and the buildings opposite is most of what a window is by day. The material is marked as glazing so the
 *  sky-visibility shading (occlusion.ts) treats it as a mirror of the open street rather than as a wall in a canyon. */
export function finishMidGlass(m: THREE.Material) {
  if (!(m instanceof THREE.MeshStandardMaterial) || m.userData.glazed || !/glazing/.test(m.name)) return false;
  m.userData.glazed = true; m.userData.envIntensity = 1.3;
  const previous = m.onBeforeCompile.bind(m);
  m.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer);
    shader.fragmentShader = shader.fragmentShader.replace('#include <lights_physical_fragment>',
      THREE.ShaderChunk.lights_physical_fragment.replace('mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor )', 'mix( vec3( 0.14 ), diffuseColor.rgb, metalnessFactor )'));
  };
  const key = m.customProgramCacheKey.bind(m);
  m.customProgramCacheKey = () => key() + '|midglass1';
  m.needsUpdate = true; reflectSky(m);
  return true;
}
