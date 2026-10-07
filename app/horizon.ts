import * as THREE from 'three';

// The backdrop around a model area (built offline by work/build_horizon.py into public/horizon/): the real ground and water
// round the model out to 60 km, from USGS elevation and lowered for the earth's curvature: the bay and the ocean, the East
// Bay hills, Marin and Mount Tamalpais, San Bruno Mountain, Mount Diablo. Nothing here is invented.
//
// San Francisco's own land beyond the model is the project's to build: it is plain ground at its real height (so it hides
// what it should hide), a neutral grey-brown with nothing on it that fades into the haze within a kilometre or so of the
// model's edge, as a city's distance does; the hills beyond rise out of it.
// Ground north of 38 N (where the elevation tiles end) is left to the haze.
//
// Shading is generic: hill slopes by their slope and aspect (sun-facing slopes of dry summer grass, north-facing ones darker
// with woodland, steep ground bare), level land the same neutral grey-brown (the data says nothing of what stands on it).
// Water is one surface at the level the model draws its own water at, laid over the ground (sunk under it offline), so
// the shore is where they meet; it moves gently, mirrors the sky and catches the sun. Haze is the scene's (atmosphere.ts).

type Meta = { angles: number; rings: number; center: { lat: number; lon: number }; innerRadiusM: number; waterLevelLocal: number; earthRadiusEffM: number; bin: string; sfMask: { bounds: number[]; shape: number[] }; arrays: Record<string, { offset: number; dtype: string; shape: number[] }> };
type Entry = { id: string; center: { lat: number; lon: number }; meta: string };

async function bytes(url: string) {
  const r = await fetch(url); if (!r.ok || !r.body) throw new Error('backdrop download failed');
  return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
function view(buf: ArrayBuffer, m: Meta, key: string) {
  const a = m.arrays[key]; const n = a.shape.reduce((s, v) => s * v, 1);
  switch (a.dtype) {
    case 'float32': return new Float32Array(buf, a.offset, n);
    case 'uint8': return new Uint8Array(buf, a.offset, n);
    case 'int8': return new Int8Array(buf, a.offset, n);
    default: throw new Error('backdrop dtype ' + a.dtype);
  }
}

const VERT = /* glsl */`
attribute float hzWater;attribute float hzSlope;attribute float hzNorth;
varying float vHzWater;varying float vHzSlope;varying float vHzNorth;varying vec3 vHzPos;
`;
const COMMON = /* glsl */`
uniform sampler2D uHzMask;uniform vec4 uHzMaskBox;uniform float uHzInner;uniform float uHzTime;uniform vec3 uHzHaze;
varying float vHzWater;varying float vHzSlope;varying float vHzNorth;varying vec3 vHzPos;
float hzHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float hzNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hzHash(i),hzHash(i+vec2(1.,0.)),f.x),mix(hzHash(i+vec2(0.,1.)),hzHash(i+vec2(1.,1.)),f.x),f.y);}
// the model's own disc and ground with no elevation data are not drawn
bool hzHidden(){vec2 en=vec2(vHzPos.x,-vHzPos.z);return dot(en,en)<uHzInner*uHzInner||vHzWater>1.5;}
float hzInSF(){
 vec2 en=vec2(vHzPos.x,-vHzPos.z);vec2 uv=vec2((en.x-uHzMaskBox.x)/(uHzMaskBox.z-uHzMaskBox.x),(en.y-uHzMaskBox.y)/(uHzMaskBox.w-uHzMaskBox.y));
 return (uv.x>0.&&uv.y>0.&&uv.x<1.&&uv.y<1.)?smoothstep(.35,.65,texture2D(uHzMask,uv).r):0.;
}
`;
const LAND = /* glsl */`
vec3 hzLand(){
 vec2 p=vHzPos.xz;float n=hzNoise(p*.004)*.6+hzNoise(p*.019)*.4;
 vec3 grass=mix(vec3(.27,.21,.11),vec3(.33,.27,.16),n);         // dry summer grass, the hills' colour from June to the rains
 vec3 wood=mix(vec3(.04,.055,.03),vec3(.065,.075,.04),n);       // oak, bay and eucalyptus woodland on shaded slopes
 vec3 scrub=mix(vec3(.09,.1,.06),vec3(.13,.13,.08),n);          // coastal scrub and chaparral between them
 vec3 bare=vec3(.21,.19,.16);
 vec3 level=mix(vec3(.12,.115,.105),vec3(.15,.142,.13),n);       // level land, and San Francisco's: nothing said of what stands on it
 float north=clamp(vHzNorth,0.,1.);
 vec3 c=mix(grass,scrub,smoothstep(.3,.75,n)*.55);
 c=mix(c,wood,smoothstep(.1,.55,north)*smoothstep(5.,14.,vHzSlope));
 c=mix(c,bare,smoothstep(32.,45.,vHzSlope));
 c=mix(level,c,smoothstep(2.5,7.,vHzSlope));
 return mix(c,level,hzInSF());
}
`;
const WATER = /* glsl */`
// gentle moving swell from a few drifting wave trains and noise (a slope field: no geometry moves)
vec2 hzWaveSlope(vec2 p){
 vec2 s=vec2(0.);float t=uHzTime;
 vec2 d1=normalize(vec2(1.,.35)),d2=normalize(vec2(-.4,1.)),d3=normalize(vec2(.8,-.6));
 s+=d1*cos(dot(p,d1)*.09+t*1.1)*.03;s+=d2*cos(dot(p,d2)*.21+t*1.7)*.024;s+=d3*cos(dot(p,d3)*.53+t*2.6)*.018;
 s+=(vec2(hzNoise(p*.07+t*.05),hzNoise(p*.07+vec2(7.,3.)-t*.05))-.5)*.06;
 s+=(vec2(hzNoise(p*.9+t*.4),hzNoise(p*.9+vec2(3.,9.)-t*.4))-.5)*.05;     // chop, a metre or so
 // the swell's slopes average out with distance (a pixel covers many waves): the far bay is a calmer mirror
 return s*clamp(400./max(length(p-cameraPosition.xz),1.),.15,1.);
}
`;

function polar(meta: Meta, height: (i: number, j: number) => number, attrs: Record<string, ArrayLike<number>>, step = 1) {
  // every `step`-th ring and angle of the grid (the water is flat but for the earth's curve: a coarse grid does)
  const A0 = meta.angles, radii0 = attrs.radii as Float32Array, rings: number[] = [];
  for (let i = 0; i < meta.rings; i += step) rings.push(i); if (rings[rings.length - 1] !== meta.rings - 1) rings.push(meta.rings - 1);
  const A = Math.floor(A0 / step), Rn = rings.length;
  const pos = new Float32Array(Rn * A * 3), extra: Record<string, Float32Array> = {};
  for (const k of Object.keys(attrs)) if (k !== 'radii') extra[k] = new Float32Array(Rn * A);
  for (let ii = 0; ii < Rn; ii++) for (let jj = 0; jj < A; jj++) {
    const i = rings[ii], j = jj * step, k = ii * A + jj, t = j / A0 * Math.PI * 2, r = radii0[i];
    pos[3 * k] = r * Math.cos(t); pos[3 * k + 1] = height(i, j); pos[3 * k + 2] = -r * Math.sin(t);
    for (const [name, arr] of Object.entries(extra)) arr[k] = (attrs[name] as ArrayLike<number>)[i * A0 + j];
  }
  const idx = new Uint32Array((Rn - 1) * A * 6); let q = 0;
  for (let i = 0; i < Rn - 1; i++) for (let j = 0; j < A; j++) {
    const a = i * A + j, b = i * A + (j + 1) % A, c = (i + 1) * A + j, d = (i + 1) * A + (j + 1) % A;
    idx[q++] = a; idx[q++] = c; idx[q++] = b; idx[q++] = b; idx[q++] = c; idx[q++] = d;   // counter-clockwise seen from above
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
  for (const [k, v] of Object.entries(extra)) g.setAttribute(k, new THREE.BufferAttribute(v, 1));
  g.computeVertexNormals(); g.computeBoundingSphere(); return g;
}

function patch(mat: THREE.MeshStandardMaterial, uniforms: Record<string, THREE.IUniform>, frag: string, edits: [string, string][], key: string) {
  const previous = mat.onBeforeCompile.bind(mat);
  mat.onBeforeCompile = (shader, renderer) => {
    previous(shader, renderer); Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\n' + VERT)
      .replace('#include <project_vertex>', '#include <project_vertex>\nvHzWater=hzWater;vHzSlope=hzSlope;vHzNorth=hzNorth;vHzPos=(modelMatrix*vec4(transformed,1.)).xyz;');
    let f = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + COMMON + frag).replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\nif(hzHidden())discard;');
    for (const [a, b] of edits) f = f.replace(a, b);
    shader.fragmentShader = f;
  };
  const k = mat.customProgramCacheKey.bind(mat); mat.customProgramCacheKey = () => k() + '|' + key;
}

export async function loadHorizon(center: { lat: number; lon: number }, setup: (m: THREE.Material) => void, haze: THREE.Color, skyEnvironment: () => THREE.Texture | null) {
  const index = await fetch('/horizon/index.json', { cache: 'no-cache' }).then(r => r.ok ? r.json() as Promise<{ areas: Entry[] }> : null).catch(() => null);
  const near = (e: Entry) => Math.hypot((e.center.lat - center.lat) * 111320, (e.center.lon - center.lon) * 111320 * Math.cos(center.lat * Math.PI / 180)) < 30;
  const entry = index?.areas?.find(near); if (!entry) return null;
  const meta = await fetch(entry.meta, { cache: 'no-cache' }).then(r => r.json() as Promise<Meta>);
  const buf = await bytes(entry.meta.replace(/[^/]*$/, '') + meta.bin);
  const group = new THREE.Group(); group.name = 'horizon';
  const A = meta.angles, radii = view(buf, meta, 'terrain.radii') as Float32Array, height = view(buf, meta, 'terrain.height') as Float32Array;
  const water = view(buf, meta, 'terrain.water') as Uint8Array, slope = view(buf, meta, 'terrain.slope') as Uint8Array, north = Array.from(view(buf, meta, 'terrain.north') as Int8Array, v => v / 127);
  // San Francisco's mainland as a mask texture (row 0 is the north edge)
  const [mh, mw] = meta.sfMask.shape, mask = view(buf, meta, 'sf.mask') as Uint8Array, flipped = new Uint8Array(mh * mw);
  for (let r = 0; r < mh; r++) flipped.set(mask.subarray(r * mw, (r + 1) * mw), (mh - 1 - r) * mw);
  const maskTex = new THREE.DataTexture(flipped, mw, mh, THREE.RedFormat, THREE.UnsignedByteType); maskTex.magFilter = maskTex.minFilter = THREE.LinearFilter; maskTex.needsUpdate = true;
  const b = meta.sfMask.bounds;
  const uniforms = { uHzMask: { value: maskTex }, uHzMaskBox: { value: new THREE.Vector4(b[0], b[1], b[2], b[3]) }, uHzInner: { value: meta.innerRadiusM }, uHzTime: { value: 0 }, uHzHaze: { value: haze } };
  // --- the ground (sunk under the water where the bay and the ocean are)
  const landWater = Array.from(water, v => v === 2 ? 2 : 0);
  const tg = polar(meta, (i, j) => height[i * A + j], { radii, hzWater: landWater, hzSlope: slope, hzNorth: north });
  const land = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: .95, metalness: 0 }); land.name = 'horizon_ground';
  // pushed back in depth: where it meets the model's own ground and water at the edge, the model draws
  land.polygonOffset = true; land.polygonOffsetFactor = 2; land.polygonOffsetUnits = 8;
  setup(land);
  patch(land, uniforms, LAND, [['#include <map_fragment>', '#include <map_fragment>\ndiffuseColor.rgb=hzLand();'],
    // San Francisco's land fades into the haze over about a kilometre from the model's edge
    ['#include <fog_fragment>', '#include <fog_fragment>\n{float e=length(vHzPos.xz)-uHzInner;gl_FragColor.rgb=mix(gl_FragColor.rgb,uHzHaze*.92,hzInSF()*(1.-exp(-max(e,0.)/650.)));}']], 'horizon-ground');
  // --- the water: one surface at the model's water level, curving with the earth
  const R = meta.earthRadiusEffM, lvl = meta.waterLevelLocal;
  const wg = polar(meta, i => lvl - radii[i] * radii[i] / (2 * R), { radii, hzWater: Array.from(water, v => v === 2 ? 2 : 1), hzSlope: new Uint8Array(water.length), hzNorth: new Uint8Array(water.length) }, 4);
  const sea = new THREE.MeshStandardMaterial({ color: new THREE.Color(.018, .032, .035), roughness: .06, metalness: 0 }); sea.name = 'horizon_water';
  sea.polygonOffset = true; sea.polygonOffsetFactor = 2; sea.polygonOffsetUnits = 8;
  setup(sea);
  patch(sea, uniforms, WATER, [['#include <normal_fragment_maps>', '#include <normal_fragment_maps>\n{vec2 ws=hzWaveSlope(vHzPos.xz);normal=normalize((viewMatrix*vec4(normalize(vec3(-ws.x,1.,-ws.y)),0.)).xyz);}']], 'horizon-water');
  const ground = new THREE.Mesh(tg, land); ground.name = 'horizon___ground';
  const waterMesh = new THREE.Mesh(wg, sea); waterMesh.name = 'horizon___water';
  for (const m of [ground, waterMesh]) { m.receiveShadow = false; m.castShadow = false; m.frustumCulled = false; group.add(m); }
  group.traverse(o => { o.userData.layer = 'horizon'; });
  return {
    group, materials: [land, sea],
    update(now: number) { uniforms.uHzTime.value = now / 1000; const env = skyEnvironment(); if (env && sea.envMap !== env) { sea.envMap = env; sea.envMapIntensity = 1; sea.needsUpdate = true; } },
    dispose() { tg.dispose(); wg.dispose(); land.dispose(); sea.dispose(); maskTex.dispose(); },
  };
}
