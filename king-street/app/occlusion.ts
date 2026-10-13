import * as THREE from 'three';

// Occlusion of the light a surface receives from the sky and its surroundings, at two scales.
//
// Street scale: how much of the sky each point sees. A narrow street's floor, the foot of a wall and a courtyard see a
// slice of sky between buildings; a roof or an open plaza sees all of it. The static world is drawn once from above into
// a height map; from it, for every spot and four heights above the surface there, the visible share of the sky is found
// by marching sixteen directions out to 130 m (a cosine-weighted horizon). Each material then scales its sky and
// surroundings light (never the sun) by the visibility at its fragment: canyon floors and building bases darken, upper
// floors and roofs stay open, as real streets are lit.
//
// Detail scale: the screen-space ambient occlusion pass (inner corners, under ledges, the foot of a kerb) used to multiply
// the whole lit image, sunlight included, which drew dark outlines along every sunlit kerb and window reveal. Each
// material now writes the share of its light that is indirect into the scene target's alpha; the AO is applied to that
// share only (the composite pass in grade.ts).
//
// Nothing here is location specific.

const LEVELS = [.5, 4, 12, 35];          // heights above the local surface the visibility is found at (m)
const DIRS = 16, STEPS = 20;

const HEIGHT_VERTEX = /* glsl */`varying float vY;void main(){vec4 w=modelMatrix*vec4(position,1.);vY=w.y;gl_Position=projectionMatrix*viewMatrix*w;}`;
const HEIGHT_FRAGMENT = /* glsl */`varying float vY;void main(){gl_FragColor=vec4(vY,0.,0.,1.);}`;

const VIS_FRAGMENT = /* glsl */`
precision highp float;
uniform sampler2D uHeight;uniform float uR;uniform vec4 uLevels;varying vec2 vUv;
float H(vec2 xz){vec2 uv=vec2(xz.x,-xz.y)/(2.*uR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return -1e4;return texture2D(uHeight,uv).r;}
void main(){
 vec2 P=vec2(vUv.x-.5,.5-vUv.y)*2.*uR;float base=H(P);
 vec4 tmax=vec4(0.);vec4 vis=vec4(0.);
 for(int d=0;d<${DIRS};d++){
  float a=(float(d)+.5)/${DIRS}.*6.2831853;vec2 dir=vec2(cos(a),sin(a));
  vec4 t=vec4(0.);float dist=1.5;
  for(int s=0;s<${STEPS};s++){
   float h=H(P+dir*dist);
   t=max(t,(vec4(h)-(vec4(base)+uLevels))/dist);
   dist*=1.25;
  }
  vis+=1./(1.+t*t);                          // cos^2 of the horizon elevation, per height level
 }
 gl_FragColor=vis/${DIRS}.;
}`;

/** Draws the static world from above into a height map and finds the sky visibility of every spot in the area. */
export function buildSkyVisibility(renderer: THREE.WebGLRenderer, scene: THREE.Scene, hide: THREE.Object3D[], reach: number) {
  const R = reach, N = 1024, V = 512;
  const height = new THREE.WebGLRenderTarget(N, N, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true});
  // looking straight down with north (-z) up: texture u = (x/R+1)/2, v = (-z/R+1)/2
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, 1, 4000); cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const mat = new THREE.ShaderMaterial({vertexShader: HEIGHT_VERTEX, fragmentShader: HEIGHT_FRAGMENT, side: THREE.DoubleSide});
  const was = hide.map(o => o.visible); hide.forEach(o => o.visible = false);
  const prevOverride = scene.overrideMaterial, prevBg = scene.background, prevFog = scene.fog, prevTarget = renderer.getRenderTarget(), prevAuto = renderer.shadowMap.autoUpdate, clear = renderer.getClearColor(new THREE.Color()), clearA = renderer.getClearAlpha();
  scene.overrideMaterial = mat; scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false;
  // a deep floor under everything stands for open ground where the area has no surface (a negative clear colour is not
  // reliable through colour management)
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8 * R, 8 * R), mat); floor.rotation.x = -Math.PI / 2; floor.position.y = -500; floor.updateMatrixWorld(true); scene.add(floor);
  renderer.setClearColor(0x000000, 1);
  renderer.setRenderTarget(height); renderer.clear(); renderer.render(scene, cam);
  scene.remove(floor); floor.geometry.dispose();
  scene.overrideMaterial = prevOverride; scene.background = prevBg; scene.fog = prevFog; renderer.shadowMap.autoUpdate = prevAuto; renderer.setClearColor(clear, clearA);
  hide.forEach((o, i) => o.visible = was[i]);
  // the visibility of each spot at four heights above its surface
  const vis = new THREE.WebGLRenderTarget(V, V, {type: THREE.HalfFloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false});
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
    vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}', fragmentShader: VIS_FRAGMENT,
    uniforms: {uHeight: {value: height.texture}, uR: {value: R}, uLevels: {value: new THREE.Vector4(...LEVELS)}}, depthTest: false, depthWrite: false}));
  const qs = new THREE.Scene(); qs.add(quad); const qc = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  renderer.setRenderTarget(vis); renderer.render(qs, qc); renderer.setRenderTarget(prevTarget);
  quad.geometry.dispose(); (quad.material as THREE.Material).dispose(); mat.dispose();
  // the height map stays: a fragment finds the surface its sky visibility is measured from
  const uniforms = {uSoHeight: {value: height.texture}, uSoVis: {value: vis.texture}, uSoR: {value: R}, uSoLevels: {value: new THREE.Vector4(...LEVELS)}, uSoStrength: {value: 1}};
  return {uniforms, height, vis, dispose() {height.dispose(); vis.dispose();}};
}
export type SkyVisibility = ReturnType<typeof buildSkyVisibility>;

const HEADER = /* glsl */`
uniform sampler2D uSoHeight;uniform sampler2D uSoVis;uniform float uSoR;uniform vec4 uSoLevels;uniform float uSoStrength;
varying vec3 vSoPos;varying vec3 vSoNrm;
float soSkyVisibility(){
 vec3 N=normalize(vSoNrm);
 // a wall reads the sky its street sees: the point is taken 1.2 m out from the wall, clear of the building's own roof
 vec2 xz=vSoPos.xz+normalize(N.xz+vec2(1e-5))*1.2*(1.-abs(N.y));
 vec2 uv=vec2(xz.x,-xz.y)/(2.*uSoR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return 1.;
 float top=texture2D(uSoHeight,uv).r;
 float h=max(vSoPos.y-top,0.);
 vec4 v=texture2D(uSoVis,uv);
 float s=h<uSoLevels.y?mix(v.x,v.y,clamp((h-uSoLevels.x)/(uSoLevels.y-uSoLevels.x),0.,1.)):
         h<uSoLevels.z?mix(v.y,v.z,(h-uSoLevels.y)/(uSoLevels.z-uSoLevels.y)):
         mix(v.z,v.w,clamp((h-uSoLevels.z)/(uSoLevels.w-uSoLevels.z),0.,1.));
 // light from the sky falls on what faces up; a wall's lower half sees the street and the facades opposite, which
 // stay lit, so a wall takes about half the occlusion a floor does
 float up=clamp(.5+.5*N.y,0.,1.);
 return mix(1.,s,uSoStrength*mix(.55,1.,up));
}
`;

/** Scales each material's sky and surroundings light by the sky visibility at its fragment, and writes the indirect
 *  share of its light into the alpha channel for the detail-scale occlusion composite. Chained after any existing
 *  onBeforeCompile. */
export function occludeMaterials(materials: Iterable<THREE.Material>, sky: SkyVisibility | null) {
  for (const m of materials) {
    if (!(m instanceof THREE.MeshStandardMaterial) || m.userData.occluded) continue; m.userData.occluded = true;
    // glazing is a mirror of the open street and sky in front of it: neither its own reveals nor the street's horizon
    // occlude what it reflects, so it takes no occlusion at either scale (alpha 0)
    const glass = !!m.userData.glazed, useSky = sky && !glass;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      if (useSky) Object.assign(shader.uniforms, sky!.uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vSoPos;varying vec3 vSoNrm;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvSoPos=(modelMatrix*vec4(transformed,1.)).xyz;vSoNrm=normalize(mat3(modelMatrix)*objectNormal);');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + (useSky ? HEADER : 'varying vec3 vSoPos;varying vec3 vSoNrm;\n'))
        .replace('#include <aomap_fragment>', '#include <aomap_fragment>\n' + (useSky ? '{float soV=soSkyVisibility();reflectedLight.indirectDiffuse*=soV;reflectedLight.indirectSpecular*=mix(1.,soV,.8);}' : ''))
        .replace('#include <opaque_fragment>', `#include <opaque_fragment>
#ifdef OPAQUE
{vec3 lw=vec3(.2126,.7152,.0722);float dl=dot(reflectedLight.directDiffuse+reflectedLight.directSpecular,lw);float il=dot(reflectedLight.indirectDiffuse+reflectedLight.indirectSpecular,lw);
 gl_FragColor.a=${glass ? '0.' : 'clamp(il/max(dl+il+dot(totalEmissiveRadiance,lw),1e-5),0.,1.)'};}
#endif`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + (useSky ? '|occluded' : glass ? '|glass-alpha' : '|indirect-alpha');
    m.needsUpdate = true;
  }
}
