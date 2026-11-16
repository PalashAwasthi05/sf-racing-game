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
float H(vec2 xz){vec2 uv=vec2(xz.x,-xz.y)/(2.*uR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return -1e4;return textureLod(uHeight,uv,0.).r;}
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
  const height = new THREE.WebGLRenderTarget(N, N, {type: THREE.FloatType, format: THREE.RedFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true});
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

// Four sloped albedo and depth captures feed a height- and direction-dependent
// indirect light field. Only changes in the sun require another field bake.
const ALBEDO_VERTEX = /* glsl */`varying vec2 vUv;varying float vBounceDepth;varying vec3 vBounceAlbedo;uniform vec2 uSide,uForward;uniform float uSlope,uProjR;uniform vec3 uView;void main(){vUv=uv;
#if defined(USE_COLOR) || defined(USE_COLOR_ALPHA)
 vBounceAlbedo=color.rgb;
#else
 vBounceAlbedo=vec3(1.);
#endif
 vec3 w=(modelMatrix*vec4(position,1.)).xyz;float side=dot(w.xz,uSide);float up=w.y-uSlope*dot(w.xz,uForward);vBounceDepth=dot(w,uView);gl_Position=vec4(side/uProjR,up/uProjR,-vBounceDepth/4000.,1.);}`;
const ALBEDO_FRAGMENT = /* glsl */`varying vec2 vUv;varying float vBounceDepth;varying vec3 vBounceAlbedo;uniform vec3 uColor;uniform sampler2D uMap;uniform mat3 uMapTransform;uniform float uHasMap;uniform float uAlphaTest;void main(){vec4 c=vec4(uColor*vBounceAlbedo,1.);if(uHasMap>.5)c*=texture2D(uMap,(uMapTransform*vec3(vUv,1.)).xy);if(c.a<uAlphaTest)discard;gl_FragColor=vec4(c.rgb,.5+vBounceDepth/4000.);}`;

const BOUNCE_BAKE_FRAGMENT = /* glsl */`
precision highp float;
uniform sampler2D uAtlas,uHeight;uniform float uR,uAtlasR,uSlope,uSunEnergy;uniform vec3 uSunDir,uSunColor;
uniform vec4 uLevels;varying vec2 vUv;
vec2 packUv(vec2 uv,int i){return uv*.5+vec2(float(i-2*(i/2))*.5,float(i/2)*.5);}
vec2 atlasUv(vec3 p,vec2 d){vec2 side=vec2(d.y,-d.x);return vec2(.5+dot(p.xz,side)/(2.*uAtlasR),.5+(p.y-uSlope*dot(p.xz,d))/(2.*uAtlasR));}
float atlasDepth(vec4 sampleColor,vec3 expected,vec3 viewDir){float actual=(sampleColor.a-.5)*4000.;return 1.-smoothstep(5.,20.,abs(actual-dot(expected,viewDir)));}
float sourceVisibility(vec2 xz,float y,vec2 d,float radius){float v=1.;for(int k=1;k<=3;k++){float t=float(k)/4.;vec2 q=xz+d*radius*t;vec2 uv=vec2(q.x,-q.y)/(2.*uR)+.5;if(uv.x>.002&&uv.y>.002&&uv.x<.998&&uv.y<.998){float h=textureLod(uHeight,uv,0.).r;if(h>y+1.35)v*=.3;}}return v;}
vec3 fetchSource(vec2 xz,float y,vec2 d,int dir,float radius){vec2 sampleXZ=xz+d*radius;vec3 p=vec3(sampleXZ.x,y,sampleXZ.y);vec3 viewDir=normalize(vec3(d.x,uSlope,d.y));vec2 local=atlasUv(p,d);if(local.x<.01||local.y<.01||local.x>.99||local.y>.99)return vec3(0.);vec2 uv=packUv(local,dir);vec4 s=textureLod(uAtlas,uv,0.);float depth=atlasDepth(s,p,viewDir);float vis=sourceVisibility(xz,y,d,radius);float sun=.16+.84*max(dot(viewDir,uSunDir),0.)*clamp(uSunEnergy/4.,0.,1.);return s.rgb*depth*vis*sun*mix(vec3(1.),uSunColor,.08*clamp(uSunEnergy/4.,0.,1.));}
void main(){
 vec2 tile=floor(gl_FragCoord.xy/128.);vec2 local=fract(gl_FragCoord.xy/128.);float lob=tile.x;float level=tile.y;
 vec2 xz=vec2((local.x-.5)*2.*uR,(.5-local.y)*2.*uR);vec2 hUv=vec2(xz.x,-xz.y)/(2.*uR)+.5;float base=texture2D(uHeight,hUv).r;
 float y=base+(level<.5?uLevels.x:level<1.5?uLevels.y:level<2.5?uLevels.z:uLevels.w);
 vec2 dirs[4];dirs[0]=vec2(1.,0.);dirs[1]=vec2(-1.,0.);dirs[2]=vec2(0.,1.);dirs[3]=vec2(0.,-1.);
 vec3 total=vec3(0.);float weight=0.;
 for(int i=0;i<4;i++){
  if(lob>3.5||abs(lob-float(i))<.5){vec2 d=dirs[i];vec3 v=fetchSource(xz,y,d,i,4.)*.62+fetchSource(xz,y,d,i,8.)*.38;float w=lob>3.5?.25:1.;total+=v*w;weight+=w;}
 }
 vec3 c=weight>0.?total/weight:vec3(0.);gl_FragColor=vec4(c,1.);
}`;

/** Capture four sloped, directional views of actual surface albedo for a local bounce cue. */
export function buildSurfaceBounce(renderer: THREE.WebGLRenderer, scene: THREE.Scene, surfaces: THREE.Mesh[], hide: THREE.Object3D[], reach: number, heightTexture: THREE.Texture | null = null) {
  const R = reach, ATLAS_R = R * 1.5, N = 512, ATLAS = N * 2, GRID = 128, FIELD_W = GRID * 5, FIELD_H = GRID * 4, slope = .42;
  let atlas: THREE.WebGLRenderTarget | null = null, field: THREE.WebGLRenderTarget | null = null;
  const swaps: {mesh: THREE.Mesh; material: THREE.Material | THREE.Material[]}[] = [];
  const was = new Map<THREE.Object3D, boolean>();
  const hidden = new Set(hide), allowed = new Set(surfaces);
  const previous = {override: scene.overrideMaterial, background: scene.background, fog: scene.fog, target: renderer.getRenderTarget(), auto: renderer.shadowMap.autoUpdate, autoClear: renderer.autoClear, color: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha()};
  const cam = new THREE.OrthographicCamera(-ATLAS_R, ATLAS_R, ATLAS_R, -ATLAS_R, 1, 4000);
  cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const projections = {uSide: {value: new THREE.Vector2()}, uForward: {value: new THREE.Vector2()}, uSlope: {value: slope}, uProjR: {value: ATLAS_R}, uView: {value: new THREE.Vector3()}};
  const cache = new Map<THREE.Material, THREE.ShaderMaterial>();
  const sunDirection = new THREE.Vector3(0,1,0), sunColor = new THREE.Color(0xffffff), levels = new THREE.Vector4(...LEVELS);
  const bakeUniforms = {uAtlas:{value:null as THREE.Texture|null},uHeight:{value:heightTexture},uR:{value:R},uAtlasR:{value:ATLAS_R},uSlope:{value:slope},uSunEnergy:{value:0},uSunDir:{value:sunDirection},uSunColor:{value:sunColor},uLevels:{value:levels}};
  const bakeMaterial = new THREE.ShaderMaterial({vertexShader:'void main(){gl_Position=vec4(position.xy,0.,1.);}',fragmentShader:BOUNCE_BAKE_FRAGMENT,uniforms:bakeUniforms,depthTest:false,depthWrite:false,toneMapped:false});
  const bakeScene = new THREE.Scene(), bakeQuad = new THREE.Mesh(new THREE.PlaneGeometry(2,2),bakeMaterial); bakeScene.add(bakeQuad);
  const bakeCamera = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  const material = (source: THREE.Material) => {
    let result = cache.get(source); if (result) return result;
    const s = source as THREE.MeshStandardMaterial & {map?: THREE.Texture | null; alphaTest?: number; transparent?: boolean};
    if (s.map?.matrixAutoUpdate) s.map.updateMatrix();
    result = new THREE.ShaderMaterial({vertexShader: ALBEDO_VERTEX, fragmentShader: ALBEDO_FRAGMENT,
      uniforms: {...projections, uColor: {value: (s.color ?? new THREE.Color(0xffffff)).clone()}, uMap: {value: s.map ?? null}, uMapTransform: {value: s.map?.matrix ?? new THREE.Matrix3()}, uHasMap: {value: s.map ? 1 : 0}, uAlphaTest: {value: Math.max(s.alphaTest ?? 0, s.transparent ? .5 : 0)}},
      side: source.side, vertexColors:s.vertexColors, depthWrite: true, depthTest: true, toneMapped: false});
    cache.set(source, result); return result;
  };
  try {
    const keep = new Set<THREE.Object3D>();
    for (const mesh of surfaces) for (let o: THREE.Object3D | null = mesh; o; o = o.parent) keep.add(o);
    scene.traverse(o => {
      was.set(o, o.visible);
      if (hidden.has(o) || ((o as THREE.Mesh).isMesh && !allowed.has(o as THREE.Mesh)) || (!keep.has(o) && o !== scene)) o.visible = false;
    });
    for (const mesh of surfaces) {
      if (!mesh.visible) continue;
      const original = mesh.material; swaps.push({mesh, material: original});
      mesh.material = Array.isArray(original) ? original.map(material) : material(original);
    }
    scene.overrideMaterial = null; scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false;
    renderer.setClearColor(0x000000, 0);
    atlas = new THREE.WebGLRenderTarget(ATLAS,ATLAS,{type:THREE.HalfFloatType,format:THREE.RGBAFormat,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:true});
    renderer.autoClear=false;
    const directions = [new THREE.Vector2(1,0),new THREE.Vector2(-1,0),new THREE.Vector2(0,1),new THREE.Vector2(0,-1)];
    for (const [i,d] of directions.entries()) {
      const view = new THREE.Vector3(d.x,slope,d.y).normalize();
      projections.uSide.value.set(d.y,-d.x); projections.uForward.value.copy(d); projections.uView.value.copy(view);
      const x=(i%2)*N,y=Math.floor(i/2)*N;
      atlas.viewport.set(x,y,N,N);atlas.scissor.set(x,y,N,N);atlas.scissorTest=true;renderer.setRenderTarget(atlas);renderer.clear(true,true,true);renderer.render(scene,cam);
    }
    field = new THREE.WebGLRenderTarget(FIELD_W,FIELD_H,{type:THREE.HalfFloatType,format:THREE.RGBAFormat,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter,depthBuffer:false});
    bakeUniforms.uAtlas.value=atlas.texture;
  } catch {
    atlas?.dispose(); field?.dispose(); atlas=null;field=null;
  } finally {
    for (const {mesh, material: original} of swaps) mesh.material = original;
    for (const [o, visible] of was) o.visible = visible;
    scene.overrideMaterial = previous.override; scene.background = previous.background; scene.fog = previous.fog;
    renderer.shadowMap.autoUpdate = previous.auto; renderer.autoClear=previous.autoClear; renderer.setClearColor(previous.color, previous.alpha); renderer.setRenderTarget(previous.target);
    for (const m of cache.values()) m.dispose();
  }
  if (!atlas || !field) {bakeQuad.geometry.dispose();bakeMaterial.dispose();return null;}
  // The color field is one 2D texture: five normal lobes by four local heights.
  const uniforms = {uSoBounce:{value:field.texture},uSoBounceR:{value:R},uSoBounceEnabled:{value:0}};
  let lastEnergy=-1;const lastDir=new THREE.Vector3(9,9,9),lastColor=new THREE.Color(9,9,9);
  function bake(){
    if(!atlas||!field||!bakeUniforms.uHeight.value)return;
    const previousTarget=renderer.getRenderTarget(),previousAuto=renderer.autoClear;renderer.autoClear=false;
    try{renderer.setRenderTarget(field);renderer.clear(true,true,true);renderer.render(bakeScene,bakeCamera);}finally{renderer.setRenderTarget(previousTarget);renderer.autoClear=previousAuto;}
  }
  return {texture:field.texture,uniforms,atlas,field,
    setHeightTexture(texture:THREE.Texture){bakeUniforms.uHeight.value=texture;if(lastEnergy>=.25){bake();uniforms.uSoBounceEnabled.value=1;}},
    setSun(direction:THREE.Vector3,color:THREE.Color,intensity:number){
      if(lastEnergy>=0&&lastDir.distanceToSquared(direction)<1e-8&&Math.abs(lastColor.r-color.r)<.001&&Math.abs(lastColor.g-color.g)<.001&&Math.abs(lastColor.b-color.b)<.001&&Math.abs(lastEnergy-intensity)<.005)return;
      lastDir.copy(direction);lastColor.copy(color);lastEnergy=intensity;sunDirection.copy(direction).normalize();sunColor.copy(color);bakeUniforms.uSunEnergy.value=Math.max(0,intensity);
      if(intensity<.25||!bakeUniforms.uHeight.value){uniforms.uSoBounceEnabled.value=0;return;}bake();uniforms.uSoBounceEnabled.value=1;
    },
    dispose(){atlas?.dispose();field?.dispose();bakeQuad.geometry.dispose();bakeMaterial.dispose();}};
}
export type SurfaceBounce = NonNullable<ReturnType<typeof buildSurfaceBounce>>;

const HEADER = /* glsl */`
uniform sampler2D uSoHeight;uniform sampler2D uSoVis;uniform float uSoR;uniform vec4 uSoLevels;uniform float uSoStrength;
varying vec3 vSoPos;varying vec3 vSoNrm;
float soSkyVisibility(out float h){
 h=0.;vec3 N=normalize(vSoNrm);
 // a wall reads the sky its street sees: the point is taken 1.2 m out from the wall, clear of the building's own roof
 vec2 xz=vSoPos.xz+normalize(N.xz+vec2(1e-5))*1.2*(1.-abs(N.y));
 vec2 uv=vec2(xz.x,-xz.y)/(2.*uSoR)+.5;if(uv.x<0.||uv.y<0.||uv.x>1.||uv.y>1.)return 1.;
 float top=textureLod(uSoHeight,uv,0.).r;
 h=max(vSoPos.y-top,0.);
 vec4 v=textureLod(uSoVis,uv,0.);
 float s=h<uSoLevels.y?mix(v.x,v.y,clamp((h-uSoLevels.x)/(uSoLevels.y-uSoLevels.x),0.,1.)):
         h<uSoLevels.z?mix(v.y,v.z,(h-uSoLevels.y)/(uSoLevels.z-uSoLevels.y)):
         mix(v.z,v.w,clamp((h-uSoLevels.z)/(uSoLevels.w-uSoLevels.z),0.,1.));
 // light from the sky falls on what faces up; a wall's lower half sees the street and the facades opposite, which
 // stay lit, so a wall takes about half the occlusion a floor does
 float up=clamp(.5+.5*N.y,0.,1.);
 return mix(1.,s,uSoStrength*mix(.55,1.,up));
}
`;

const BOUNCE_HEADER = /* glsl */`
uniform sampler2D uSoBounce;uniform float uSoBounceR,uSoBounceEnabled;
vec3 soBounceFetch(vec2 local,int lobe,int slice){vec2 p=clamp(local,vec2(.004),vec2(.996));return textureLod(uSoBounce,vec2((float(lobe)+p.x)/5.,(float(slice)+p.y)/4.),0.).rgb;}
vec3 soBounceTint(float skyVis,float h){
 if(uSoBounceEnabled<.5)return vec3(1.);
 vec2 local=vec2(vSoPos.x,-vSoPos.z)/(2.*uSoBounceR)+.5;if(local.x<.002||local.y<.002||local.x>.998||local.y>.998)return vec3(1.);
 float lo=0.,hi=1.,t=0.;
 if(h<4.){lo=0.;hi=1.;t=clamp((h-uSoLevels.x)/(uSoLevels.y-uSoLevels.x),0.,1.);}
 else if(h<12.){lo=1.;hi=2.;t=clamp((h-uSoLevels.y)/(uSoLevels.z-uSoLevels.y),0.,1.);}
 else if(h<35.){lo=2.;hi=3.;t=clamp((h-uSoLevels.z)/(uSoLevels.w-uSoLevels.z),0.,1.);}
 else {lo=3.;hi=3.;}
 vec3 n=normalize(vSoNrm);int side=abs(n.x)>abs(n.z)?(n.x>=0.?0:1):(n.z>=0.?2:3);
 vec3 horizontal=mix(soBounceFetch(local,side,int(lo)),soBounceFetch(local,side,int(hi)),t);
 vec3 upper=mix(soBounceFetch(local,4,int(lo)),soBounceFetch(local,4,int(hi)),t);
 float up=clamp(n.y,0.,1.);vec3 c=mix(horizontal,upper,up);float lum=dot(c,vec3(.2126,.7152,.0722));if(lum<.025)return vec3(1.);
 vec3 chroma=clamp(c/max(lum,.04),.68,1.32);float amplitude=clamp(.78+1.65*lum,.8,1.38);
 return mix(vec3(1.),chroma,.2)*amplitude*mix(.92,1.08,skyVis);
}
`;

/** Scales each material's sky and surroundings light by the sky visibility at its fragment, and writes the indirect
 *  share of its light into the alpha channel for the detail-scale occlusion composite. Chained after any existing
 *  onBeforeCompile. */
export function occludeMaterials(materials: Iterable<THREE.Material>, sky: SkyVisibility | null, bounce: SurfaceBounce | null = null) {
  for (const m of materials) {
    if (!(m instanceof THREE.MeshStandardMaterial) || m.userData.occluded) continue; m.userData.occluded = true;
    // glazing is a mirror of the open street and sky in front of it: neither its own reveals nor the street's horizon
    // occlude what it reflects, so it takes no occlusion at either scale (alpha 0)
    const glass = !!m.userData.glazed, useSky = sky && !glass, useBounce = bounce && sky && !glass;
    const previous = m.onBeforeCompile.bind(m);
    m.onBeforeCompile = (shader, renderer) => {
      previous(shader, renderer);
      if (useSky) {
        Object.assign(shader.uniforms, sky!.uniforms);
        // Both effects sample the same measured roof-height texture. Sharing
        // its GLSL sampler leaves capacity for the local lighting atlas on
        // WebGL2 hardware with sixteen fragment texture units.
        if(shader.uniforms.uWxHeight?.value===sky!.height.texture){
          shader.fragmentShader=shader.fragmentShader.replace('uniform sampler2D uWxHeight;', '').replace(/\buWxHeight\b/g,'uSoHeight');
          delete shader.uniforms.uWxHeight;
        }
      }
      if (useBounce) Object.assign(shader.uniforms, bounce!.uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vSoPos;varying vec3 vSoNrm;')
        .replace('#include <project_vertex>', '#include <project_vertex>\nvSoPos=cameraPosition+mvPosition.xyz*mat3(viewMatrix);vSoNrm=normalize(transformedNormal*mat3(viewMatrix));');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\n' + (useSky ? HEADER : 'varying vec3 vSoPos;varying vec3 vSoNrm;\n') + (useBounce ? BOUNCE_HEADER : ''))
        .replace('#include <aomap_fragment>', '#include <aomap_fragment>\n' + (useSky ? '{float soH=0.;float soV=soSkyVisibility(soH);reflectedLight.indirectDiffuse*=soV;reflectedLight.indirectSpecular*=mix(1.,soV,.8);' + (useBounce ? 'reflectedLight.indirectDiffuse*=soBounceTint(soV,soH);' : '') + '}' : ''))
        .replace('#include <opaque_fragment>', `#include <opaque_fragment>
#ifdef OPAQUE
{vec3 lw=vec3(.2126,.7152,.0722);float dl=dot(reflectedLight.directDiffuse+reflectedLight.directSpecular,lw);float il=dot(reflectedLight.indirectDiffuse+reflectedLight.indirectSpecular,lw);
 gl_FragColor.a=${glass ? '0.' : 'clamp(il/max(dl+il+dot(totalEmissiveRadiance,lw),1e-5),0.,1.)'};}
#endif`);
    };
    const key = m.customProgramCacheKey.bind(m);
    m.customProgramCacheKey = () => key() + (useSky ? '|occluded' : glass ? '|glass-alpha' : '|indirect-alpha') + (useBounce ? '|surface-bounce' : '');
    m.needsUpdate = true;
  }
}
