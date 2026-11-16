import * as THREE from 'three';

// Real blades on the lawns near the camera. A lawn drawn as one flat green surface has no depth however it is textured;
// here every blade is a small curved, tapered strip standing on the lawn, leaning its own way and moving in the wind.
//  where   the lawns are drawn once from above into a map of their height; every other ground surface and building drawn
//          into the same depth buffer hides the lawn under it, so no blade grows through a path or a kerb.
//  which   a square grid of cells follows the camera, snapped to the cell size: a blade belongs to its cell in the world,
//          so it stays put as the camera moves. Each blade's place, size, lean and colour come from a hash of its cell.
//          Nothing is stored per blade and nothing is updated on the CPU but three uniforms a frame.
//  far     two tiers: fine blades near the camera, wider tufts beyond; each thins out and widens with distance and
//          shrinks to nothing at its edge, where the lawn's own shading (ground-detail.ts) takes over.
// Nothing here is location specific.

const FIELD_VERTEX = /* glsl */`varying float vY;void main(){vec4 w=modelMatrix*vec4(position,1.);vY=w.y;gl_Position=projectionMatrix*viewMatrix*w;}`;
const FIELD_FRAGMENT = /* glsl */`varying float vY;uniform float uLawn;void main(){gl_FragColor=vec4(vY*uLawn,uLawn,0.,1.);}`;

const HEADER = /* glsl */`
uniform sampler2D uGbField;uniform float uGbR,uGbCell,uGbGrid,uGbTime,uGbReach,uGbInner,uGbWidth;uniform vec3 uGbEye;
varying float vGbT;varying float vGbLong;varying vec3 vGbTint;
float gbHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
vec2 gbHash2(vec2 p){return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);}
float gbNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(gbHash(i),gbHash(i+vec2(1.,0.)),f.x),mix(gbHash(i+vec2(0.,1.)),gbHash(i+vec2(1.,1.)),f.x),f.y);}
vec3 gbNormal;
vec3 gbBlade(vec2 corner){
 float id=float(gl_InstanceID);vec2 cell=floor(uGbEye.xz/uGbCell)+vec2(mod(id,uGbGrid),floor(id/uGbGrid))-uGbGrid*.5;
 vec2 r=gbHash2(cell),q=gbHash2(cell+19.7);vec2 xz=(cell+r)*uGbCell;
 vec4 field=textureLod(uGbField,vec2(xz.x,-xz.y)/(2.*uGbR)+.5,0.);
 float far=length(xz-uGbEye.xz)/uGbReach;
 // fewer, wider blades with distance; none past the reach or off the lawn
 float thin=smoothstep(.3,.9,far);
 // the far tier starts where the near one thins out, so the two overlap and neither edge shows
 float keep=step(.62,field.g)*(1.-smoothstep(.8,1.,far))*step(q.x,1.-.6*thin)*smoothstep(uGbInner*.55,uGbInner,far*uGbReach);
 // tufts: taller, denser clumps a hand's width across; over tens of metres the lawn runs from close-mown to patches
 // left long, where some blades are seed stalks: twice the height, thin, with a pale head
 float tuft=gbNoise(xz*5.3),longer=smoothstep(.42,.72,gbNoise(xz*.085+4.)+.25*gbNoise(xz*.4+1.7)-.12);
 float stalk=step(.955-.1*longer,q.x)*step(.25,longer);
 float height=mix(.07,.2,r.y*r.y)*(.65+.7*tuft)*(1.+1.5*longer)*(1.+1.1*stalk)*keep;
 float width=uGbWidth*(1.+1.6*thin)*(1.+.5*longer)*keep;
 float turn=q.y*6.2831853;vec2 face=vec2(cos(turn),sin(turn));
 // every blade leans its own way, long ones more; the wind bends them all together in slow waves with a quicker flutter
 vec2 lean=(gbHash2(cell+3.1)-.5)*(1.1+.7*longer);
 float wave=sin(uGbTime*1.3+dot(xz,vec2(.42,.27)))+.5*sin(uGbTime*2.9+dot(xz,vec2(-.9,1.3)));
 lean+=vec2(.8,.45)*(.16*wave+.05*sin(uGbTime*7.+r.x*40.))*(1.+.6*longer);
 float t=corner.y;
 vec3 p=vec3(xz.x,field.r/max(field.g,1e-3)-.012,xz.y);
 // a blade tapers to its point; a stalk is a thread carrying a spindle-shaped head on its top third
 float profile=mix(1.-.85*t,.22+1.5*smoothstep(.62,.8,t)*(1.-smoothstep(.86,1.,t)),stalk);
 p.xz+=face*corner.x*width*profile+lean*height*t*t;
 p.y+=height*t*(1.-.2*dot(lean,lean)*t);
 vGbT=t;
 // lush and dry areas follow the lawn's own pattern (ground-detail.ts); one blade in nine has gone to straw
 float lush=gbNoise(xz*.12+2.1),dry=smoothstep(.55,.85,gbNoise(xz*.22+9.4));
 vGbTint=mix(vec3(.78,.86,.7),vec3(1.12,1.18,1.02),lush)*mix(vec3(1.),vec3(1.35,1.18,.78),.55*dry)*(.82+.36*r.x);
 if(q.x>.89)vGbTint*=vec3(1.5,1.3,.75);
 // long grass is a deeper green; a seed head is straw
 vGbTint*=mix(vec3(1.),vec3(.8,.95,.62),longer);
 vGbTint=mix(vGbTint,vec3(2.3,1.75,.95)*(.8+.4*r.x),stalk*smoothstep(.55,.7,t));
 vGbLong=longer;
 // lit as the lawn is, tilted a little by the blade's own bend, so the field reads as one surface with relief
 gbNormal=normalize(vec3(lean.x*.55+face.y*.25,1.,lean.y*.55-face.x*.25));
 return p;
}
`;

export type GrassBlades = ReturnType<typeof buildGrassBlades>;

/** `lawns`: the grass surfaces; `covers`: everything that may lie on or stand over a lawn (roads, paths, pavements,
 *  buildings). `reach`: the half width of the area in metres. `setup` adds the viewer's lighting hooks to the material. */
export function buildGrassBlades(renderer: THREE.WebGLRenderer, scene: THREE.Scene, lawns: THREE.Object3D[], covers: THREE.Object3D[], reach: number, colour: THREE.Color, setup: (material: THREE.MeshStandardMaterial) => void) {
  const R = reach, N = 2048, LAYER = 9;
  const field = new THREE.WebGLRenderTarget(N, N, {type: THREE.FloatType, format: THREE.RGBAFormat, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true});
  const cam = new THREE.OrthographicCamera(-R, R, R, -R, 1, 4000); cam.position.set(0, 2000, 0); cam.up.set(0, 0, -1); cam.lookAt(0, 0, 0); cam.layers.set(LAYER); cam.updateMatrixWorld(true); cam.updateProjectionMatrix();
  const mat = new THREE.ShaderMaterial({vertexShader: FIELD_VERTEX, fragmentShader: FIELD_FRAGMENT, uniforms: {uLawn: {value: 1}}, side: THREE.DoubleSide});
  const previous = {override: scene.overrideMaterial, background: scene.background, fog: scene.fog, target: renderer.getRenderTarget(), auto: renderer.shadowMap.autoUpdate, autoClear: renderer.autoClear, color: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha()};
  scene.overrideMaterial = mat; scene.background = null; scene.fog = null; renderer.shadowMap.autoUpdate = false; renderer.autoClear = false; renderer.setClearColor(0x000000, 0);
  try {
    renderer.setRenderTarget(field); renderer.clear(true, true, true);
    // the covers are raised a hand's width: a path lying a centimetre over a lawn still hides it, and so does the lawn
    // just under a kerb's edge
    for (const [set, lawn] of [[lawns, 1], [covers, 0]] as const) {
      if (!set.length) continue;
      set.forEach(o => o.traverse(x => x.layers.enable(LAYER))); mat.uniforms.uLawn.value = lawn; mat.uniformsNeedUpdate = true;
      cam.position.y = lawn ? 2000 : 1999.88; cam.updateMatrixWorld(true);
      renderer.render(scene, cam);
      set.forEach(o => o.traverse(x => x.layers.disable(LAYER)));
    }
  } finally {
    scene.overrideMaterial = previous.override; scene.background = previous.background; scene.fog = previous.fog;
    renderer.shadowMap.autoUpdate = previous.auto; renderer.autoClear = previous.autoClear; renderer.setClearColor(previous.color, previous.alpha); renderer.setRenderTarget(previous.target); mat.dispose();
  }

  // one blade: three quads narrowing to a point (seven corners), x across the blade, y up it
  const blade = () => {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, 0, 0, 1, 0, 0, -1, .36, 0, 1, .36, 0, -1, .72, 0, 1, .72, 0, 0, 1, 0], 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(Array.from({length: 7}, () => [0, 1, 0]).flat(), 3));
    geometry.setIndex([0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5, 4, 5, 6]); return geometry;
  };
  const eye = new THREE.Vector3(), time = {value: 0};
  // Two tiers: fine blades out to 9 m, then tufts three times as wide and as far apart out to 34 m.
  const tiers = [{grid: 440, cell: .042, inner: 0, width: .0062}, {grid: 440, cell: .16, inner: 7.6, width: .022}].map(({grid, cell, inner, width}) => {
    const geometry = blade(); geometry.instanceCount = grid * grid;
    const uniforms = {uGbField: {value: field.texture}, uGbR: {value: R}, uGbCell: {value: cell}, uGbGrid: {value: grid}, uGbTime: time, uGbReach: {value: grid * cell * .5}, uGbInner: {value: inner}, uGbWidth: {value: width}, uGbEye: {value: eye}};
    const material = new THREE.MeshStandardMaterial({color: colour.clone(), roughness: .82, metalness: 0, side: THREE.DoubleSide}); material.name = 'grass_blades';
    setup(material);
    const chained = material.onBeforeCompile.bind(material);
    material.onBeforeCompile = (shader, r) => {
      chained(shader, r); Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\n' + HEADER)
        .replace('#include <beginnormal_vertex>', 'vec3 gbPlace=gbBlade(position.xy);\nvec3 objectNormal=gbNormal;')
        .replace('#include <begin_vertex>', 'vec3 transformed=gbPlace;');
      shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying float vGbT;varying float vGbLong;varying vec3 vGbTint;')
        // dark and shaded at the root among its neighbours, lighter and a little yellower toward the tip
        .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb*=vGbTint*vec3(.74,.98,.5)*mix(.4-.14*vGbLong,1.5,pow(vGbT,.8))*mix(vec3(1.),vec3(1.12,1.06,.8),vGbT*vGbT);')
        // a blade is thin: with the sun beyond it, the light comes through and it glows a yellower green
        .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>
#if NUM_DIR_LIGHTS > 0
{float through=pow(clamp(dot(-normalize(vViewPosition),directionalLights[0].direction),0.,1.),2.)*vGbT;
 reflectedLight.directDiffuse*=1.+vec3(2.6,2.4,.9)*through;}
#endif`)
        // both faces of a blade are lit as the lawn is: undo the flip a back face's normal is given
        .replace('#include <normal_fragment_begin>', '#include <normal_fragment_begin>\nnormal=normalize(vNormal);');
    };
    const key = material.customProgramCacheKey.bind(material);
    material.customProgramCacheKey = () => key() + '|grass-blades4';
    const mesh = new THREE.Mesh(geometry, material); mesh.name = 'grass_blades'; mesh.frustumCulled = false; mesh.castShadow = false; mesh.receiveShadow = true; mesh.matrixAutoUpdate = false;
    return {mesh, material, geometry, grid, cell, width, uniforms};
  });
  const mesh = new THREE.Group(); mesh.name = 'grass_blades'; tiers.forEach(t => mesh.add(t.mesh)); scene.add(mesh);
  return {mesh, field,
    /** per frame: the point the grid follows (the camera) and the time in seconds (wind) */
    update(at: THREE.Vector3, seconds: number) { eye.copy(at); time.value = seconds; },
    /** 0..1: the share of blades per metre drawn (the frame budget may lower it). The grid keeps its reach: fewer,
     *  wider-spaced and slightly wider blades, so a lawn thins evenly instead of losing a strip. */
    density(share: number) { for (const t of tiers) { const s = Math.max(.2, Math.min(1, share)), grid = Math.round(t.grid * s);
      t.geometry.instanceCount = grid * grid; t.uniforms.uGbGrid.value = grid; t.uniforms.uGbCell.value = t.cell * t.grid / grid; t.uniforms.uGbWidth.value = t.width * Math.sqrt(t.grid / grid); } },
    dispose() { scene.remove(mesh); for (const t of tiers) { t.geometry.dispose(); t.material.dispose(); } field.dispose(); }};
}
