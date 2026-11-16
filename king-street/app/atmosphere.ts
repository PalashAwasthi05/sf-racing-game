import * as THREE from 'three';
import {Sky} from 'three/addons/objects/Sky.js';
import {CSM} from 'three/addons/csm/CSM.js';
import {patchCoatedLight} from './coated-light';
import {patchRoadsterOptics} from './car-optics';

// Distance haze is exponential in distance, as an atmosphere's is (THREE's FogExp2 is exponential-squared: a fog bank
// that hid the model's edge at about a kilometre and would hide the real hills 20 km out). The density is the inverse of
// the haze length: a clear afternoon here sees the East Bay hills and Mt Tamalpais, softened (22 km).
// The haze thins with height (a scale height of 1.2 km): a ray to a hilltop crosses thinner air than one along the water,
// so a mountain's top is clearer than its foot, as it is seen across the bay. At the height of the streets the two agree.
THREE.ShaderChunk.fog_pars_vertex=THREE.ShaderChunk.fog_pars_vertex.replace('varying float vFogDepth;','varying float vFogDepth;varying float vFogWorldY;');
THREE.ShaderChunk.fog_vertex=THREE.ShaderChunk.fog_vertex.replace('vFogDepth = - mvPosition.z;','vFogDepth = - mvPosition.z;vFogWorldY = ( transpose( mat3( viewMatrix ) ) * ( mvPosition.xyz - viewMatrix[ 3 ].xyz ) ).y;');
THREE.ShaderChunk.fog_pars_fragment=THREE.ShaderChunk.fog_pars_fragment.replace('varying float vFogDepth;','varying float vFogDepth;varying float vFogWorldY;');
THREE.ShaderChunk.fog_fragment=THREE.ShaderChunk.fog_fragment.replace('float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );',
 'float hzH0 = max( cameraPosition.y, 0.0 ), hzH1 = max( vFogWorldY, 0.0 ), hzDh = hzH1 - hzH0;\n\t\tfloat hzK = abs( hzDh ) < 1.0 ? exp( - hzH0 / 1200.0 ) : 1200.0 / hzDh * ( exp( - hzH0 / 1200.0 ) - exp( - hzH1 / 1200.0 ) );\n\t\tfloat fogFactor = 1.0 - exp( - fogDensity * vFogDepth * hzK );');

// Sky, clouds and lighting for the world viewer. The daytime presets and the night preset each set the
// sun's position and colour, the atmosphere's scattering, a procedural cloud layer, the sky-lit environment,
// the hemisphere fill, fog and exposure. Shadows come from cascaded shadow maps that follow the camera, so they
// stay crisp where the viewer stands and still cover the whole area. Nothing here is location specific.

export type Preset='afternoon'|'golden'|'overcast'|'night';
export const PRESET_ORDER:Preset[]=['afternoon','golden','overcast','night'];
export const PRESET_LABEL:Record<Preset,string>={afternoon:'Afternoon sun',golden:'Golden hour',overcast:'Overcast',night:'Night drive'};

type Spec={hazeKm:number;azimuthDeg:number;elevationDeg:number;sunColor:string;sunIntensity:number;hemiSky:string;hemiGround:string;hemiIntensity:number;env:number;exposure:number;turbidity:number;rayleigh:number;mie:number;mieG:number;fog:[string,number,number];coverage:number;cloudTint:string;cloudShade:string;cloudOpacity:number;cloudSpeed:number};
const SPECS:Record<Preset,Spec>={
 afternoon:{hazeKm:22,azimuthDeg:262,elevationDeg:31,sunColor:'#fff0d8',sunIntensity:4.4,hemiSky:'#aeb6bf',hemiGround:'#a08a6c',hemiIntensity:.15,env:.95,exposure:1.08,turbidity:2.2,rayleigh:3.1,mie:.004,mieG:.82,fog:['#b8cadb',380,1500],coverage:.42,cloudTint:'#ffffff',cloudShade:'#a9b7c7',cloudOpacity:.97,cloudSpeed:1},
 golden:{hazeKm:14,azimuthDeg:286,elevationDeg:7,sunColor:'#ffb466',sunIntensity:3,hemiSky:'#c9b9ab',hemiGround:'#7a5f48',hemiIntensity:.2,env:2.2,exposure:1.2,turbidity:5.5,rayleigh:3.6,mie:.022,mieG:.93,fog:['#d6b69c',300,1400],coverage:.5,cloudTint:'#ffd6b0',cloudShade:'#7e6270',cloudOpacity:.95,cloudSpeed:1},
 overcast:{hazeKm:6,azimuthDeg:212,elevationDeg:45,sunColor:'#e9eef3',sunIntensity:.55,hemiSky:'#c9d0d6',hemiGround:'#85817a',hemiIntensity:.35,env:1.9,exposure:1.15,turbidity:18,rayleigh:1.1,mie:.006,mieG:.7,fog:['#c4cbd2',260,1300],coverage:.97,cloudTint:'#e0e4e9',cloudShade:'#98a1aa',cloudOpacity:1,cloudSpeed:.6},
 night:{hazeKm:10,azimuthDeg:122,elevationDeg:36,sunColor:'#b8cde8',sunIntensity:.18,hemiSky:'#93a7c4',hemiGround:'#937e73',hemiIntensity:.36,env:1.05,exposure:1.1,turbidity:2,rayleigh:1,mie:.004,mieG:.8,fog:['#222633',0,0],coverage:.32,cloudTint:'#303647',cloudShade:'#202430',cloudOpacity:.63,cloudSpeed:.65},
};
// The analytic sky is authored for an exposure near one half; at filmic exposure it is scaled so the zenith keeps its
// blue and the horizon does not wash to white. The same scaled sky, without its solar disc, is the environment that lights
// and is reflected by every surface: rendered unscaled and with the disc, the environment was 2.5 times the sky one sees
// and carried a second, unshadowable sun, so shaded walls were lit nearly as brightly as sunlit ones.
const SKY_LOOK={scale:.4,saturation:1.55,zenithDim:.62};
// Ambient calibrated on 18 % grey cards (2026-10-07). The environment that lights the world is the sky, less saturated than
// the sky one sees (envSat: at full saturation every shaded road and pavement came out twice as blue as red) and at
// skyK of its brightness, over a lower half of sunlit warm-grey ground (the street and the facades across it, albedo
// `bounce`): before, the lower half was the sky shader's bright underside, so a shaded wall was lit more than the
// shaded ground in front of it and every surface read evenly, flatly lit. Measured with these values (afternoon): sunlit
// ground about 4x shaded ground, a wall facing the sun about 4.5x one facing away, shade a cool neutral.
const LOOK={envSat:.42,groundAlbedo:[.2,.19,.17],skyIrradiance:[.55,.62,.75],bounce:.45,skyK:.68};
// The sun's 0.53 degree disc makes a penumbra whose width grows with the gap
// between a blocker and the receiving surface. PCSS below estimates that gap
// from the same depth maps CSM already draws; no extra render pass is needed.
const SUN_ANGULAR_RADIUS=Math.tan(THREE.MathUtils.degToRad(.266));
const SUN_PENUMBRA=/* glsl */`
#if defined( USE_CSM ) && defined( CSM_CASCADES )
uniform float uSunPenumbraScale[CSM_CASCADES];
// sixteen points spread evenly over the unit disc (a Poisson-disc set)
const vec2 SUN_DISC[16]=vec2[16](vec2(-.94201624,-.39906216),vec2(.94558609,-.76890725),vec2(-.094184101,-.92938870),vec2(.34495938,.29387760),vec2(-.91588581,.45771432),vec2(-.81544232,-.87912464),vec2(-.38277543,.27676845),vec2(.97484398,.75648379),vec2(.44323325,-.97511554),vec2(.53742981,-.47373420),vec2(-.26496911,-.41893023),vec2(.79197514,.19090188),vec2(-.24188840,.99706507),vec2(-.81409955,.91437590),vec2(.19984126,.78641367),vec2(.14383161,-.14100790));
void addSunBlocker(float depth,float receiver,inout float sum,inout float count){
 #ifdef USE_REVERSED_DEPTH_BUFFER
  if(depth>receiver){sum+=depth;count+=1.;}
 #else
  if(depth<receiver){sum+=depth;count+=1.;}
 #endif
}
bool receivesDirectSun(vec3 normal,vec3 coatNormal,vec3 direction){
 #if defined(LEAF_BACKLIGHT)
  return true;
 #elif defined(USE_CLEARCOAT)
  return max(dot(normal,direction),dot(coatNormal,direction))>0.;
 #else
  return dot(normal,direction)>0.;
 #endif
}
// facing: the cosine between the surface normal and the sun. A surface nearly edge-on to the sun changes depth by many
// times a texel's width from one shadow texel to the next, so a filter several texels wide compared against one depth
// reads the surface as shadowing itself, in texel-sized squares (a car body under a 7 degree sun was a checkerboard).
// The compared depth is therefore moved toward the sun by the surface's slope times the reach of each lookup.
float sunSlopeDepth(float facing,float penumbraScale){
 float slope=min(sqrt(max(1.-facing*facing,0.))/max(facing,.06),14.);
 return slope*SUN_ANGULAR_RADIUS_GLSL/max(penumbraScale,1e-6);                 // depth per texel of sideways reach
}
float sunBiased(float z,float amount){
 #ifdef USE_REVERSED_DEPTH_BUFFER
  return z+amount;
 #else
  return z-amount;
 #endif
}
float getSunShadow(sampler2D shadowMap,vec2 shadowMapSize,float shadowIntensity,float shadowBias,vec4 shadowCoord,float penumbraScale,float facing){
 shadowCoord.xyz/=shadowCoord.w;shadowCoord.z+=shadowBias;
 float slopeDepth=sunSlopeDepth(facing,penumbraScale),receiverZ=shadowCoord.z;
 shadowCoord.z=sunBiased(receiverZ,slopeDepth*2.2);
 bool inFrustum=shadowCoord.x>=0.&&shadowCoord.x<=1.&&shadowCoord.y>=0.&&shadowCoord.y<=1.&&shadowCoord.z<=1.;
 if(!inFrustum)return 1.;
 vec2 texel=1./shadowMapSize;float blocker=0.;float blockers=0.;
 // A fixed pattern avoids temporal shimmer as the camera moves. Its small
 // search disk preserves isolated contact shadows before the penumbra grows.
 addSunBlocker(unpackRGBAToDepth(texture2D(shadowMap,shadowCoord.xy+texel*vec2(-1.5,-.5))),shadowCoord.z,blocker,blockers);
 addSunBlocker(unpackRGBAToDepth(texture2D(shadowMap,shadowCoord.xy+texel*vec2(.5,-1.5))),shadowCoord.z,blocker,blockers);
 addSunBlocker(unpackRGBAToDepth(texture2D(shadowMap,shadowCoord.xy+texel*vec2(1.5,.5))),shadowCoord.z,blocker,blockers);
 addSunBlocker(unpackRGBAToDepth(texture2D(shadowMap,shadowCoord.xy+texel*vec2(-.5,1.5))),shadowCoord.z,blocker,blockers);
 if(blockers<1.)return 1.;blocker/=blockers;
 float spread=max(clamp(abs(receiverZ-blocker)*penumbraScale,0.,18.),1.);
 // The penumbra is sampled over a disc of that radius: sixteen points, the disc turned by a different angle at every
 // pixel. (Nine taps snapped to texel centres, as before, are constant across each texel once the penumbra is several
 // texels wide: under a tree every leaf shadow became a grid of squares.) The per-pixel turn trades those squares for
 // fine grain, which the temporal resolve averages away.
 float turn=6.2831853*fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))));
 mat2 spin=mat2(cos(turn),-sin(turn),sin(turn),cos(turn));
 float radius=max(spread,1.5);
 shadowCoord.z=sunBiased(receiverZ,slopeDepth*(radius+1.));
 float shadow=0.;
 for(int k=0;k<16;k++)shadow+=texture2DCompare(shadowMap,shadowCoord.xy+spin*SUN_DISC[k]*radius*texel,shadowCoord.z);
 shadow*=1./16.;
 return mix(1.,shadow,shadowIntensity);
}
#endif
`;
const CSM_SUN_CALLS=/getShadow\( directionalShadowMap\[ i \], directionalLightShadow\.shadowMapSize, directionalLightShadow\.shadowIntensity, directionalLightShadow\.shadowBias, directionalLightShadow\.shadowRadius, vDirectionalShadowCoord\[ i \] \)/g;
function patchCsmSunShadow(shader:THREE.WebGLProgramParametersWithUniforms,scales:Float32Array){
 // CSM replaces this include globally in r180; expand its chunk only for a
 // material registered with this CSM, then put helper declarations at fragment
 // scope after depth unpacking/comparison functions are available.
 const rawChunk=THREE.ShaderChunk.lights_fragment_begin;
 const callCount=rawChunk.match(CSM_SUN_CALLS)?.length??0;
 // r180's CSM chunk has one call in the fade branch, one in its non-fade
 // branch, and a third in the mutually-exclusive non-CSM fallback branch.
 if(callCount!==3||!shader.fragmentShader.includes('#include <shadowmap_pars_fragment>')||!shader.fragmentShader.includes('#include <lights_fragment_begin>'))throw new Error(`CSM sun shadow patch expected three r180 call sites and standard shader includes; found ${callCount}`);
 const csmChunk=rawChunk.replace(CSM_SUN_CALLS,'(receivesDirectSun(geometryNormal,geometryClearcoatNormal,directLight.direction) ? getSunShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, vDirectionalShadowCoord[ i ], uSunPenumbraScale[ i ], max(dot(geometryNormal,directLight.direction),0.) ) : 1.)');
 shader.fragmentShader=shader.fragmentShader
  .replace('#include <shadowmap_pars_fragment>','#include <shadowmap_pars_fragment>\n'+SUN_PENUMBRA.replace(/SUN_ANGULAR_RADIUS_GLSL/g,SUN_ANGULAR_RADIUS.toFixed(7)))
  .replace('#include <lights_fragment_begin>',csmChunk);
 shader.uniforms.uSunPenumbraScale={value:scales};
}
// The saturation boost is for the sky as seen; the light it casts keeps the scattered colour it has (boosted, it tinted
// every shadow a strong blue).
// The sky as seen fades, over its lowest 20 degrees, to the preset's haze colour at its own brightness: the analytic
// sky turns neutral grey-white there and read as an overcast band; the haze on distant buildings has the same colour.
// Night radiance is independent of the daylight scattering model. A low city-light
// horizon and moon halo appear in reflections. The disc is masked out of the
// irradiance bake because its directional illumination already has shadows.
const NIGHT_SKY = /* glsl */`
 vec3 nightRadiance(vec3 viewDir) {
  float horizon = exp(-5.8 * max(viewDir.y, 0.0));
  vec3 sky = mix(vec3(.0058,.0065,.0086), vec3(.015,.018,.025), horizon);
  float alignment = clamp(dot(viewDir, normalize(vSunDirection)), -1., 1.);
  float separation = 1. - alignment;
  sky += vec3(.008,.011,.017) * exp(-separation * 56.);
  sky += vec3(.018,.022,.032) * exp(-separation * 840.);
  float disc = 1. - smoothstep(.000007, .000014, separation);
  return sky + uDrawDisc * vec3(2.2,2.45,2.72) * disc;
 }`;
function patchSky(sky:Sky,disc:boolean){
 const m=sky.material as THREE.ShaderMaterial;Object.assign(m.uniforms,{uNight:{value:0},uClearDay:{value:0},uDrawDisc:{value:disc?1:0},uSkyScale:{value:SKY_LOOK.scale},uSkySat:{value:disc?SKY_LOOK.saturation:1},uZenithDim:{value:SKY_LOOK.zenithDim},uHaze:{value:new THREE.Color('#b8cadb')},uHazeMix:{value:disc?.7:0},uDaySky:{value:null},uDaySkyEnabled:{value:0},uDayAzimuthOffset:{value:0},uDayElevationScale:{value:1},uDaySkyScale:{value:SKY_LOOK.scale}});
 let f=m.fragmentShader.replace('uniform float mieDirectionalG;','uniform float mieDirectionalG;uniform float uNight;uniform float uClearDay;uniform float uDrawDisc;uniform float uSkyScale;uniform float uSkySat;uniform float uZenithDim;uniform vec3 uHaze;uniform float uHazeMix;uniform sampler2D uDaySky;uniform float uDaySkyEnabled;uniform float uDayAzimuthOffset;uniform float uDayElevationScale;uniform float uDaySkyScale;')
  .replace('void main() {', NIGHT_SKY + '\nvoid main() {')
  .replace('vec3 direction = normalize( vWorldPosition - cameraPosition );', '$&\n if(uNight>.5){gl_FragColor=vec4(nightRadiance(direction),1.);}else{')
  .replace('// in scattering', /* glsl */`
// The captured daylight sky already includes atmospheric scattering. Only its
// solar disc needs the extinction term; the analytic sky uses the full model.
   if(uDaySkyEnabled>.5&&direction.y>0.){
    float sourceElev=atan(uDayElevationScale*direction.y,max(length(direction.xz),1e-6));
    float az=atan(direction.x,-direction.z);
    vec2 dayUv=vec2(fract(az/(2.*pi)+uDayAzimuthOffset),sourceElev/(.5*pi));
    vec3 dayRadiance=exp2(8.*texture2D(uDaySky,dayUv).rgb)-1.;
    vec3 skyCol=dayRadiance*uDaySkyScale;
    float sunCos=dot(direction,vSunDirection);
    float liveDisc=smoothstep(cos(.0046426),cos(.0046426)+.000008,sunCos);
    skyCol+=uDrawDisc*(vSunE*19000.*Fex)*liveDisc*.04*uDaySkyScale;

    gl_FragColor=vec4(skyCol,1.);
   }else{
   // in scattering`)
  .replace('gl_FragColor = vec4( retColor, 1.0 );', /* glsl */`
   vec3 skyCol = retColor * uSkyScale * mix(1.,uZenithDim,sqrt(clamp(direction.y,0.,1.)));
   float skyLum = dot(skyCol,vec3(.2126,.7152,.0722));
   skyCol = max(mix(vec3(skyLum),skyCol,uSkySat),vec3(0.));
   float hz = (1.-smoothstep(0.,.35,direction.y)) * uHazeMix;
   vec3 haze = uHaze * dot(skyCol,vec3(.2126,.7152,.0722)) / max(dot(uHaze,vec3(.2126,.7152,.0722)),1e-4);
   skyCol = mix(skyCol,haze*.94,hz);
   // Preserve the low horizon and solar aureole where scattering is white.
   float clearAir = smoothstep(.015,.32,direction.y) * (1.-smoothstep(.90,.995,dot(direction,vSunDirection)));
   skyCol *= mix(vec3(1.),vec3(.65,.80,1.04),uClearDay * clearAir);
   gl_FragColor=vec4(skyCol,1.);
   }}`);
 f=f.replace('L0 += ( vSunE * 19000.0 * Fex ) * sundisk;',
  'L0 += (vSunE * 19000.0 * Fex) * sundisk * uDrawDisc;');
 m.fragmentShader=f;m.needsUpdate=true;
}
function sunDirection(azimuthDeg:number,elevationDeg:number){const az=THREE.MathUtils.degToRad(azimuthDeg),el=THREE.MathUtils.degToRad(elevationDeg);return new THREE.Vector3(Math.sin(az)*Math.cos(el),Math.sin(el),-Math.cos(az)*Math.cos(el)).normalize();}

const CLOUD_VERTEX=`varying vec3 vDir;void main(){vDir=(modelMatrix*vec4(position,1.)).xyz-cameraPosition;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const CLOUD_FRAGMENT=`
precision highp float;varying vec3 vDir;
uniform float uTime,uCoverage,uOpacity;uniform vec3 uSunDir,uTint,uShade,uHorizon;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*noise(p);p=p*2.03+vec2(17.,9.);a*=.5;}return v;}
void main(){
 vec3 d=normalize(vDir);if(d.y<.005)discard;
 // Perspective through a low cumulus deck and a higher, sheared veil. Limit
 // high-frequency detail at the horizon where many metres project to a pixel.
 vec2 flow=uTime*vec2(.011,.005);
 vec2 uv=d.xz*(1.45/max(d.y,.03))+flow;
 float detail=1.-smoothstep(.09,.35,max(length(dFdx(uv)),length(dFdy(uv))));
 vec2 warp=vec2(noise(uv*.63),noise(uv*.63+vec2(5.,7.)))-.5;
 uv+=warp*.62;
 float base=fbm(uv);
 float small=noise(uv*8.1+vec2(12.,3.));
 float n=base+detail*.08*(small-.5);
 float edge=.60-.31*uCoverage;
 float optical=max(0.,(n-edge)*8.8);
 float density=1.-exp(-optical);
 vec2 sunOffset=normalize(uSunDir.xz+vec2(1e-4))*.17;
 float ahead=fbm(uv+sunOffset);
 float shade=clamp(.15+optical*.24+(ahead-base)*2.6,0.,1.);
 float silver=pow(max(dot(d,uSunDir),0.),7.)*(1.-density)*.36;
 vec3 col=mix(uTint,uShade,shade)+uTint*silver;
 vec2 high=d.xz*(2.9/max(d.y,.045))+flow*.42;
 high=mat2(.78,-.63,.63,.78)*high;
 float wisps=smoothstep(.54,.79,fbm(high*vec2(.28,1.9)))*.28;
 float alpha=density+wisps*(1.-density);
 col=mix(uTint*.91,col,density/max(alpha,.001));
 float horizon=smoothstep(.017,.19,d.y);
 gl_FragColor=vec4(mix(uHorizon,col,horizon),alpha*horizon*uOpacity);
}`;

export async function loadDaySky(){
 try{
  const texture=await new THREE.TextureLoader().loadAsync('/environment/day-clouds.webp');
  texture.colorSpace=THREE.NoColorSpace;texture.wrapS=THREE.RepeatWrapping;texture.wrapT=THREE.ClampToEdgeWrapping;
  texture.needsUpdate=true;return texture;
 }catch{return null;}
}
export function createAtmosphere(scene:THREE.Scene,renderer:THREE.WebGLRenderer,camera:THREE.PerspectiveCamera,initial:Preset='afternoon',initialDaySky?:THREE.Texture|null){
 // Sky.js places every fragment at the far clip plane and writes no depth. Draw it after opaque
 // terrain and buildings so only uncovered sky pixels run the atmosphere shader.
 const sky=new Sky();sky.scale.setScalar(1600);sky.renderOrder=1e4;scene.add(sky);patchSky(sky,true);
 const cloudUniforms={uTime:{value:0},uCoverage:{value:.4},uOpacity:{value:.9},uSunDir:{value:new THREE.Vector3(0,1,0)},uTint:{value:new THREE.Color('#ffffff')},uShade:{value:new THREE.Color('#a0abb8')},uHorizon:{value:new THREE.Color('#b8cad8')}};
 const clouds=new THREE.Mesh(new THREE.SphereGeometry(900,48,20,0,Math.PI*2,0,Math.PI*.52),new THREE.ShaderMaterial({vertexShader:CLOUD_VERTEX,fragmentShader:CLOUD_FRAGMENT,uniforms:cloudUniforms,transparent:true,depthWrite:false,side:THREE.BackSide,fog:false}));
 clouds.renderOrder=1;clouds.frustumCulled=false;scene.add(clouds);
 const hemi=new THREE.HemisphereLight('#cfe2ff','#8b7458',.34);scene.add(hemi);
 // Cascaded shadow maps: the near cascade around the camera keeps shadows sharp on the street, the far one
 // still shadows the rest of the area; the cascades fade into each other.
 // The split is set by hand. The 'practical' scheme put it at 93 m, so the near map covered a box some 200 m wide at
 // about 20 cm a texel: leaf shadows on the car and the road beside it were squares the size of a hand. The near map
 // now covers the first NEAR_SHADOW_M (about 3 cm a texel), where shadow detail is seen; the far map takes the rest at
 // twice the resolution it had. A third cascade would add a texture to every lit material, and the textured wall
 // materials are already at the 16 a fragment shader may use.
 const NEAR_SHADOW_M=32;
 const csm=new CSM({camera,parent:scene,cascades:2,maxFar:360,mode:'custom',customSplitsCallback:(_cascades:number,_near:number,far:number,breaks:number[])=>{breaks.push(NEAR_SHADOW_M/far,1);},shadowMapSize:2048,shadowBias:-.00012,lightDirection:new THREE.Vector3(0,-1,0),lightIntensity:3,lightNear:1,lightFar:1500,lightMargin:160});
 csm.fade=true;
 for(const light of csm.lights){light.shadow.normalBias=.03;}
 const sunPenumbraScale=new Float32Array(csm.cascades);
 // The far cascade covers the static world: it is redrawn when the camera has moved 15 m or half a second has
 // passed, not every frame; its light keeps the matrix of its last render, so the map and the lookup stay paired.
 const farCascade=csm.lights[1];farCascade.shadow.mapSize.set(4096,4096);farCascade.shadow.autoUpdate=false;farCascade.shadow.needsUpdate=true;const farCascadeAt=new THREE.Vector3(Infinity,Infinity,Infinity);let farCascadeTime=-Infinity;
 const pmrem=new THREE.PMREMGenerator(renderer);let environment:THREE.WebGLRenderTarget|null=null,skyEnvironment:THREE.WebGLRenderTarget|null=null;
 let current:Preset='afternoon';let shadowFov=camera.fov,shadowAspect=camera.aspect;const sunDir=new THREE.Vector3();
 const projection=new THREE.Matrix4(),inverseProjection=new THREE.Matrix4();
 function resizeShadowFrustums(){
  // Acceleration eases the lens every frame. Fit shadow maps to a slightly
  // wider lens so sub-degree changes don't revisit every material in the city.
  const viewFov=camera.fov;
  shadowFov=Math.ceil(viewFov*4)/4;shadowAspect=camera.aspect;
  projection.copy(camera.projectionMatrix);inverseProjection.copy(camera.projectionMatrixInverse);
  try{camera.fov=shadowFov;csm.updateFrustums();}
  finally{camera.fov=viewFov;camera.projectionMatrix.copy(projection);camera.projectionMatrixInverse.copy(inverseProjection);}
 }
 let daySky=initialDaySky??null,disposed=false;
 function daySkyUniforms(u:Record<string,THREE.IUniform>,s:Spec,scale:number){
  const useImage=current==='afternoon'&&daySky!==null,sourceAz=(1218.5/2048-.5)*Math.PI*2,sourceEl=THREE.MathUtils.degToRad(47.9),targetAz=THREE.MathUtils.degToRad(s.azimuthDeg),targetEl=THREE.MathUtils.degToRad(s.elevationDeg);
  u.uDaySky.value=daySky;u.uDaySkyEnabled.value=Number(useImage);u.uDaySkyScale.value=scale;
  u.uDayAzimuthOffset.value=.5+(sourceAz-targetAz)/(Math.PI*2);
  u.uDayElevationScale.value=Math.tan(sourceEl)/Math.tan(targetEl);
 }
 function apply(preset:Preset){
  const s=SPECS[preset];const night=preset==='night';const LOOK_=LOOK;current=preset;sunDir.copy(sunDirection(s.azimuthDeg,s.elevationDeg));
  const u=sky.material.uniforms;u.uNight.value=Number(night);u.uClearDay.value=Number(preset==='afternoon');u.turbidity.value=s.turbidity;u.rayleigh.value=s.rayleigh;u.mieCoefficient.value=s.mie;u.mieDirectionalG.value=s.mieG;u.sunPosition.value.copy(sunDir);daySkyUniforms(u,s,SKY_LOOK.scale);
  csm.lightDirection.copy(sunDir).negate();csm.lightIntensity=s.sunIntensity;
  for(const light of csm.lights){light.color.set(s.sunColor);light.intensity=s.sunIntensity;}
  farCascade.shadow.needsUpdate=true;farCascadeTime=-Infinity;
  hemi.color.set(s.hemiSky);hemi.groundColor.set(s.hemiGround);hemi.intensity=s.hemiIntensity;
  // Haze by the preset's haze length (exponential, thinning with height; see the shader chunks at the top of this file).
  (sky.material as THREE.ShaderMaterial).uniforms.uHaze.value.set(s.fog[0]);
  // Distant terrain holds a reference to this same haze colour. Keep it live
  // across lighting changes so a daytime horizon never shows through at night.
  const fog=scene.fog instanceof THREE.FogExp2 ? scene.fog : new THREE.FogExp2(s.fog[0]);
  fog.color.set(s.fog[0]);fog.density=1/(s.hazeKm*1000);scene.fog=fog;
  scene.background=new THREE.Color(s.fog[0]);
  renderer.toneMappingExposure=s.exposure;
  cloudUniforms.uCoverage.value=s.coverage;cloudUniforms.uOpacity.value=preset==='afternoon'&&daySky?0:s.cloudOpacity;cloudUniforms.uSunDir.value.copy(sunDir);
  clouds.visible=cloudUniforms.uOpacity.value>0;
  cloudUniforms.uTint.value.set(s.cloudTint);cloudUniforms.uShade.value.set(s.cloudShade);cloudUniforms.uHorizon.value.set(s.fog[0]);
  // The environment is the sky itself, filtered: glazing, paint and metal reflect the preset's sky, and shaded
  // walls are lit by it.
  const envScene=new THREE.Scene();const envSky=new Sky();envSky.scale.setScalar(1600);patchSky(envSky,false);const eu=envSky.material.uniforms;
  eu.uNight.value=Number(night);eu.turbidity.value=s.turbidity;eu.rayleigh.value=s.rayleigh;eu.mieCoefficient.value=s.mie;eu.mieDirectionalG.value=s.mieG;eu.sunPosition.value.copy(sunDir);daySkyUniforms(eu,s,SKY_LOOK.scale*LOOK_.skyK);envScene.add(envSky);
  eu.uSkySat.value=LOOK_.envSat;eu.uSkyScale.value=SKY_LOOK.scale*LOOK_.skyK;
  const sunE=s.sunIntensity*Math.max(0,sunDir.y),sc=new THREE.Color(s.sunColor),gs=LOOK_.bounce/.2;
  const groundMat=new THREE.MeshBasicMaterial({fog:false,side:THREE.DoubleSide});
  groundMat.color.setRGB((sc.r*sunE+LOOK_.skyIrradiance[0])*LOOK_.groundAlbedo[0]*gs/Math.PI,(sc.g*sunE+LOOK_.skyIrradiance[1])*LOOK_.groundAlbedo[1]*gs/Math.PI,(sc.b*sunE+LOOK_.skyIrradiance[2])*LOOK_.groundAlbedo[2]*gs/Math.PI);
  if(night) groundMat.color.setRGB(.015,.013,.012);
  const ground=new THREE.Mesh(new THREE.CircleGeometry(1500,48),groundMat);ground.rotation.x=-Math.PI/2;ground.position.y=-1.5;envScene.add(ground);
  const next=pmrem.fromScene(envScene,.04,.1,10000);environment?.dispose();environment=next;scene.environment=next.texture;scene.environmentIntensity=s.env;
  envSky.geometry.dispose();envSky.material.dispose();ground.geometry.dispose();groundMat.dispose();
  // The sky as one sees it (its display saturation and horizon haze, no solar disc: the sun's glint is the direct light's),
  // for surfaces that mirror it outright, open water: the lighting environment above is desaturated and its lower half
  // is ground, so water that mirrored it read grey.
  const seeScene=new THREE.Scene();const seeSky=new Sky();seeSky.scale.setScalar(1600);patchSky(seeSky,false);const su=seeSky.material.uniforms;
  su.uNight.value=Number(night);su.uClearDay.value=Number(preset==='afternoon');su.turbidity.value=s.turbidity;su.rayleigh.value=s.rayleigh;su.mieCoefficient.value=s.mie;su.mieDirectionalG.value=s.mieG;su.sunPosition.value.copy(sunDir);daySkyUniforms(su,s,SKY_LOOK.scale);
  su.uSkySat.value=SKY_LOOK.saturation;su.uHazeMix.value=.7;su.uHaze.value.set(s.fog[0]);seeScene.add(seeSky);
  // Water and polished surfaces reflect the cloud banks seen overhead. Draw the
  // same sky materials into this probe while keeping the calibrated diffuse sky.
  const cloudReflection = new THREE.Mesh(clouds.geometry,clouds.material);
  cloudReflection.renderOrder=clouds.renderOrder;
  if(clouds.visible)seeScene.add(cloudReflection);
  const seen=pmrem.fromScene(seeScene,.02,.1,10000);skyEnvironment?.dispose();skyEnvironment=seen;
  seeSky.geometry.dispose();seeSky.material.dispose();
 }
 function setupMaterial(material:THREE.Material|THREE.Material[]){
  for(const m of Array.isArray(material)?material:[material]){
   if(csm.shaders.has(m)||!(m instanceof THREE.MeshStandardMaterial||m instanceof THREE.MeshLambertMaterial||m instanceof THREE.MeshPhongMaterial))continue;
   csm.setupMaterial(m);
   const previous=m.onBeforeCompile.bind(m);
   m.onBeforeCompile=(shader,activeRenderer)=>{previous(shader,activeRenderer);patchCsmSunShadow(shader,sunPenumbraScale);if(m instanceof THREE.MeshPhysicalMaterial)patchCoatedLight(shader);patchRoadsterOptics(m,shader);};
   const retire=()=>{csm.shaders.delete(m);m.removeEventListener('dispose',retire);};
   m.addEventListener('dispose',retire);
  }
 }
 function update(now:number){
  cloudUniforms.uTime.value=now/1000*.06*SPECS[current].cloudSpeed;
  clouds.position.set(camera.position.x,0,camera.position.z);
  if(Math.ceil(camera.fov*4)/4!==shadowFov||camera.aspect!==shadowAspect)resizeShadowFrustums();
  csm.update();
  for(let i=0;i<csm.cascades;i++){const shadowCamera=csm.lights[i].shadow.camera as THREE.OrthographicCamera;sunPenumbraScale[i]=(shadowCamera.far-shadowCamera.near)*SUN_ANGULAR_RADIUS*csm.lights[i].shadow.mapSize.x/Math.max(shadowCamera.right-shadowCamera.left,1e-4);}
  if(camera.position.distanceToSquared(farCascadeAt)>225||now-farCascadeTime>500){farCascade.shadow.needsUpdate=true;farCascadeAt.copy(camera.position);farCascadeTime=now;}
 }
 apply(initial);
 if(initialDaySky===undefined)void loadDaySky().then(texture=>{
  if(!texture)return;
  if(disposed){texture.dispose();return;}
  daySky=texture;if(current==='afternoon')apply(current);
 });
 return {apply,update,setupMaterial,current:()=>current,nearShadowM:NEAR_SHADOW_M,skyEnvironment:()=>skyEnvironment?.texture??null,environmentFilter:pmrem,resize:resizeShadowFrustums,sunDirection:()=>sunDir.clone(),csm,sky,clouds,hemi,
  dispose:()=>{disposed=true;daySky?.dispose();csm.dispose();environment?.dispose();skyEnvironment?.dispose();pmrem.dispose();scene.remove(sky,clouds,hemi);sky.geometry.dispose();sky.material.dispose();clouds.geometry.dispose();(clouds.material as THREE.Material).dispose();}};
}
export type Atmosphere=ReturnType<typeof createAtmosphere>;
