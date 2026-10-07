import * as THREE from 'three';
import {Sky} from 'three/addons/objects/Sky.js';
import {CSM} from 'three/addons/csm/CSM.js';

// Sky, clouds and lighting for the world viewer. Three presets (afternoon, golden hour, overcast) each set the
// sun's position and colour, the atmosphere's scattering, a procedural cloud layer, the sky-lit environment,
// the hemisphere fill, fog and exposure. Shadows come from cascaded shadow maps that follow the camera, so they
// stay crisp where the viewer stands and still cover the whole area. Nothing here is location specific.

export type Preset='afternoon'|'golden'|'overcast';
export const PRESET_ORDER:Preset[]=['afternoon','golden','overcast'];
export const PRESET_LABEL:Record<Preset,string>={afternoon:'Afternoon sun',golden:'Golden hour',overcast:'Overcast'};

type Spec={azimuthDeg:number;elevationDeg:number;sunColor:string;sunIntensity:number;hemiSky:string;hemiGround:string;hemiIntensity:number;env:number;exposure:number;turbidity:number;rayleigh:number;mie:number;mieG:number;fog:[string,number,number];coverage:number;cloudTint:string;cloudShade:string;cloudOpacity:number;cloudSpeed:number};
const SPECS:Record<Preset,Spec>={
 afternoon:{azimuthDeg:262,elevationDeg:31,sunColor:'#fff0d8',sunIntensity:3,hemiSky:'#aeb6bf',hemiGround:'#a08a6c',hemiIntensity:.15,env:1.8,exposure:.9,turbidity:2.2,rayleigh:3.1,mie:.004,mieG:.82,fog:['#b8cadb',380,1500],coverage:.42,cloudTint:'#ffffff',cloudShade:'#a9b7c7',cloudOpacity:.97,cloudSpeed:1},
 golden:{azimuthDeg:286,elevationDeg:7,sunColor:'#ffb466',sunIntensity:3,hemiSky:'#c9b9ab',hemiGround:'#7a5f48',hemiIntensity:.2,env:2.2,exposure:1.2,turbidity:5.5,rayleigh:3.6,mie:.022,mieG:.93,fog:['#d6b69c',300,1400],coverage:.5,cloudTint:'#ffd6b0',cloudShade:'#7e6270',cloudOpacity:.95,cloudSpeed:1},
 overcast:{azimuthDeg:212,elevationDeg:45,sunColor:'#e9eef3',sunIntensity:.55,hemiSky:'#c9d0d6',hemiGround:'#85817a',hemiIntensity:.35,env:1.9,exposure:1.15,turbidity:18,rayleigh:1.1,mie:.006,mieG:.7,fog:['#c4cbd2',260,1300],coverage:.97,cloudTint:'#e0e4e9',cloudShade:'#98a1aa',cloudOpacity:1,cloudSpeed:.6},
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
// The saturation boost is for the sky as seen; the light it casts keeps the scattered colour it has (boosted, it tinted
// every shadow a strong blue).
// The sky as seen fades, over its lowest 20 degrees, to the preset's haze colour at its own brightness: the analytic
// sky turns neutral grey-white there and read as an overcast band; the haze on distant buildings has the same colour.
function patchSky(sky:Sky,disc:boolean){
 const m=sky.material as THREE.ShaderMaterial;Object.assign(m.uniforms,{uSkyScale:{value:SKY_LOOK.scale},uSkySat:{value:disc?SKY_LOOK.saturation:1},uZenithDim:{value:SKY_LOOK.zenithDim},uHaze:{value:new THREE.Color('#b8cadb')},uHazeMix:{value:disc?.7:0}});
 let f=m.fragmentShader.replace('uniform float mieDirectionalG;','uniform float mieDirectionalG;uniform float uSkyScale;uniform float uSkySat;uniform float uZenithDim;uniform vec3 uHaze;uniform float uHazeMix;')
  .replace('gl_FragColor = vec4( retColor, 1.0 );','vec3 skyCol = retColor * uSkyScale * mix( 1.0, uZenithDim, pow( clamp( direction.y, 0.0, 1.0 ), 0.5 ) );\n\t\t\tfloat skyLum = dot( skyCol, vec3( 0.2126, 0.7152, 0.0722 ) );\n\t\t\tskyCol = max( mix( vec3( skyLum ), skyCol, uSkySat ), vec3( 0.0 ) );\n\t\t\tfloat hz = ( 1.0 - smoothstep( 0.0, 0.35, direction.y ) ) * uHazeMix;\n\t\t\tvec3 haze = uHaze * dot( skyCol, vec3( 0.2126, 0.7152, 0.0722 ) ) / max( dot( uHaze, vec3( 0.2126, 0.7152, 0.0722 ) ), 1e-4 );\n\t\t\tskyCol = mix( skyCol, haze * 0.94, hz );\n\t\t\tgl_FragColor = vec4( skyCol, 1.0 );');
 if(!disc)f=f.replace('L0 += ( vSunE * 19000.0 * Fex ) * sundisk;','');
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
 // A flat cloud deck 1.5 km up, read where the view ray meets it; slow drift with the wind.
 vec2 uv=d.xz*(1500./d.y)*.00055+uTime*vec2(.011,.005);
 float n=fbm(uv)+.35*fbm(uv*3.1+vec2(4.2,1.7));
 float t0=.78-.5*uCoverage;float dens=smoothstep(t0,t0+.15,n);
 // Self-shadowing: where the cloud thickens toward the sun, the point is in the cloud's own shade; sunlit tops and
 // edges stay white, bases and far sides turn blue-grey, and thin edges toward the sun glow (the silver lining).
 vec2 toSun=normalize(uSunDir.xz+vec2(1e-4))*.05;vec2 uv2=uv+toSun;
 float n2=fbm(uv2)+.35*fbm(uv2*3.1+vec2(4.2,1.7));
 float shade=clamp((n2-n)*5.+.35,0.,1.)*smoothstep(0.,.6,dens);
 float towardSun=max(dot(d,uSunDir),0.);
 vec3 col=mix(uTint,uShade,shade);
 col+=uTint*pow(towardSun,6.)*.8*(1.-dens)*dens*2.;
 float horizon=smoothstep(.02,.2,d.y);
 col=mix(uHorizon,col,horizon);
 gl_FragColor=vec4(col,dens*horizon*uOpacity);
}`;

export function createAtmosphere(scene:THREE.Scene,renderer:THREE.WebGLRenderer,camera:THREE.PerspectiveCamera){
 const sky=new Sky();sky.scale.setScalar(1600);scene.add(sky);patchSky(sky,true);
 const cloudUniforms={uTime:{value:0},uCoverage:{value:.4},uOpacity:{value:.9},uSunDir:{value:new THREE.Vector3(0,1,0)},uTint:{value:new THREE.Color('#ffffff')},uShade:{value:new THREE.Color('#a0abb8')},uHorizon:{value:new THREE.Color('#b8cad8')}};
 const clouds=new THREE.Mesh(new THREE.SphereGeometry(900,48,20,0,Math.PI*2,0,Math.PI*.52),new THREE.ShaderMaterial({vertexShader:CLOUD_VERTEX,fragmentShader:CLOUD_FRAGMENT,uniforms:cloudUniforms,transparent:true,depthWrite:false,side:THREE.BackSide,fog:false}));
 clouds.renderOrder=1;clouds.frustumCulled=false;scene.add(clouds);
 const hemi=new THREE.HemisphereLight('#cfe2ff','#8b7458',.34);scene.add(hemi);
 // Cascaded shadow maps: the near cascade around the camera keeps shadows sharp on the street, the far one
 // still shadows the rest of the area; the cascades fade into each other.
 const csm=new CSM({camera,parent:scene,cascades:2,maxFar:360,mode:'practical',shadowMapSize:2048,shadowBias:-.00012,lightDirection:new THREE.Vector3(0,-1,0),lightIntensity:3,lightNear:1,lightFar:1500,lightMargin:160});
 csm.fade=true;
 for(const light of csm.lights){light.shadow.normalBias=.03;light.shadow.radius=2;}
 // The far cascade covers the static world: it is redrawn when the camera has moved 15 m or half a second has
 // passed, not every frame; its light keeps the matrix of its last render, so the map and the lookup stay paired.
 const farCascade=csm.lights[1];farCascade.shadow.autoUpdate=false;farCascade.shadow.needsUpdate=true;const farCascadeAt=new THREE.Vector3(Infinity,Infinity,Infinity);let farCascadeTime=-Infinity;
 const pmrem=new THREE.PMREMGenerator(renderer);let environment:THREE.WebGLRenderTarget|null=null;
 let current:Preset='afternoon';let lastFov=camera.fov,lastAspect=camera.aspect;const sunDir=new THREE.Vector3();
 function apply(preset:Preset){
  const s=SPECS[preset];const LOOK_=LOOK;current=preset;sunDir.copy(sunDirection(s.azimuthDeg,s.elevationDeg));
  const u=sky.material.uniforms;u.turbidity.value=s.turbidity;u.rayleigh.value=s.rayleigh;u.mieCoefficient.value=s.mie;u.mieDirectionalG.value=s.mieG;u.sunPosition.value.copy(sunDir);
  csm.lightDirection.copy(sunDir).negate();csm.lightIntensity=s.sunIntensity;
  for(const light of csm.lights){light.color.set(s.sunColor);light.intensity=s.sunIntensity;}
  hemi.color.set(s.hemiSky);hemi.groundColor.set(s.hemiGround);hemi.intensity=s.hemiIntensity;
  // Exponential haze reaching half density near the far end of the linear ramp it replaces: streets stay crisp, the
  // edge of the modelled area fades into the horizon instead of ending against a flat background.
  (sky.material as THREE.ShaderMaterial).uniforms.uHaze.value.set(s.fog[0]);
  scene.fog=new THREE.FogExp2(s.fog[0],Math.sqrt(Math.LN2)/(s.fog[1]+.55*(s.fog[2]-s.fog[1])));scene.background=new THREE.Color(s.fog[0]);
  renderer.toneMappingExposure=s.exposure;
  cloudUniforms.uCoverage.value=s.coverage;cloudUniforms.uOpacity.value=s.cloudOpacity;cloudUniforms.uSunDir.value.copy(sunDir);
  cloudUniforms.uTint.value.set(s.cloudTint);cloudUniforms.uShade.value.set(s.cloudShade);cloudUniforms.uHorizon.value.set(s.fog[0]);
  // The environment is the sky itself, filtered: glazing, paint and metal reflect the preset's sky, and shaded
  // walls are lit by it.
  const envScene=new THREE.Scene();const envSky=new Sky();envSky.scale.setScalar(1600);patchSky(envSky,false);const eu=envSky.material.uniforms;
  eu.turbidity.value=s.turbidity;eu.rayleigh.value=s.rayleigh;eu.mieCoefficient.value=s.mie;eu.mieDirectionalG.value=s.mieG;eu.sunPosition.value.copy(sunDir);envScene.add(envSky);
  eu.uSkySat.value=LOOK_.envSat;eu.uSkyScale.value=SKY_LOOK.scale*LOOK_.skyK;
  const sunE=s.sunIntensity*Math.max(0,sunDir.y),sc=new THREE.Color(s.sunColor),gs=LOOK_.bounce/.2;
  const groundMat=new THREE.MeshBasicMaterial({fog:false,side:THREE.DoubleSide});
  groundMat.color.setRGB((sc.r*sunE+LOOK_.skyIrradiance[0])*LOOK_.groundAlbedo[0]*gs/Math.PI,(sc.g*sunE+LOOK_.skyIrradiance[1])*LOOK_.groundAlbedo[1]*gs/Math.PI,(sc.b*sunE+LOOK_.skyIrradiance[2])*LOOK_.groundAlbedo[2]*gs/Math.PI);
  const ground=new THREE.Mesh(new THREE.CircleGeometry(1500,48),groundMat);ground.rotation.x=-Math.PI/2;ground.position.y=-1.5;envScene.add(ground);
  const next=pmrem.fromScene(envScene,.04,.1,10000);environment?.dispose();environment=next;scene.environment=next.texture;scene.environmentIntensity=s.env;
  envSky.geometry.dispose();envSky.material.dispose();ground.geometry.dispose();groundMat.dispose();
 }
 function setupMaterial(material:THREE.Material|THREE.Material[]){for(const m of Array.isArray(material)?material:[material]){if(m instanceof THREE.MeshStandardMaterial||m instanceof THREE.MeshLambertMaterial||m instanceof THREE.MeshPhongMaterial){csm.setupMaterial(m);}}}
 function update(now:number){
  cloudUniforms.uTime.value=now/1000*.06*SPECS[current].cloudSpeed;
  clouds.position.set(camera.position.x,0,camera.position.z);
  if(camera.fov!==lastFov||camera.aspect!==lastAspect){lastFov=camera.fov;lastAspect=camera.aspect;csm.updateFrustums();}
  csm.update();
  if(camera.position.distanceToSquared(farCascadeAt)>225||now-farCascadeTime>500){farCascade.shadow.needsUpdate=true;farCascadeAt.copy(camera.position);farCascadeTime=now;}
 }
 apply('afternoon');
 return {apply,update,setupMaterial,current:()=>current,resize:()=>csm.updateFrustums(),sunDirection:()=>sunDir.clone(),csm,sky,clouds,hemi,
  dispose:()=>{csm.dispose();environment?.dispose();pmrem.dispose();scene.remove(sky,clouds,hemi);sky.geometry.dispose();sky.material.dispose();clouds.geometry.dispose();(clouds.material as THREE.Material).dispose();}};
}
export type Atmosphere=ReturnType<typeof createAtmosphere>;
