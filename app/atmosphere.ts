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
 afternoon:{azimuthDeg:212,elevationDeg:38,sunColor:'#fff2dd',sunIntensity:3.0,hemiSky:'#cfe2ff',hemiGround:'#8b7458',hemiIntensity:.34,env:.42,exposure:.95,turbidity:3,rayleigh:2.3,mie:.004,mieG:.82,fog:['#b8cad8',700,1900],coverage:.44,cloudTint:'#ffffff',cloudShade:'#9fabb8',cloudOpacity:.92,cloudSpeed:1},
 golden:{azimuthDeg:262,elevationDeg:7.5,sunColor:'#ffbb72',sunIntensity:2.3,hemiSky:'#e6c6ae',hemiGround:'#5f4a3d',hemiIntensity:.3,env:.5,exposure:.95,turbidity:5.5,rayleigh:3.6,mie:.022,mieG:.93,fog:['#d6b69c',480,1600],coverage:.5,cloudTint:'#ffd6b0',cloudShade:'#7e6270',cloudOpacity:.95,cloudSpeed:1},
 overcast:{azimuthDeg:212,elevationDeg:45,sunColor:'#e9eef3',sunIntensity:.55,hemiSky:'#d8dee4',hemiGround:'#7b7973',hemiIntensity:.8,env:.55,exposure:1.06,turbidity:18,rayleigh:1.1,mie:.006,mieG:.7,fog:['#c4cbd2',420,1400],coverage:.97,cloudTint:'#e0e4e9',cloudShade:'#98a1aa',cloudOpacity:1,cloudSpeed:.6},
};
function sunDirection(azimuthDeg:number,elevationDeg:number){const az=THREE.MathUtils.degToRad(azimuthDeg),el=THREE.MathUtils.degToRad(elevationDeg);return new THREE.Vector3(Math.sin(az)*Math.cos(el),Math.sin(el),-Math.cos(az)*Math.cos(el)).normalize();}

const CLOUD_VERTEX=`varying vec3 vDir;void main(){vDir=(modelMatrix*vec4(position,1.)).xyz-cameraPosition;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const CLOUD_FRAGMENT=`
precision highp float;varying vec3 vDir;
uniform float uTime,uCoverage,uOpacity;uniform vec3 uSunDir,uTint,uShade,uHorizon;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*noise(p);p=p*2.03+vec2(17.,9.);a*=.5;}return v;}
void main(){
 vec3 d=normalize(vDir);if(d.y<.005)discard;
 // A flat cloud deck 1.5 km up, read where the view ray meets it; slow drift with the wind.
 vec2 uv=d.xz*(1500./d.y)*.00055+uTime*vec2(.011,.005);
 float n=fbm(uv)+.35*fbm(uv*3.1+vec2(4.2,1.7));
 float dens=smoothstep(1.05-uCoverage,1.32-uCoverage,n);
 float horizon=smoothstep(.015,.16,d.y);
 float towardSun=max(dot(d,uSunDir),0.);
 vec3 col=mix(uTint,uShade,dens*.72)*(.86+.28*(.5+.5*dot(d,uSunDir)));
 col+=uTint*pow(towardSun,10.)*.5*(1.-dens);
 col=mix(uHorizon,col,horizon);
 gl_FragColor=vec4(col,dens*horizon*uOpacity);
}`;

export function createAtmosphere(scene:THREE.Scene,renderer:THREE.WebGLRenderer,camera:THREE.PerspectiveCamera){
 const sky=new Sky();sky.scale.setScalar(1600);scene.add(sky);
 const cloudUniforms={uTime:{value:0},uCoverage:{value:.4},uOpacity:{value:.9},uSunDir:{value:new THREE.Vector3(0,1,0)},uTint:{value:new THREE.Color('#ffffff')},uShade:{value:new THREE.Color('#a0abb8')},uHorizon:{value:new THREE.Color('#b8cad8')}};
 const clouds=new THREE.Mesh(new THREE.SphereGeometry(900,48,20,0,Math.PI*2,0,Math.PI*.52),new THREE.ShaderMaterial({vertexShader:CLOUD_VERTEX,fragmentShader:CLOUD_FRAGMENT,uniforms:cloudUniforms,transparent:true,depthWrite:false,side:THREE.BackSide,fog:false}));
 clouds.renderOrder=1;clouds.frustumCulled=false;scene.add(clouds);
 const hemi=new THREE.HemisphereLight('#cfe2ff','#8b7458',.34);scene.add(hemi);
 // Cascaded shadow maps: the near cascade around the camera keeps shadows sharp on the street, the far one
 // still shadows the rest of the area; the cascades fade into each other.
 const csm=new CSM({camera,parent:scene,cascades:2,maxFar:360,mode:'practical',shadowMapSize:3072,shadowBias:-.00012,lightDirection:new THREE.Vector3(0,-1,0),lightIntensity:3,lightNear:1,lightFar:1500,lightMargin:160});
 csm.fade=true;
 for(const light of csm.lights){light.shadow.normalBias=.03;light.shadow.radius=2;}
 const pmrem=new THREE.PMREMGenerator(renderer);let environment:THREE.WebGLRenderTarget|null=null;
 let current:Preset='afternoon';let lastFov=camera.fov,lastAspect=camera.aspect;const sunDir=new THREE.Vector3();
 function apply(preset:Preset){
  const s=SPECS[preset];current=preset;sunDir.copy(sunDirection(s.azimuthDeg,s.elevationDeg));
  const u=sky.material.uniforms;u.turbidity.value=s.turbidity;u.rayleigh.value=s.rayleigh;u.mieCoefficient.value=s.mie;u.mieDirectionalG.value=s.mieG;u.sunPosition.value.copy(sunDir);
  csm.lightDirection.copy(sunDir).negate();csm.lightIntensity=s.sunIntensity;
  for(const light of csm.lights){light.color.set(s.sunColor);light.intensity=s.sunIntensity;}
  hemi.color.set(s.hemiSky);hemi.groundColor.set(s.hemiGround);hemi.intensity=s.hemiIntensity;
  scene.fog=new THREE.Fog(s.fog[0],s.fog[1],s.fog[2]);scene.background=new THREE.Color(s.fog[0]);
  renderer.toneMappingExposure=s.exposure;
  cloudUniforms.uCoverage.value=s.coverage;cloudUniforms.uOpacity.value=s.cloudOpacity;cloudUniforms.uSunDir.value.copy(sunDir);
  cloudUniforms.uTint.value.set(s.cloudTint);cloudUniforms.uShade.value.set(s.cloudShade);cloudUniforms.uHorizon.value.set(s.fog[0]);
  // The environment is the sky itself, filtered: glazing, paint and metal reflect the preset's sky, and shaded
  // walls are lit by it.
  const envScene=new THREE.Scene();const envSky=new Sky();envSky.scale.setScalar(1600);const eu=envSky.material.uniforms;
  eu.turbidity.value=s.turbidity;eu.rayleigh.value=s.rayleigh;eu.mieCoefficient.value=s.mie;eu.mieDirectionalG.value=s.mieG;eu.sunPosition.value.copy(sunDir);envScene.add(envSky);
  const next=pmrem.fromScene(envScene,.04,.1,10000);environment?.dispose();environment=next;scene.environment=next.texture;scene.environmentIntensity=s.env;
  envSky.geometry.dispose();envSky.material.dispose();
 }
 function setupMaterial(material:THREE.Material|THREE.Material[]){for(const m of Array.isArray(material)?material:[material]){if(m instanceof THREE.MeshStandardMaterial||m instanceof THREE.MeshLambertMaterial||m instanceof THREE.MeshPhongMaterial){csm.setupMaterial(m);}}}
 function update(now:number){
  cloudUniforms.uTime.value=now/1000*.06*SPECS[current].cloudSpeed;
  clouds.position.set(camera.position.x,0,camera.position.z);
  if(camera.fov!==lastFov||camera.aspect!==lastAspect){lastFov=camera.fov;lastAspect=camera.aspect;csm.updateFrustums();}
  csm.update();
 }
 apply('afternoon');
 return {apply,update,setupMaterial,current:()=>current,resize:()=>csm.updateFrustums(),sunDirection:()=>sunDir.clone(),
  dispose:()=>{csm.dispose();environment?.dispose();pmrem.dispose();scene.remove(sky,clouds,hemi);sky.geometry.dispose();sky.material.dispose();clouds.geometry.dispose();(clouds.material as THREE.Material).dispose();}};
}
export type Atmosphere=ReturnType<typeof createAtmosphere>;
