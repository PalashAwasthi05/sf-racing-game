'use client';
import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {createAtmosphere,PRESET_ORDER,type Preset} from './atmosphere';
import {createGrade} from './grade';
import {loadRoadster,buildRoadster} from './car';

// The car on its own for model review: a neutral studio, opt-in daylight, an orbit camera,
// and a review hook (`__studio`) for fixed-pose captures and reloading the model without reloading the page.
export default function CarStudio(){
 const mount=useRef<HTMLDivElement>(null);const [status,setStatus]=useState('Loading the car…');const [lighting,setLighting]=useState('studio');
 const lightingAction=useRef<(p:string)=>void>(()=>{});
 useEffect(()=>{
  const host=mount.current;if(!host)return;let disposed=false;
  const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(34,1,.1,2000);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;
  // setSize(..., false) changes raster dimensions only. Keep the display size
  // tied to the host so Retina pixels do not enlarge and crop the review view.
  renderer.domElement.style.width='100%';renderer.domElement.style.height='100%';renderer.domElement.style.display='block';
  host.appendChild(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.55,0);camera.position.set(4.6,1.6,5.2);controls.enableDamping=true;controls.maxPolarAngle=Math.PI*.495;
  const atmosphere=createAtmosphere(scene,renderer,camera,'overcast',null);
  const composer=new EffectComposer(renderer,new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,samples:4}));composer.addPass(new RenderPass(scene,camera));
  const ao=new GTAOPass(scene,camera,1,1);ao.updateGtaoMaterial({radius:.6,distanceExponent:1.4,thickness:.6,samples:16});ao.blendIntensity=.9;composer.addPass(ao);
  const grade=createGrade(composer);   // its last pass tone maps and draws to the screen
  // Daylight has an uninterrupted level ground. The separate studio sweep
  // expands around the camera, including distant, narrow-FOV reference poses.
  const studioSweep=new THREE.Mesh(new THREE.LatheGeometry([
   new THREE.Vector2(0,0),
   ...Array.from({length:25},(_,i)=>{const a=i/24*Math.PI/2;return new THREE.Vector2(6+4*Math.sin(a),4-4*Math.cos(a));}),
   new THREE.Vector2(10,10)],128),new THREE.MeshStandardMaterial({color:'#66686a',roughness:.72,side:THREE.DoubleSide}));
  studioSweep.position.y=-.015;studioSweep.receiveShadow=true;scene.add(studioSweep);
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(20000,20000),studioSweep.material);ground.rotation.x=-Math.PI/2;ground.position.y=-.015;ground.receiveShadow=true;scene.add(ground);atmosphere.setupMaterial(ground.material);
  function updateStage(){
   const studio=lightMode==='studio';ground.visible=!studio;studioSweep.visible=studio;
   if(studio)studioSweep.scale.setScalar(Math.max(2.5,(Math.hypot(camera.position.x,camera.position.z)+8)/6));
  }
  // a local reflection probe at the car so the paint reflects the ground and sky around it
  const probeTarget=new THREE.WebGLCubeRenderTarget(256,{type:THREE.HalfFloatType,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});const probe=new THREE.CubeCamera(.1,500,probeTarget);probe.position.set(0,.9,0);const pmrem=new THREE.PMREMGenerator(renderer);let probeEnv:THREE.WebGLRenderTarget|null=null;
  // Linear HDR radiance from broad softboxes; a dark lower hemisphere lets
  // white paint describe the body rather than reflect a uniformly white dome.
  const width=512,height=256,radiance=new Float32Array(width*height*4);
  const wrap=(a:number)=>Math.atan2(Math.sin(a),Math.cos(a));
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const az=(x+.5)/width*Math.PI*2-Math.PI,el=Math.PI/2-(y+.5)/height*Math.PI;
   let value=el>0?.28:.14;
   for(const [a,e,w,h,power] of [[-.7,.78,.68,.14,7],[2.1,.55,.48,.11,5],[.65,.12,.14,.55,2.4]]){
    const shape=(Math.abs(wrap(az-a))/w)**8+(Math.abs(el-e)/h)**8;value+=power*Math.exp(-shape);
   }
   const k=(y*width+x)*4;radiance[k]=radiance[k+1]=radiance[k+2]=value;radiance[k+3]=1;
  }
  const studioMap=new THREE.DataTexture(radiance,width,height,THREE.RGBAFormat,THREE.FloatType);studioMap.mapping=THREE.EquirectangularReflectionMapping;studioMap.colorSpace=THREE.LinearSRGBColorSpace;studioMap.needsUpdate=true;
  const studioEnv=pmrem.fromEquirectangular(studioMap);studioMap.dispose();
  let car:THREE.Group|null=null;let materials:THREE.Material[]=[];
  let lightMode='studio',loadVersion=0,autoFrame=true;
  controls.addEventListener('start',()=>{autoFrame=false;});
  function frameCar(){if(!car||!autoFrame)return;
   const box=new THREE.Box3().setFromObject(car),center=box.getCenter(new THREE.Vector3());controls.target.copy(center);
   const backward=new THREE.Vector3(1,.24,1.05).normalize(),right=new THREE.Vector3().crossVectors(new THREE.Vector3(0,1,0),backward).normalize(),up=new THREE.Vector3().crossVectors(backward,right);
   const tanV=Math.tan(THREE.MathUtils.degToRad(camera.fov/2)),tanH=tanV*camera.aspect;let distance=0;
   for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){
    const p=new THREE.Vector3(x,y,z).sub(center),depth=p.dot(backward);
    distance=Math.max(distance,depth+Math.abs(p.dot(right))/(tanH*.83),depth+Math.abs(p.dot(up))/(tanV*.72));
   }
   camera.position.copy(center).addScaledVector(backward,distance);camera.lookAt(center);controls.update();
  }
  function preset(p:string){if(p!=='studio'&&!(PRESET_ORDER as string[]).includes(p))return;
   lightMode=p;setLighting(p);grade.preset(p==='studio'?'overcast':p);
   if(p==='studio'){
    atmosphere.sky.visible=false;atmosphere.clouds.visible=false;atmosphere.hemi.color.set('#ffffff');atmosphere.hemi.groundColor.set('#b8bdc2');atmosphere.hemi.intensity=.55;
    scene.background=new THREE.Color('#45484b');scene.fog=null;scene.environment=studioEnv.texture;scene.environmentIntensity=1;
    atmosphere.csm.maxFar=30;atmosphere.csm.lightDirection.set(-.5,-1,-.35).normalize();
    atmosphere.csm.lights.forEach(l=>{l.color.set('#ffffff');l.intensity=1.3;l.shadow.normalBias=.008;l.shadow.radius=3;l.shadow.needsUpdate=true;});
    renderer.toneMappingExposure=1;
    // Neutral output for inspecting paint and glass without a daylight grade.
    const u=grade.grade.uniforms;u.uWarm.value=0;u.uSaturation.value=0;u.uVibrance.value=0;u.uContrast.value=1.05;u.uVignette.value=.1;grade.bloom.strength=.025;
    ao.blendIntensity=.5;ao.updateGtaoMaterial({radius:.35,thickness:.22});
   }else{atmosphere.sky.visible=true;atmosphere.csm.maxFar=360;atmosphere.csm.lights.forEach(l=>{l.shadow.normalBias=.03;l.shadow.radius=1;});atmosphere.apply(p as Preset);ao.blendIntensity=.9;ao.updateGtaoMaterial({radius:.6,thickness:.6});}
   atmosphere.resize();updateStage();updateProbe();grade.snap();
  }
  lightingAction.current=preset;
  function updateProbe(){
   if(lightMode==='studio'){for(const m of materials)if(m instanceof THREE.MeshStandardMaterial){m.envMap=studioEnv.texture;m.envMapIntensity=1;m.needsUpdate=true;}return;}
   // Rough finishes return to the selected atmosphere instead of retaining
   // the studio softboxes. Only polished finishes use the local sky/ground probe.
   for(const m of materials)if(m instanceof THREE.MeshStandardMaterial){m.envMap=null;m.envMapIntensity=1;m.needsUpdate=true;}
   const disc=atmosphere.sky.material.uniforms.uDrawDisc, previousDisc=disc.value, visible=car?.visible;
   // Sunlight already contributes the specular highlight through the material shader.
   try{disc.value=0;if(car)car.visible=false;probe.update(renderer,scene);}
   finally{disc.value=previousDisc;if(car)car.visible=visible??true;}
   const next=pmrem.fromCubemap(probeTarget.texture);probeEnv?.dispose();probeEnv=next;for(const m of materials){if(m instanceof THREE.MeshStandardMaterial&&m.roughness<.6){m.envMap=next.texture;m.envMapIntensity=1;m.needsUpdate=true;}}}
  function releaseCar(){if(!car)return;scene.remove(car);
   const geometries=new Set<THREE.BufferGeometry>(),finishes=new Set<THREE.Material>(materials);
   car.traverse(o=>{if(o instanceof THREE.Mesh){geometries.add(o.geometry);for(const m of Array.isArray(o.material)?o.material:[o.material]){finishes.add(m);if(o.name==='rear_registration'&&m instanceof THREE.MeshStandardMaterial)m.map?.dispose();}}});
   geometries.forEach(g=>g.dispose());for(const m of finishes)m.dispose();car=null;materials=[];
  }
  async function load(){
   const version=++loadVersion;setStatus('Loading the car…');
   const requested=new URL(window.location.href).searchParams.get('model');const url=requested&&/^http:\/\/127\.0\.0\.1:\d+\/[\w.-]+\.glb$/.test(requested)?requested+'?'+Date.now():'/vehicle/roadster.glb';
   try{const source=await loadRoadster(url);
    if(disposed||version!==loadVersion){source?.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});return;}
    if(!source){setStatus('The car could not be loaded.');return;}
    releaseCar();const importedMaterials=new Set<THREE.Material>();source.traverse(o=>{if(o instanceof THREE.Mesh)for(const m of Array.isArray(o.material)?o.material:[o.material])importedMaterials.add(m);});const built=buildRoadster(source);importedMaterials.forEach(m=>m.dispose());car=built.car;materials=Object.values(built.materials);for(const m of materials)atmosphere.setupMaterial(m);scene.add(car);frameCar();updateProbe();grade.snap();setStatus('');
   }catch{if(!disposed&&version===loadVersion)setStatus('The car could not be loaded.');}
  }
  function size(w=host!.clientWidth,h=host!.clientHeight){if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false);composer.setSize(w,h);atmosphere.resize();}
  const resize=new ResizeObserver(()=>{size();frameCar();});resize.observe(host);size();
  const initialLight=new URL(window.location.href).searchParams.get('light')??'studio';preset(initialLight==='studio'||(PRESET_ORDER as string[]).includes(initialLight)?initialLight:'studio');
  let frame=0;function animate(now:number){frame=requestAnimationFrame(animate);if(camera.up.y>.9999)controls.update();updateStage();atmosphere.update(now);composer.render();}
  frame=requestAnimationFrame(animate);void load();
  (window as unknown as {__studio?:unknown}).__studio={
   reload:load,
   pose:(from:number[],at:number[],fov=40,up=[0,1,0])=>{autoFrame=false;camera.position.fromArray(from);camera.up.fromArray(up);controls.target.fromArray(at);camera.fov=fov;camera.updateProjectionMatrix();camera.lookAt(controls.target);updateStage();},
   preset,
   capture:(w=1600,h=1000)=>new Promise<Blob|null>(resolve=>{size(w,h);updateStage();if(camera.up.y>.9999)controls.update();else camera.lookAt(controls.target);atmosphere.update(performance.now()+1000);for(const l of atmosphere.csm.lights)l.shadow.needsUpdate=true;composer.render();renderer.domElement.toBlob(b=>{size();resolve(b);},'image/png');}),
   // flat mask of the car on black, for silhouette overlays on reference photos
   mask:(w=1280,h=960)=>new Promise<Blob|null>(resolve=>{if(!car){resolve(null);return;}const flat=new THREE.MeshBasicMaterial({color:'#ff00ff'});const kept=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();
    car.traverse(o=>{if(o instanceof THREE.Mesh){kept.set(o,o.material);o.material=flat;}});const hidden=scene.children.filter(o=>o!==car&&o.visible);hidden.forEach(o=>{o.visible=false;});const bg=scene.background;scene.background=new THREE.Color('#000');
    size(w,h);if(camera.up.y>.9999)controls.update();else camera.lookAt(controls.target);renderer.render(scene,camera);renderer.domElement.toBlob(b=>{kept.forEach((m,o)=>{o.material=m;});hidden.forEach(o=>{o.visible=true;});scene.background=bg;flat.dispose();size();resolve(b);},'image/png');}),
   project:(points:number[][],w=1280,h=960)=>{const aspect=camera.aspect;camera.aspect=w/h;camera.updateProjectionMatrix();camera.updateMatrixWorld();const out=points.map(p=>{const v=new THREE.Vector3(p[0],p[1],p[2]).project(camera);return [(v.x+1)/2,(1-v.y)/2];});camera.aspect=aspect;camera.updateProjectionMatrix();return out;},
   scene,camera,renderer,get car(){return car;},get materials(){return materials;}};
  return()=>{disposed=true;++loadVersion;lightingAction.current=()=>{};delete (window as unknown as {__studio?:unknown}).__studio;cancelAnimationFrame(frame);resize.disconnect();controls.dispose();releaseCar();ground.geometry.dispose();studioSweep.geometry.dispose();ground.material.dispose();grade.dispose();ao.dispose();composer.dispose();atmosphere.dispose();studioEnv.dispose();pmrem.dispose();probeTarget.dispose();probeEnv?.dispose();renderer.dispose();renderer.domElement.remove();};
 },[]);
 return <main style={{position:'fixed',inset:0,background:'#45484b'}}><div ref={mount} style={{position:'absolute',inset:0}} aria-label="Car model review"/><div style={{position:'absolute',right:18,top:18,display:'flex',gap:4,padding:4,borderRadius:20,background:'#202326cc',font:'12px system-ui'}}>{['studio','afternoon'].map(p=><button key={p} type="button" onClick={()=>lightingAction.current(p)} aria-pressed={lighting===p} style={{border:0,borderRadius:16,padding:'7px 13px',cursor:'pointer',background:lighting===p?'#ecece7':'transparent',color:lighting===p?'#202326':'#e4e7e8'}}>{p==='studio'?'Studio':'Daylight'}</button>)}<a href="/?drive=1" style={{padding:'7px 10px',color:'#e4e7e8',textDecoration:'none',whiteSpace:'nowrap'}}>Drive neighborhood ↗</a></div>{status&&<p style={{position:'absolute',left:18,bottom:18,color:'#f2f3f4',font:'14px system-ui'}}>{status}</p>}</main>;
}
