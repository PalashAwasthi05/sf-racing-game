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

// The car on its own for model review: the world's sky, sun, shadows, grade and a street-like ground, an orbit camera,
// and a review hook (`__studio`) for fixed-pose captures and reloading the model without reloading the page.
export default function CarStudio(){
 const mount=useRef<HTMLDivElement>(null);const [status,setStatus]=useState('Loading the car…');
 useEffect(()=>{
  const host=mount.current;if(!host)return;let disposed=false;
  const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(40,1,.1,2000);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.85;
  host.appendChild(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.55,0);camera.position.set(4.6,1.6,5.2);controls.enableDamping=true;controls.maxPolarAngle=Math.PI*.495;
  const atmosphere=createAtmosphere(scene,renderer,camera);
  const composer=new EffectComposer(renderer,new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,samples:4}));composer.addPass(new RenderPass(scene,camera));
  const ao=new GTAOPass(scene,camera,1,1);ao.updateGtaoMaterial({radius:.6,distanceExponent:1.4,thickness:.6,samples:16});ao.blendIntensity=.9;composer.addPass(ao);
  const grade=createGrade(composer);   // its last pass tone maps and draws to the screen
  // ground: a large asphalt disc with a faint aggregate texture
  const tex=(()=>{const c=document.createElement('canvas');c.width=c.height=512;const g=c.getContext('2d')!;g.fillStyle='#5b5c5d';g.fillRect(0,0,512,512);for(let i=0;i<9000;i++){const v=70+Math.random()*50;g.fillStyle=`rgb(${v},${v},${v+2})`;g.fillRect(Math.random()*512,Math.random()*512,1.4,1.4);}const t=new THREE.CanvasTexture(c);t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(60,60);t.colorSpace=THREE.SRGBColorSpace;t.anisotropy=8;return t;})();
  const ground=new THREE.Mesh(new THREE.CircleGeometry(120,96),new THREE.MeshStandardMaterial({map:tex,roughness:.92,color:'#9a9a98'}));ground.rotation.x=-Math.PI/2;ground.receiveShadow=true;scene.add(ground);atmosphere.setupMaterial(ground.material);
  // a local reflection probe at the car so the paint reflects the ground and sky around it
  const probeTarget=new THREE.WebGLCubeRenderTarget(256,{type:THREE.HalfFloatType,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});const probe=new THREE.CubeCamera(.1,500,probeTarget);probe.position.set(0,.9,0);const pmrem=new THREE.PMREMGenerator(renderer);let probeEnv:THREE.WebGLRenderTarget|null=null;
  let car:THREE.Group|null=null;let materials:THREE.Material[]=[];
  function updateProbe(){if(car)car.visible=false;probe.update(renderer,scene);if(car)car.visible=true;const next=pmrem.fromCubemap(probeTarget.texture);probeEnv?.dispose();probeEnv=next;for(const m of materials){if(m instanceof THREE.MeshStandardMaterial&&m.roughness<.6){m.envMap=next.texture;m.envMapIntensity=1;m.needsUpdate=true;}}}
  async function load(){
   // `?model=http://127.0.0.1:<port>/<file>.glb` reviews a model straight from a local build output (no page reload)
   const requested=new URL(window.location.href).searchParams.get('model');const url=requested&&/^http:\/\/127\.0\.0\.1:\d+\/[\w.-]+\.glb$/.test(requested)?requested+'?'+Date.now():'/vehicle/roadster.glb';
   const source=await loadRoadster(url);if(disposed)return;if(!source){setStatus('No car model at '+url);return;}
   if(car){scene.remove(car);car.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});for(const m of materials)m.dispose();}
   const built=buildRoadster(source);car=built.car;materials=Object.values(built.materials);for(const m of materials)atmosphere.setupMaterial(m);scene.add(car);updateProbe();setStatus('');
  }
  function size(w=host!.clientWidth,h=host!.clientHeight){if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h,false);composer.setSize(w,h);ao.setSize(w,h);atmosphere.resize();}
  const resize=new ResizeObserver(()=>size());resize.observe(host);size();
  let frame=0;function animate(now:number){frame=requestAnimationFrame(animate);if(camera.up.y>.9999)controls.update();atmosphere.update(now);composer.render();}
  frame=requestAnimationFrame(animate);void load();
  (window as unknown as {__studio?:unknown}).__studio={
   reload:load,
   pose:(from:number[],at:number[],fov=40,up=[0,1,0])=>{camera.position.fromArray(from);camera.up.fromArray(up);controls.target.fromArray(at);camera.fov=fov;camera.updateProjectionMatrix();camera.lookAt(controls.target);},
   preset:(p:string)=>{if((PRESET_ORDER as string[]).includes(p)){atmosphere.apply(p as Preset);grade.preset(p as Preset);updateProbe();}},
   capture:(w=1600,h=1000)=>new Promise<Blob|null>(resolve=>{size(w,h);if(camera.up.y>.9999)controls.update();else camera.lookAt(controls.target);atmosphere.update(performance.now()+1000);for(const l of atmosphere.csm.lights)l.shadow.needsUpdate=true;composer.render();renderer.domElement.toBlob(b=>{size();resolve(b);},'image/png');}),
   // flat mask of the car on black, for silhouette overlays on reference photos
   mask:(w=1280,h=960)=>new Promise<Blob|null>(resolve=>{if(!car){resolve(null);return;}const flat=new THREE.MeshBasicMaterial({color:'#ff00ff'});const kept=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();
    car.traverse(o=>{if(o instanceof THREE.Mesh){kept.set(o,o.material);o.material=flat;}});const hidden=scene.children.filter(o=>o!==car&&o.visible);hidden.forEach(o=>{o.visible=false;});const bg=scene.background;scene.background=new THREE.Color('#000');
    size(w,h);if(camera.up.y>.9999)controls.update();else camera.lookAt(controls.target);renderer.render(scene,camera);renderer.domElement.toBlob(b=>{kept.forEach((m,o)=>{o.material=m;});hidden.forEach(o=>{o.visible=true;});scene.background=bg;flat.dispose();size();resolve(b);},'image/png');}),
   project:(points:number[][],w=1280,h=960)=>{const aspect=camera.aspect;camera.aspect=w/h;camera.updateProjectionMatrix();camera.updateMatrixWorld();const out=points.map(p=>{const v=new THREE.Vector3(p[0],p[1],p[2]).project(camera);return [(v.x+1)/2,(1-v.y)/2];});camera.aspect=aspect;camera.updateProjectionMatrix();return out;},
   scene,camera,renderer,get car(){return car;},get materials(){return materials;}};
  return()=>{disposed=true;cancelAnimationFrame(frame);resize.disconnect();controls.dispose();grade.dispose();atmosphere.dispose();pmrem.dispose();probeTarget.dispose();probeEnv?.dispose();renderer.dispose();renderer.domElement.remove();};
 },[]);
 return <main style={{position:'fixed',inset:0,background:'#9aa3ab'}}><div ref={mount} style={{position:'absolute',inset:0}} aria-label="Car model review"/>{status&&<p style={{position:'absolute',left:16,top:12,color:'#111',font:'14px system-ui'}}>{status}</p>}</main>;
}
