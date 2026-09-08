import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {Sky} from 'three/addons/objects/Sky.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {GTAOPass} from 'three/addons/postprocessing/GTAOPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';
import type {World} from './king-pilot';

export async function createWorldView(host:HTMLDivElement,data:World,onSelect:(id:string)=>void,onProgress:(message:string)=>void){
 const scene=new THREE.Scene();scene.background=new THREE.Color('#91b8d7');scene.fog=new THREE.Fog('#a9c3d4',650,1700);
 const camera=new THREE.PerspectiveCamera(58,1,.08,1800);
 const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
 renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.AgXToneMapping;renderer.toneMappingExposure=.85;
 host.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','3D neighborhood. Use the camera controls to explore.');
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=4;controls.maxDistance=700;controls.maxPolarAngle=Math.PI*.49;
 const sky=new Sky();sky.scale.setScalar(1600);sky.material.uniforms.turbidity.value=2.6;sky.material.uniforms.rayleigh.value=2;sky.material.uniforms.mieCoefficient.value=.003;sky.material.uniforms.mieDirectionalG.value=.8;
 const sunDirection=new THREE.Vector3(-.25,.55,.80).normalize();sky.material.uniforms.sunPosition.value.copy(sunDirection);scene.add(sky);
 const pmrem=new THREE.PMREMGenerator(renderer);const environmentScene=new THREE.Scene();environmentScene.add(sky.clone());const env=pmrem.fromScene(environmentScene,.03,.1,10000);scene.environment=env.texture;scene.environmentIntensity=.12;pmrem.dispose();
 const hemi=new THREE.HemisphereLight('#d5e5ff','#75624c',.18);scene.add(hemi);
 const sun=new THREE.DirectionalLight('#fff2df',3.1);sun.position.copy(sunDirection).multiplyScalar(260);sun.castShadow=true;sun.shadow.mapSize.set(4096,4096);Object.assign(sun.shadow.camera,{left:-65,right:65,top:65,bottom:-65,near:1,far:500});sun.shadow.bias=-.00003;sun.shadow.normalBias=.018;scene.add(sun,sun.target);
 const composer=new EffectComposer(renderer);composer.addPass(new RenderPass(scene,camera));
 const ao=new GTAOPass(scene,camera,1,1);ao.updateGtaoMaterial({radius:1.25,distanceExponent:1.5,thickness:.8,samples:12});ao.blendIntensity=.7;composer.addPass(ao);const output=new OutputPass();composer.addPass(output);
 function lighting(overcast:boolean){sun.intensity=overcast?.55:3.1;scene.environmentIntensity=overcast?.22:.12;hemi.intensity=overcast?.3:.18;renderer.toneMappingExposure=overcast?1.05:.85;sky.material.uniforms.turbidity.value=overcast?10:2.6;}
 const manifest=await fetch('/model.json',{cache:'no-cache'}).then(r=>{if(!r.ok)throw new Error('Scene manifest unavailable');return r.json() as Promise<{chunks:string[];compressedBytes:number}>});
 let loaded=0;
 const packed=new ReadableStream<BufferSource>({async start(controller){try{for(const path of manifest.chunks){const response=await fetch(path);if(!response.ok||!response.body)throw new Error('Scene download failed');const reader=response.body.getReader();while(true){const {done,value}=await reader.read();if(done)break;loaded+=value.byteLength;onProgress('Loading detailed scene… '+Math.round(loaded/manifest.compressedBytes*100)+'%');controller.enqueue(value)}}controller.close()}catch(e){controller.error(e)}}});
 const buffer=await new Response(packed.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();onProgress('Preparing materials and geometry…');
 const downloadUrl=URL.createObjectURL(new Blob([buffer],{type:'model/gltf-binary'}));
 const model=await new GLTFLoader().parseAsync(buffer,'/');scene.add(model.scene);
 const objects:THREE.Mesh[]=[];const originals=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();const evidences=new Map<THREE.Mesh,THREE.Material>();
 model.scene.traverse(o=>{if(!(o instanceof THREE.Mesh))return;objects.push(o);o.castShadow=!['terrain','roads','markings','sidewalks','cycleways'].some(n=>o.name.startsWith(n));o.receiveShadow=true;
  if(o.material instanceof THREE.MeshStandardMaterial){for(const t of [o.material.map,o.material.normalMap,o.material.roughnessMap]){if(t){t.anisotropy=Math.min(8,renderer.capabilities.getMaxAnisotropy());t.needsUpdate=true;}}}
  originals.set(o,o.material);
 });
 const ring=new THREE.Mesh(new THREE.RingGeometry(199.65,200,160),new THREE.MeshBasicMaterial({color:'#e3c78b',transparent:true,opacity:.65,side:THREE.DoubleSide}));ring.rotation.x=-Math.PI/2;ring.position.y=.22;scene.add(ring);
 const highlight=new THREE.Box3Helper(new THREE.Box3(),new THREE.Color('#f5c76b'));highlight.visible=false;scene.add(highlight);
 function getLayer(o:THREE.Object3D){return String(o.userData.layer||o.name.split('___')[0].split('__')[0])}
 function getId(o:THREE.Object3D){return String(o.userData.featureId||o.name.split('___')[0].split('__')[1]||'')}
 function groundAt(x:number,z:number){const y=-z;const {min,step,n}=data.grid;const px=Math.max(0,Math.min(n-1.00001,(x-min)/step)),py=Math.max(0,Math.min(n-1.00001,(y-min)/step)),ix=Math.floor(px),iy=Math.floor(py),fx=px-ix,fy=py-iy;const a=data.ground[iy*n+ix],b=data.ground[iy*n+ix+1],c=data.ground[(iy+1)*n+ix],d=data.ground[(iy+1)*n+ix+1];return fx+fy<=1?a+(b-a)*fx+(c-a)*fy:d+(c-d)*(1-fx)+(b-d)*(1-fy)}

 function highlightId(id:string){const box=new THREE.Box3();objects.filter(o=>getId(o)===id&&getLayer(o)==='buildings').forEach(o=>box.expandByObject(o));if(!box.isEmpty()){highlight.box.copy(box);highlight.visible=true;}onSelect(id)}
 let mode='street',yaw=.785,pitch=0,drag=false,previousX=0,previousY=0,startX=0,startY=0,station=0;
 const keys=new Set<string>();let targetPosition:THREE.Vector3|null=null,targetLook:THREE.Vector3|null=null;
 function view(v:string){mode=v;controls.enabled=v!=='street';keys.clear();
  if(v==='street'){targetPosition=null;targetLook=null;const p=data.streetViews?.[station];const x=p?.position[0]??37,z=-(p?.position[1]??-38);camera.position.set(x,groundAt(x,z)+1.25,z);camera.rotation.order='YXZ';if(p){camera.lookAt(p.lookAt[0],groundAt(p.lookAt[0],-p.lookAt[1])+p.lookAt[2],-p.lookAt[1]);yaw=camera.rotation.y;pitch=camera.rotation.x;}else{yaw=-.785;pitch=.025;camera.rotation.set(pitch,yaw,0);}highlight.visible=false;}
  else if(v==='top'){controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(0,490,.1);targetLook=new THREE.Vector3(0,0,0)}
  else{controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(150,115,185);targetLook=new THREE.Vector3(15,5,-8)}
 }
 function select(id:string){const f=data.buildings.find(b=>b.id===id);if(!f)return;highlightId(id);mode='orbit';controls.enabled=true;targetLook=new THREE.Vector3(f.center[0],f.center[2]+f.height*.35,-f.center[1]);targetPosition=targetLook.clone().add(new THREE.Vector3(65,40,75));}
 function nextStreet(){station=(station+1)%Math.max(1,data.streetViews?.length||1);view('street');}
 camera.position.set(102,74,125);controls.target.set(1,12,0);controls.update();
 const target=[...data.buildings].sort((a,b)=>Math.hypot(a.center[0],a.center[1])-Math.hypot(b.center[0],b.center[1]))[0];if(target)highlightId(target.id);view('street');
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 function down(e:PointerEvent){drag=true;startX=previousX=e.clientX;startY=previousY=e.clientY;targetPosition=null;targetLook=null;}
 function move(e:PointerEvent){if(mode==='street'&&drag){yaw-=(e.clientX-previousX)*.004;pitch=Math.max(-1.2,Math.min(1.2,pitch-(e.clientY-previousY)*.004));camera.rotation.set(pitch,yaw,0);}previousX=e.clientX;previousY=e.clientY;}
 function up(e:PointerEvent){const wasDragging=drag;drag=false;if(!wasDragging||Math.hypot(e.clientX-startX,e.clientY-startY)>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(objects,false).find(h=>h.object.visible&&getId(h.object));if(hit)highlightId(getId(hit.object));}
 function keydown(e:KeyboardEvent){if((e.target as HTMLElement)?.closest('input,textarea,[contenteditable="true"],[role="dialog"]'))return;if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight'].includes(e.code)){keys.add(e.code);if(mode==='street')e.preventDefault();}}
 function keyup(e:KeyboardEvent){keys.delete(e.code)}
 function blur(){keys.clear();drag=false}
 renderer.domElement.addEventListener('pointerdown',down);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('keydown',keydown);window.addEventListener('keyup',keyup);window.addEventListener('blur',blur);
 const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);composer.setSize(w,h);ao.setSize(Math.round(w*renderer.getPixelRatio()*.75),Math.round(h*renderer.getPixelRatio()*.75))});resize.observe(host);
 let frame=0,previous=performance.now();const forward=new THREE.Vector3(),right=new THREE.Vector3();
 function animate(now:number){frame=requestAnimationFrame(animate);const dt=Math.min((now-previous)/1000,.05);previous=now;
  if(mode==='street'){
   camera.getWorldDirection(forward);forward.y=0;forward.normalize();right.crossVectors(forward,new THREE.Vector3(0,1,0));const speed=dt*(keys.has('ShiftLeft')||keys.has('ShiftRight')?18:5);
   if(keys.has('KeyW')||keys.has('ArrowUp'))camera.position.addScaledVector(forward,speed);if(keys.has('KeyS')||keys.has('ArrowDown'))camera.position.addScaledVector(forward,-speed);if(keys.has('KeyD')||keys.has('ArrowRight'))camera.position.addScaledVector(right,speed);if(keys.has('KeyA')||keys.has('ArrowLeft'))camera.position.addScaledVector(right,-speed);
   const len=Math.hypot(camera.position.x,camera.position.z);if(len>255){camera.position.x*=255/len;camera.position.z*=255/len;}camera.position.y=groundAt(camera.position.x,camera.position.z)+1.25;
  }else{if(targetPosition&&targetLook){camera.position.lerp(targetPosition,.07);controls.target.lerp(targetLook,.07);if(camera.position.distanceTo(targetPosition)<.05){targetPosition=null;targetLook=null;}}controls.update();}
  const anchor=mode==='street'?camera.position:controls.target;sun.target.position.set(Math.round(anchor.x/2)*2,0,Math.round(anchor.z/2)*2);sun.position.copy(sun.target.position).addScaledVector(sunDirection,260);sun.target.updateMatrixWorld();
  composer.render(dt);
 }
 frame=requestAnimationFrame(animate);
 return {downloadUrl,view,select,lighting,nextStreet,layer:(layer:string,on:boolean)=>{objects.filter(o=>(getLayer(o)===layer||(layer==='street_objects'&&getLayer(o)==='signs'))).forEach(o=>o.visible=on)},evidence:(on:boolean)=>{for(const o of objects){if(on){let m=evidences.get(o);if(!m){const l=getLayer(o);const color=['roofs','terrain'].includes(l)?'#56b3c4':o.userData.evidence==='visual-fit'?'#7caf8c':'#d2a25e';m=new THREE.MeshStandardMaterial({color,roughness:.9});evidences.set(o,m);}o.material=m;}else o.material=originals.get(o)!;}},dispose:()=>{cancelAnimationFrame(frame);resize.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointerdown',down);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('keydown',keydown);window.removeEventListener('keyup',keyup);window.removeEventListener('blur',blur);objects.forEach(o=>o.geometry.dispose());new Set([...originals.values()].flat()).forEach(m=>m.dispose());evidences.forEach(m=>m.dispose());ao.dispose();output.dispose();composer.dispose();env.dispose();ring.geometry.dispose();(ring.material as THREE.Material).dispose();highlight.geometry.dispose();(highlight.material as THREE.Material).dispose();sky.geometry.dispose();sky.material.dispose();URL.revokeObjectURL(downloadUrl);renderer.dispose();renderer.domElement.remove();}};
}
