import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {World} from './king-pilot';

export async function createWorldView(host:HTMLDivElement,data:World,onSelect:(id:string)=>void){
 const scene=new THREE.Scene();scene.background=new THREE.Color('#bbd0db');scene.fog=new THREE.Fog('#bbd0db',420,950);
 const camera=new THREE.PerspectiveCamera(48,1,.12,1800);
 const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
 renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.6));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.45;
 host.appendChild(renderer.domElement);renderer.domElement.setAttribute('aria-label','3D neighborhood. Use the camera controls to explore.');
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=4;controls.maxDistance=700;controls.maxPolarAngle=Math.PI*.49;
 const hemi=new THREE.HemisphereLight('#e2f2ff','#869787',2.7);scene.add(hemi);
 const sun=new THREE.DirectionalLight('#fff2db',3.2);sun.position.set(-130,220,120);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-290,right:290,top:290,bottom:-290,near:1,far:700});sun.shadow.bias=-.0002;sun.shadow.normalBias=.12;scene.add(sun);
 const model=await new GLTFLoader().loadAsync('/king-street.glb');scene.add(model.scene);
 const objects:THREE.Mesh[]=[];const originals=new Map<THREE.Mesh,THREE.Material|THREE.Material[]>();const evidences=new Map<THREE.Mesh,THREE.Material>();
 model.scene.traverse(o=>{if(!(o instanceof THREE.Mesh))return;objects.push(o);o.castShadow=!['terrain','roads','markings'].some(n=>o.name.startsWith(n));o.receiveShadow=true;
  if(o.material instanceof THREE.MeshStandardMaterial){o.material.roughness=.82;o.material.metalness=.08;}
  originals.set(o,o.material);
 });
 const ring=new THREE.Mesh(new THREE.RingGeometry(199.65,200,160),new THREE.MeshBasicMaterial({color:'#e3c78b',transparent:true,opacity:.65,side:THREE.DoubleSide}));ring.rotation.x=-Math.PI/2;ring.position.y=.22;scene.add(ring);
 const highlight=new THREE.Box3Helper(new THREE.Box3(),new THREE.Color('#f5c76b'));highlight.visible=false;scene.add(highlight);
 function getLayer(o:THREE.Object3D){return String(o.userData.layer||o.name.split('__')[0])}
 function getId(o:THREE.Object3D){return String(o.userData.featureId||o.name.split('__')[1]||'')}
 function groundAt(x:number,z:number){const y=-z;const {min,step,n}=data.grid;const ix=Math.max(0,Math.min(n-1,Math.round((x-min)/step))),iy=Math.max(0,Math.min(n-1,Math.round((y-min)/step)));return data.ground[iy*n+ix]||0}
 function highlightId(id:string){const box=new THREE.Box3();objects.filter(o=>getId(o)===id&&getLayer(o)==='buildings').forEach(o=>box.expandByObject(o));if(!box.isEmpty()){highlight.box.copy(box);highlight.visible=true;}onSelect(id)}
 let mode='orbit',yaw=.785,pitch=0,drag=false,previousX=0,previousY=0,startX=0,startY=0;
 const keys=new Set<string>();let targetPosition:THREE.Vector3|null=null,targetLook:THREE.Vector3|null=null;
 function view(v:string){mode=v;controls.enabled=v!=='street';keys.clear();
  if(v==='street'){targetPosition=null;targetLook=null;camera.position.set(37,groundAt(37,38)+1.8,38);yaw=.785;pitch=.08;camera.rotation.order='YXZ';camera.rotation.set(pitch,yaw,0);}
  else if(v==='top'){controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(0,490,.1);targetLook=new THREE.Vector3(0,0,0)}
  else{controls.maxPolarAngle=Math.PI*.49;targetPosition=new THREE.Vector3(280,235,320);targetLook=new THREE.Vector3(15,5,-8)}
 }
 function select(id:string){const f=data.buildings.find(b=>b.id===id);if(!f)return;highlightId(id);mode='orbit';controls.enabled=true;targetLook=new THREE.Vector3(f.center[0],f.center[2]+f.height*.35,-f.center[1]);targetPosition=targetLook.clone().add(new THREE.Vector3(65,40,75));}
 camera.position.set(280,235,320);controls.target.set(15,5,-8);controls.update();
 const target=data.buildings.find(b=>b.name==='188 King Street');if(target)highlightId(target.id);
 const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();
 function down(e:PointerEvent){drag=true;startX=previousX=e.clientX;startY=previousY=e.clientY;targetPosition=null;targetLook=null;}
 function move(e:PointerEvent){if(mode==='street'&&drag){yaw-=(e.clientX-previousX)*.004;pitch=Math.max(-1.2,Math.min(1.2,pitch-(e.clientY-previousY)*.004));camera.rotation.set(pitch,yaw,0);}previousX=e.clientX;previousY=e.clientY;}
 function up(e:PointerEvent){const wasDragging=drag;drag=false;if(!wasDragging||Math.hypot(e.clientX-startX,e.clientY-startY)>5)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hit=ray.intersectObjects(objects,false).find(h=>h.object.visible&&getId(h.object));if(hit)highlightId(getId(hit.object));}
 function keydown(e:KeyboardEvent){if((e.target as HTMLElement)?.closest('input,button,[role="dialog"]'))return;if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight'].includes(e.code)){keys.add(e.code);if(mode==='street')e.preventDefault();}}
 function keyup(e:KeyboardEvent){keys.delete(e.code)}
 function blur(){keys.clear();drag=false}
 renderer.domElement.addEventListener('pointerdown',down);window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);window.addEventListener('keydown',keydown);window.addEventListener('keyup',keyup);window.addEventListener('blur',blur);
 const resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h)});resize.observe(host);
 let frame=0,previous=performance.now();const forward=new THREE.Vector3(),right=new THREE.Vector3();
 function animate(now:number){frame=requestAnimationFrame(animate);const dt=Math.min((now-previous)/1000,.05);previous=now;
  if(mode==='street'){
   camera.getWorldDirection(forward);forward.y=0;forward.normalize();right.crossVectors(forward,new THREE.Vector3(0,1,0));const speed=dt*(keys.has('ShiftLeft')||keys.has('ShiftRight')?18:5);
   if(keys.has('KeyW')||keys.has('ArrowUp'))camera.position.addScaledVector(forward,speed);if(keys.has('KeyS')||keys.has('ArrowDown'))camera.position.addScaledVector(forward,-speed);if(keys.has('KeyD')||keys.has('ArrowRight'))camera.position.addScaledVector(right,speed);if(keys.has('KeyA')||keys.has('ArrowLeft'))camera.position.addScaledVector(right,-speed);
   const len=Math.hypot(camera.position.x,camera.position.z);if(len>255){camera.position.x*=255/len;camera.position.z*=255/len;}camera.position.y=groundAt(camera.position.x,camera.position.z)+1.8;
  }else{if(targetPosition&&targetLook){camera.position.lerp(targetPosition,.07);controls.target.lerp(targetLook,.07);if(camera.position.distanceTo(targetPosition)<.05){targetPosition=null;targetLook=null;}}controls.update();}
  renderer.render(scene,camera);
 }
 frame=requestAnimationFrame(animate);
 return {view,select,layer:(layer:string,on:boolean)=>{objects.filter(o=>getLayer(o)===layer).forEach(o=>o.visible=on)},evidence:(on:boolean)=>{for(const o of objects){if(on){let m=evidences.get(o);if(!m){const l=getLayer(o);const color=['buildings','roofs','terrain'].includes(l)?'#56b3c4':['vegetation','rails','paths','street_objects'].includes(l)?'#7caf8c':'#d2a25e';m=new THREE.MeshStandardMaterial({color,roughness:.9});evidences.set(o,m);}o.material=m;}else o.material=originals.get(o)!;}},dispose:()=>{cancelAnimationFrame(frame);resize.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointerdown',down);window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);window.removeEventListener('keydown',keydown);window.removeEventListener('keyup',keyup);window.removeEventListener('blur',blur);objects.forEach(o=>o.geometry.dispose());new Set([...originals.values()].flat()).forEach(m=>m.dispose());evidences.forEach(m=>m.dispose());renderer.dispose();renderer.domElement.remove();}};
}
