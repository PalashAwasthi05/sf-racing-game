import * as THREE from 'three';
import {opticalCover} from './car-optics.ts';
import {parseSceneStream} from './scene-stream.ts';

// A drivable mid-engined roadster with a planar vehicle model: rigid body (position, yaw, longitudinal and
// lateral velocity, yaw rate), bicycle tyre model with slip angles and a magic-formula lateral force, friction
// circle for combined slip, longitudinal load transfer, engine torque curve through a six-speed gearbox to the
// rear axle, aerodynamic drag and rolling resistance, stepped at a fixed rate. Buildings are a distance field
// rasterised from the scene's foundation outlines, street objects are circles (car-ground.ts), and the body's
// rounded outline meets both through contact impulses, so the car slides along walls and glances off posts. The
// wheels stand on the drivable surface (car-ground.ts): a kerb is a step they climb. Nothing here is location specific.

import type {CarGround} from './car-ground';
import {CarAudio} from './car-audio.ts';
import {CarEffects,type CarHit,type WheelTrack} from './car-effects.ts';

export type CarInput={throttle:number;brake:number;steer:number;handbrake:boolean};
export type CarTelemetry={speedKmh:number;gear:string;rpm:number;throttle:number;slip:number};

const G=9.81;
const SPEC={
 massKg:1485,           // 1,405 kg unladen plus a driver
 wheelbaseM:2.475,
 cgToFrontM:1.36,       // 45/55 mid-engine weight split
 cgHeightM:0.45,
 yawInertia:2600,
 trackFrontM:1.527,
 trackRearM:1.535,
 wheelRadiusFrontM:0.336, // 235/35 R20
 wheelRadiusM:0.347,    // 265/35 R20 (driven)
 tyreMu:1.28,           // Horizon-style grip: sticky, with a broad, forgiving peak
 lockedTyreMu:0.55,
 dragArea:0.64,         // Cd 0.32 x 2.0 m2
 rollingResistance:0.013,
 maxBrakeG:1.3,
 brakeFrontShare:0.62,
 gears:[3.31,1.95,1.41,1.13,0.95,0.81],
 finalDrive:3.89,
 drivelineEfficiency:0.9,
 idleRpm:900,
 limiterRpm:7800,
 upshiftRpm:7100,
 downshiftRpm:3000,
 maxSteerRad:0.55,
};
function engineTorqueNm(rpm:number){
 // Naturally aspirated 4.0 flat six: broad plateau, hard limiter.
 if(rpm>=SPEC.limiterRpm)return 0;
 const points:[number,number][]=[[900,250],[2000,360],[3500,400],[5000,420],[6500,420],[7200,405],[7800,380]];
 for(let i=1;i<points.length;i++){const [r0,t0]=points[i-1],[r1,t1]=points[i];if(rpm<=r1){const k=(rpm-r0)/(r1-r0);return t0+(t1-t0)*Math.max(0,Math.min(1,k));}}
 return 380;
}

// ---- collision field -------------------------------------------------------------------------------------
// Distance (m, up to FIELD_CAP) to the nearest cell the car cannot enter. Cells are marked at load; the distances are an
// exact squared-distance transform (Felzenszwalb-Huttenlocher, rows then columns) made tile by tile where the car or the
// chase camera first asks, over the tile and a margin as wide as the cap, so a tile costs well under a millisecond and
// nothing is computed for streets never driven.
const FIELD_TILE=64,FIELD_CAP=4;
class DistanceField{
 readonly cell:number;readonly n:number;readonly origin:number;private readonly bits:Uint8Array;
 private readonly per:number;private readonly pad:number;private readonly tiles=new Map<number,Float32Array>();
 constructor(extentM:number,cellM:number){this.cell=cellM;this.n=Math.ceil(extentM*2/cellM)+1;this.origin=-extentM;this.bits=new Uint8Array(Math.ceil(this.n*this.n/8));this.per=Math.ceil(this.n/FIELD_TILE);this.pad=Math.ceil(FIELD_CAP/cellM);}
 occupiedCell(index:number){return !!(this.bits[index>>>3] & (1<<(index&7)));}
 private markCell(index:number){this.bits[index>>>3]|=1<<(index&7);}
 private index(x:number,z:number){const i=Math.floor((x-this.origin)/this.cell),j=Math.floor((z-this.origin)/this.cell);return (i<0||j<0||i>=this.n||j>=this.n)?-1:j*this.n+i;}
 markTriangle(ax:number,az:number,bx:number,bz:number,cx:number,cz:number){
  const i0=Math.max(0,Math.floor((Math.min(ax,bx,cx)-this.origin)/this.cell)),i1=Math.min(this.n-1,Math.ceil((Math.max(ax,bx,cx)-this.origin)/this.cell));
  const j0=Math.max(0,Math.floor((Math.min(az,bz,cz)-this.origin)/this.cell)),j1=Math.min(this.n-1,Math.ceil((Math.max(az,bz,cz)-this.origin)/this.cell));
  const det=(bx-ax)*(cz-az)-(cx-ax)*(bz-az);if(Math.abs(det)<1e-9)return;
  for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){const px=this.origin+(i+.5)*this.cell,pz=this.origin+(j+.5)*this.cell;
   const w0=((bx-px)*(cz-pz)-(cx-px)*(bz-pz))/det,w1=((cx-px)*(az-pz)-(ax-px)*(cz-pz))/det,w2=1-w0-w1;
   if(w0>=-1e-6&&w1>=-1e-6&&w2>=-1e-6)this.markCell(j*this.n+i);}
 }
 markPoint(x:number,z:number){const idx=this.index(x,z);if(idx>=0)this.markCell(idx);}
 markSegment(x0:number,z0:number,x1:number,z1:number){const len=Math.hypot(x1-x0,z1-z0);const steps=Math.max(1,Math.ceil(len/(this.cell*.7)));for(let s=0;s<=steps;s++){const k=s/steps;const idx=this.index(x0+(x1-x0)*k,z0+(z1-z0)*k);if(idx>=0)this.markCell(idx);}}
 finish(){this.tiles.clear();}
 get tilesMade(){return this.tiles.size;}
 private tile(ti:number,tj:number){
  const key=tj*this.per+ti;let t=this.tiles.get(key);if(t)return t;
  const T=FIELD_TILE,P=this.pad,n=this.n,i0=Math.max(0,ti*T-P),j0=Math.max(0,tj*T-P),i1=Math.min(n,ti*T+T+P),j1=Math.min(n,tj*T+T+P),W=i1-i0,H=j1-j0;
  const d=new Float64Array(W*H);for(let j=0;j<H;j++)for(let i=0;i<W;i++)d[j*W+i]=this.occupiedCell((j0+j)*n+i0+i)?0:1e12;
  const m=Math.max(W,H),f=new Float64Array(m),out=new Float64Array(m),v=new Int32Array(m),zb=new Float64Array(m+1);
  const pass=(len:number)=>{
   let q=0;v[0]=0;zb[0]=-Infinity;zb[1]=Infinity;
   for(let k=1;k<len;k++){let s=((f[k]+k*k)-(f[v[q]]+v[q]*v[q]))/(2*k-2*v[q]);while(s<=zb[q]){q--;s=((f[k]+k*k)-(f[v[q]]+v[q]*v[q]))/(2*k-2*v[q]);}q++;v[q]=k;zb[q]=s;zb[q+1]=Infinity;}
   q=0;for(let k=0;k<len;k++){while(zb[q+1]<k)q++;out[k]=(k-v[q])*(k-v[q])+f[v[q]];}
  };
  for(let j=0;j<H;j++){for(let i=0;i<W;i++)f[i]=d[j*W+i];pass(W);for(let i=0;i<W;i++)d[j*W+i]=out[i];}
  for(let i=0;i<W;i++){for(let j=0;j<H;j++)f[j]=d[j*W+i];pass(H);for(let j=0;j<H;j++)d[j*W+i]=out[j];}
  t=new Float32Array(T*T).fill(FIELD_CAP);
  for(let j=0;j<T;j++)for(let i=0;i<T;i++){const gi=ti*T+i,gj=tj*T+j;if(gi<n&&gj<n)t[j*T+i]=Math.min(FIELD_CAP,Math.sqrt(d[(gj-j0)*W+gi-i0])*this.cell);}
  // Cap the floating-point distance cache at 2 MiB as the car explores.
  // The packed occupancy mask retains the information to reproduce any tile.
  if(this.tiles.size>=128)this.tiles.delete(this.tiles.keys().next().value!);
  this.tiles.set(key,t);return t;
 }
 private value(i:number,j:number){if(i<0||j<0||i>=this.n||j>=this.n)return FIELD_CAP;return this.tile(Math.floor(i/FIELD_TILE),Math.floor(j/FIELD_TILE))[(j%FIELD_TILE)*FIELD_TILE+i%FIELD_TILE];}
 /** Distance at (x, z), bilinear between cell centres. */
 at(x:number,z:number){
  const px=(x-this.origin)/this.cell-.5,pz=(z-this.origin)/this.cell-.5,i=Math.floor(px),j=Math.floor(pz),fx=px-i,fz=pz-j;
  return (this.value(i,j)*(1-fx)+this.value(i+1,j)*fx)*(1-fz)+(this.value(i,j+1)*(1-fx)+this.value(i+1,j+1)*fx)*fz;
 }
 gradient(x:number,z:number){const h=this.cell;const gx=(this.at(x+h,z)-this.at(x-h,z))/(2*h),gz=(this.at(x,z+h)-this.at(x,z-h))/(2*h);const l=Math.hypot(gx,gz)||1;return [gx/l,gz/l];}
}
// `extra` marks further cells the car cannot enter as points and lines (steps, high faces, water; see car-ground.ts).
export function buildDistanceField(objects:THREE.Object3D[],layerOf:(o:THREE.Object3D)=>string,extentM=275,cellM=.5,extra?:(mark:(x:number,z:number)=>void,line:(x0:number,z0:number,x1:number,z1:number)=>void)=>void){
 const field=new DistanceField(extentM,cellM);const p=new THREE.Vector3(),q=new THREE.Vector3(),r=new THREE.Vector3();
 for(const o of objects){
  if(!(o instanceof THREE.Mesh)||layerOf(o)!=='foundations')continue;
  const pos=o.geometry.getAttribute('position');const idx=o.geometry.getIndex();const count=idx?idx.count:pos.count;
  const at=(k:number,v:THREE.Vector3)=>v.fromBufferAttribute(pos,idx?idx.getX(k):k).applyMatrix4(o.matrixWorld);
  for(let k=0;k<count;k+=3){at(k,p);at(k+1,q);at(k+2,r);field.markSegment(p.x,p.z,q.x,q.z);field.markSegment(q.x,q.z,r.x,r.z);field.markSegment(r.x,r.z,p.x,p.z);if(Math.abs(p.y-q.y)<.05&&Math.abs(p.y-r.y)<.05)field.markTriangle(p.x,p.z,q.x,q.z,r.x,r.z);}
 }
 extra?.((x,z)=>field.markPoint(x,z),(x0,z0,x1,z1)=>field.markSegment(x0,z0,x1,z1));field.finish();return field;
}

// ---- model ------------------------------------------------------------------------------------------------
// The body, glass, lamps, trim, cabin and wheels are generated offline (work/car) into one binary glTF whose mesh names
// carry the part and its finish (`part__finish`). Each finish gets its material here; the wheel assemblies are modelled
// once per axle for the right side (axle along +z) and placed at the four corners in steering hubs with spinning groups,
// mirrored for the left. Forward is +x, up is +y, +z is the car's right.
const AXLE=SPEC.wheelbaseM/2;
export async function loadRoadster(url='/vehicle/roadster.glb'):Promise<THREE.Object3D|null>{
 try{
  const response=await fetch(url,{cache:'no-cache'});
  if(!response.ok||!response.body)return null;
  return (await parseSceneStream(response.body)).scene;
 }catch{return null;}
}
function windDeflectorMaterial(){
 // Coverage averages through mipmaps as the woven strands get smaller than a pixel.
 const size=16,data=new Uint8Array(size*size*4);
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){
  const i=(y*size+x)*4;
  data[i]=data[i+1]=data[i+2]=(x<3||y<3)?255:0;data[i+3]=255;
 }
 const weave=new THREE.DataTexture(data,size,size);
 weave.wrapS=weave.wrapT=THREE.RepeatWrapping;weave.generateMipmaps=true;
 weave.minFilter=THREE.LinearMipmapLinearFilter;weave.magFilter=THREE.LinearFilter;weave.needsUpdate=true;
 const material=new THREE.MeshStandardMaterial({color:'#202123',alphaMap:weave,transparent:true,
  side:THREE.DoubleSide,depthWrite:false,roughness:.91});
 material.forceSinglePass=true;material.addEventListener('dispose',()=>weave.dispose());return material;
}
function finishes(){
 const std=(o:THREE.MeshStandardMaterialParameters)=>new THREE.MeshStandardMaterial(o);
 const phys=(o:THREE.MeshPhysicalMaterialParameters)=>new THREE.MeshPhysicalMaterial(o);
 const f={
  wind_deflector:windDeflectorMaterial(),
  // Solid paint has a diffuse base and a dielectric clear coat.
  paint:phys({color:'#cdd0cd',metalness:.035,roughness:.38,specularIntensity:.45,clearcoat:1,clearcoatRoughness:.04}),
  hoop:phys({color:'#363839',metalness:.18,roughness:.39,clearcoat:.24,clearcoatRoughness:.18}),
  black_gloss:phys({color:'#070808',metalness:0,roughness:.42,clearcoat:.12,clearcoatRoughness:.2}),
  black_satin:std({color:'#121315',metalness:0,roughness:.55}),
  black_matte:std({color:'#0d0d0e',metalness:0,roughness:.85}),
  gap:std({color:'#2a2b2c',metalness:0,roughness:.85}),
  glass:opticalCover('#879d9b',.075),
  lens:opticalCover('#e5e9eb',.025),
  lamp_chrome:std({color:'#9aa0a6',metalness:1,roughness:.18}),
  lamp_black:std({color:'#16181a',metalness:.4,roughness:.35}),
  drl:std({color:'#f4f7ff',emissive:'#eaf1ff',emissiveIntensity:2.2,roughness:.3}),
  tail_lens:opticalCover('#48323b',.16,.075),
  tail_led:std({color:'#5a0508',emissive:'#f62018',emissiveIntensity:.50,roughness:.4}),
  reflector:phys({color:'#47090c',metalness:0,roughness:.26,clearcoat:.68,clearcoatRoughness:.05}),
  tyre:std({color:'#262729',metalness:0,roughness:.9}),
  rim:std({color:'#202225',metalness:.35,roughness:.44,side:THREE.DoubleSide}),   // the barrel is an open surface
  disc:std({color:'#a3a6aa',metalness:.55,roughness:.42}),
  hat:std({color:'#3a3c3f',metalness:.7,roughness:.45}),
  caliper:phys({color:'#b0141c',metalness:.1,roughness:.35,clearcoat:.6,clearcoatRoughness:.1}),
  cabin:std({color:'#1c1c1f',metalness:0,roughness:.55}),          // leather: a little sheen shows its shape
  cabin_soft:std({color:'#2b2b2f',metalness:0,roughness:.96}),     // seat centres: suede-like cloth
  cabin_trim:std({color:'#232529',metalness:.15,roughness:.42}),
  steel:std({color:'#8d9196',metalness:1,roughness:.3}),
  mirror:std({color:'#dfe3e6',metalness:1,roughness:.04}),
 };
 for(const [name,m] of Object.entries(f))m.name=name;
 return f;
}
type Finish=keyof ReturnType<typeof finishes>;
function rearPlate(body:THREE.Mesh){
 const canvas=document.createElement('canvas');canvas.width=384;canvas.height=192;
 const ctx=canvas.getContext('2d')!;
 ctx.fillStyle='#282c30';ctx.fillRect(0,0,384,192);
 ctx.fillStyle='#d9d9ce';ctx.beginPath();ctx.roundRect(7,7,370,178,12);ctx.fill();
 ctx.textAlign='center';ctx.fillStyle='#964743';ctx.font='italic 22px Georgia';ctx.fillText('World Pilot',192,42);
 ctx.font='bold 77px monospace';ctx.fillStyle='#203553';ctx.fillText('PILOT',192,131);
 for(const x of [40,344])for(const y of [22,169]){
  ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle='#565c60';ctx.fill();
 }
 const map=new THREE.CanvasTexture(canvas);map.colorSpace=THREE.SRGBColorSpace;map.anisotropy=4;
 const finish=new THREE.MeshStandardMaterial({map,roughness:.67,metalness:.02});
 const plate=new THREE.Mesh(new THREE.PlaneGeometry(.315,.158),finish);
 const mount=body.userData.rearPlateMount;
 if(mount && Array.isArray(mount.position) && Array.isArray(mount.normal)){
  plate.position.fromArray(mount.position);
  plate.lookAt(new THREE.Vector3().fromArray(mount.normal).add(plate.position));
 }else{
  body.geometry.computeBoundingBox();plate.position.set(body.geometry.boundingBox!.min.x-.009,.475,0);
  plate.rotation.y=-Math.PI/2;
 }
 plate.receiveShadow=true;plate.name='rear_registration';
 finish.addEventListener('dispose',()=>map.dispose());
 return plate;
}
function buildModel(source:THREE.Object3D){
 const car=new THREE.Group();const materials=finishes();
 const stopLamp=materials.tail_led.clone();stopLamp.emissiveIntensity=0;
 const brakeLamps=new Map<THREE.MeshStandardMaterial,number>();   // an imported car's tail lamps (material `light_tail`) and their resting glow
 const wheelParts={front:[] as THREE.Mesh[],rear:[] as THREE.Mesh[]},hubParts={front:[] as THREE.Mesh[],rear:[] as THREE.Mesh[]};
 let body:THREE.Mesh|null=null;
 const meshes:THREE.Mesh[]=[];source.traverse(o=>{if(o instanceof THREE.Mesh)meshes.push(o);});
 for(const o of meshes){
  const [part,raw='']=o.name.split('__');const finish=raw.replace(/_\d+$/,'');
  // `src` keeps the material the model file carries (a textured part of an imported car).
  const own=finish==='src'&&!Array.isArray(o.material)?o.material as THREE.Material:null;
  if(own instanceof THREE.MeshStandardMaterial&&own.name==='light_tail')brakeLamps.set(own,own.emissiveIntensity);
  const material=own??(part==='spoiler'&&finish==='tail_led'?stopLamp:materials[(finish in materials?finish:'black_satin') as Finish]);
  o.material=part==='body'&&finish==='drl'?materials.lamp_black:material;
  if(part==='wind-deflector'){
   const position=o.geometry.attributes.position,uv=new Float32Array(2*position.count);
   for(let i=0;i<position.count;i++){uv[2*i]=position.getZ(i)/.006;uv[2*i+1]=position.getY(i)/.006;}
   o.geometry.setAttribute('uv',new THREE.BufferAttribute(uv,2));o.material=materials.wind_deflector;
  }
  o.castShadow=!material.transparent;o.receiveShadow=true;
  if(!o.geometry.attributes.normal)o.geometry.computeVertexNormals();
  const axle=part.endsWith('_front')?'front':'rear';
  if(part.startsWith('wheel_'))wheelParts[axle].push(o);
  else if(part.startsWith('hub_'))hubParts[axle].push(o);
  else{o.removeFromParent();car.add(o);if(part==='body'&&finish==='paint')body=o;}
 }
 if(body)car.add(rearPlate(body));
 const wheels:{hub:THREE.Group;spin:THREE.Group}[]=[];
 const clone=(m:THREE.Mesh,side:number)=>{const c=new THREE.Mesh(m.geometry,m.material);c.scale.z=side;c.castShadow=m.castShadow;c.receiveShadow=true;c.name=m.name;return c;};
 const makeWheel=(axle:'front'|'rear',x:number,z:number,radius:number)=>{
  const hub=new THREE.Group();hub.position.set(x,radius,z);const spin=new THREE.Group();hub.add(spin);const side=Math.sign(z);
  for(const m of wheelParts[axle])spin.add(clone(m,side));for(const m of hubParts[axle])hub.add(clone(m,side));
  car.add(hub);wheels.push({hub,spin});
 };
 makeWheel('front',AXLE,SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM);makeWheel('front',AXLE,-SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM);
 makeWheel('rear',-AXLE,SPEC.trackRearM/2,SPEC.wheelRadiusM);makeWheel('rear',-AXLE,-SPEC.trackRearM/2,SPEC.wheelRadiusM);
 return {car,wheels,body,tailOn:materials.tail_led,stopLamp,brakeLamps,materials};
}
export function buildRoadster(source:THREE.Object3D){return buildModel(source);}

// ---- sound ---------------------------------------------------------------------------------------------------
// The engine, tyres, wind and collision sounds live in car-audio.ts.

// ---- instrument cluster --------------------------------------------------------------------------------------
// A rev counter drawn as a 270-degree arc with the speed in the middle and the gear below; the needle and the lit arc
// follow the engine every frame.
const TACH={cx:100,cy:96,r:80,start:135,sweep:270,max:8000,red:7000};
function polar(a:number,r:number):[number,number]{const t=a*Math.PI/180;return [TACH.cx+Math.cos(t)*r,TACH.cy+Math.sin(t)*r];}
function arc(a0:number,a1:number,r:number){const [x0,y0]=polar(a0,r),[x1,y1]=polar(a1,r);return `M${x0.toFixed(2)} ${y0.toFixed(2)}A${r} ${r} 0 ${a1-a0>180?1:0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;}
function gaugeMarkup(){
 const ticks:string[]=[];
 for(let k=0;k<=16;k++){const a=TACH.start+TACH.sweep*k/16,major=k%2===0,red=k>=14?' class="is-red"':'';const [x0,y0]=polar(a,TACH.r-6),[x1,y1]=polar(a,TACH.r-(major?15:10));
  ticks.push(`<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}"${red}/>`);
  if(major){const [lx,ly]=polar(a,TACH.r-25);ticks.push(`<text x="${lx.toFixed(1)}" y="${(ly+3.4).toFixed(1)}"${red}>${k/2}</text>`);}}
 const redStart=TACH.start+TACH.sweep*TACH.red/TACH.max,full=arc(TACH.start,TACH.start+TACH.sweep,TACH.r);
 return `<svg class="car-tach" viewBox="0 0 200 164" aria-hidden="true"><path class="tach-track" d="${full}"/><path class="tach-red" d="${arc(redStart,TACH.start+TACH.sweep,TACH.r)}"/><path class="tach-fill" d="${full}" pathLength="1000" stroke-dasharray="0 1000"/><g class="tach-ticks">${ticks.join('')}</g><text class="tach-rpm" x="${TACH.cx}" y="${TACH.cy-34}">RPM x1000</text><text class="tach-speed" x="${TACH.cx}" y="${TACH.cy+9}">0</text><text class="tach-unit" x="${TACH.cx}" y="${TACH.cy+26}">km/h</text><line class="tach-needle" x1="${TACH.cx+16}" y1="${TACH.cy}" x2="${TACH.cx+TACH.r-18}" y2="${TACH.cy}" transform="rotate(${TACH.start} ${TACH.cx} ${TACH.cy})"/><text class="tach-gear" x="${TACH.cx}" y="${TACH.cy+58}">N</text></svg>`;
}

// ---- contact and ride -----------------------------------------------------------------------------------------
const STEP=1/240;                          // physics step (s): fixed, so the handling is the same at any display rate
const BODY_L=2.2,BODY_W=.9,BODY_R=.3;     // plan outline: 4.4 x 1.8 m about the mid-wheelbase origin, corners rounded
const MARGIN=.15;                          // distance-field reading at which the outline touches a building (0.25 m cells)
const RESTITUTION=.12,FRICTION=.3;         // a hit keeps a little of its approach speed; a scraping contact scrubs speed
const KERB=.3;                             // share of its approach speed a wheel loses climbing a step as tall as its radius
const TRAVEL=.09;                          // suspension travel either way from the ride height (m)
const HEAVE_RATE=2*Math.PI*1.9, TILT_RATE=2*Math.PI*1.65; // critically damped chassis response (rad/s)
// The springs only push. At ride height they carry the car's weight; extended past SAG they carry nothing, so where the
// road falls away faster than gravity can pull the car down (a crest taken fast, the far side of a kerb) the car flies.
// Past TRAVEL in compression the bump stops take over, much stiffer, and the body can never sink further than MAX_COMP.
const SAG=G/HEAVE_RATE**2, BUMP_RATE=2*Math.PI*6, MAX_COMP=.2;
// Wheels in the order buildModel makes them (front right, front left, rear right, rear left): forward and left offsets
// from the origin, and the radius.
const WHEELS:[number,number,number][]=[[AXLE,-SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM],[AXLE,SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM],[-AXLE,-SPEC.trackRearM/2,SPEC.wheelRadiusM],[-AXLE,SPEC.trackRearM/2,SPEC.wheelRadiusM]];
// Probe points along the outline, 0.2 m apart, so a building corner cannot slip in between two of them.
const OUTLINE:[number,number][]=(()=>{const pts:[number,number][]=[],sx=BODY_L-BODY_R,sy=BODY_W-BODY_R;
 for(let k=0,n=Math.ceil(2*sx/.2);k<=n;k++){const x=-sx+2*sx*k/n;pts.push([x,BODY_W],[x,-BODY_W]);}
 for(let k=0,n=Math.ceil(2*sy/.2);k<=n;k++){const y=-sy+2*sy*k/n;pts.push([BODY_L,y],[-BODY_L,y]);}
 for(const [cx,cy] of [[sx,sy],[sx,-sy],[-sx,sy],[-sx,-sy]])for(const t of [.2,.5,.8]){const q=t*Math.PI/2;pts.push([cx+Math.sign(cx)*BODY_R*Math.cos(q),cy+Math.sign(cy)*BODY_R*Math.sin(q)]);}
 return pts;})();
type Contact={nx:number;nz:number;px:number;pz:number;pen:number;bias:number;jn:number;jt:number;kind:number};   // kind: 0 field, 1 edge, 2 object

// ---- vehicle ----------------------------------------------------------------------------------------------
export function createCar(scene:THREE.Scene,host:HTMLElement,camera:THREE.PerspectiveCamera,groundAt:(x:number,z:number)=>number,field:DistanceField|null,source:THREE.Object3D,ground:CarGround|null, worldRadiusM:number){
 const {car,wheels,body,tailOn,stopLamp,brakeLamps}=buildModel(source);scene.add(car);
 const sound=new CarAudio();
 // Driver aids: traction control and stability control together, as a car's "ESC off" switch does; ABS always stays on.
 // Off, the throttle can spin the rears and hold a slide. Remembered across visits.
 const AIDS_KEY='king-street-car-aids';
 let aids=true;try{aids=globalThis.localStorage?.getItem(AIDS_KEY)!=='off';}catch{}
 const hud=document.createElement('div');hud.className='car-hud';hud.setAttribute('aria-live','off');
 hud.innerHTML=gaugeMarkup()+'<div class="car-help">W accelerate · Hold S to brake, then reverse · A / D steer · Space handbrake · C camera · R reset · M mute · T traction control · pointer to the top edge for views</div><div class="car-aids" style="padding:1px 8px;border-radius:4px;font:600 11px/16px system-ui,sans-serif;letter-spacing:.06em;color:#ffcf6b;background:rgba(0,0,0,.55)">TC / ESC OFF</div>';
 host.appendChild(hud);const aidsEl=hud.querySelector('.car-aids') as HTMLElement|null;function aidsBadge(){if(aidsEl)aidsEl.hidden=aids;}aidsBadge();
 // The key help shows until the car has been driven for a few seconds, and again after a reset.
 const helpEl=hud.querySelector('.car-help') as HTMLElement|null;let driven=0;const speedEl=hud.querySelector('.tach-speed')!,gearEl=hud.querySelector('.tach-gear')!,needle=hud.querySelector('.tach-needle') as SVGLineElement,fill=hud.querySelector('.tach-fill') as SVGPathElement;
 // A change of one step moves the outer tip by less than half a CSS pixel.
 // Identical telemetry must not invalidate the SVG or the surrounding page.
 let previousStep=-1,previousHot=false;
 function gauge(){
  const step=Math.round(1000*THREE.MathUtils.clamp(s.rpm/TACH.max,0,1));
  if(step!==previousStep){
   previousStep=step;
   needle.setAttribute('transform',`rotate(${TACH.start+TACH.sweep*step/1000} ${TACH.cx} ${TACH.cy})`);
   fill.setAttribute('stroke-dasharray',`${step} 1000`);
  }
  const hot=s.rpm>=TACH.red;
  if(hot!==previousHot){fill.classList.toggle('is-hot',hot);previousHot=hot;}
 }
 // State in the body frame: u forward, v left, r yaw rate (positive turns left). Heading: forward=(-sin yaw, -cos yaw).
 const s={x:0,z:0,yaw:0,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,throttle:0,rpm:SPEC.idleRpm,slip:0,pitch:0,roll:0,skidF:0,skidR:0,squeal:0,ay:0};
 let trace={u:0,v:0,r:0,delta:0,alphaF:0,alphaR:0,fyFront:0,fyRear:0,fxRear:0,fxFront:0,ax:0,ay:0,rDot:0,fzFront:0,fzRear:0,gear:1,rpm:0,shift:0,sax:0,contacts:0,rideZ:0};
 let cameraMode:'chase'|'bonnet'='chase';let lookYaw=0;let hudTimer=0;let reverseHold=0;let shiftTimer=0;let slideHold=0;
 const camPos=new THREE.Vector3(),camLook=new THREE.Vector3();
 const {massKg:m,wheelbaseM:L,cgToFrontM:a,cgHeightM:h,yawInertia:Iz,wheelRadiusM:Rw}=SPEC;const b=L-a;
 const surfaceAt=ground?(x:number,z:number)=>ground.surface.at(x,z):groundAt;
 const effects=new CarEffects(scene,surfaceAt);
 // What happened during this frame's physics steps, for the sparks, marks and sounds: contacts (at most one per step),
 // the hardest new hit, the fastest scrape along a wall, the hardest bump-stop strike and kerb strike.
 const hits:CarHit[]=[];const frame={impact:0,impactKind:-1,scrape:0,bottom:0,kerb:0};
 const wheelTouch=[true,true,true,true];const tracks:WheelTrack[]=WHEELS.map(([,,R])=>({x:0,z:0,touching:true,skid:0,halfWidth:R<.34?.118:.133}));
 // The body rides on its four wheels: the surface under each (front right, front left, rear right, rear left) and its
 // height above the terrain grid (a kerb is a jump in it, a hill is not); the body's height, pitch and roll are springs
 // toward the plane of the four contacts.
 const wheelH=new Float64Array(4),detail=new Float64Array(4);
 // load: the tyres' share of the car's weight right now (0 in the air); grip: the same, eased, as the tyres feel it;
 // air: seconds since the springs last carried any load.
 const ride={z:0,zv:0,pv:0,qv:0,roadPitch:0,roadRoll:0,load:1,grip:1,air:0};
 // The frame shows the physics state interpolated between the last two steps.
 const prev={x:0,z:0,yaw:0,y:0,pitch:0,roll:0},rs={x:0,z:0,yaw:0,y:0,pitch:0,roll:0};let acc=0;
 const contacts:Contact[]=[];let contactCount=0;const nearby:number[]=[];
 // the hardest hit since the car was placed: approach speed along the contact normal (m/s) and what was hit (kind)
 const impact={speed:0,kind:-1};
 function placeAt(x:number,z:number,yaw:number){Object.assign(s,{x,z,yaw,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,throttle:0,slip:0,rpm:SPEC.idleRpm});settle();acc=0;reverseHold=0;shiftTimer=0;impact.speed=0;impact.kind=-1;driven=0;if(helpEl&&helpEl!==(hud as unknown))helpEl.hidden=false;effects.reset();omega.fill(0);abs.fill(1);tc=1;s.ay=0;Object.assign(prev,{x,z,yaw,y:ride.z,pitch:s.pitch,roll:s.roll});Object.assign(rs,prev);lookYaw=0;hud.hidden=false;syncCamera(0,true);if(navigator.userActivation?.hasBeenActive)sound.prepare();}
 function forward(){return [-Math.sin(s.yaw),-Math.cos(s.yaw)];}
 function left(){const [fx,fz]=forward();return [fz,-fx];}
 // ---- tyres and wheels ------------------------------------------------------------------------------------------
 // Four tyres, each with its own load, its own wheel speed and its own slip. A wheel turns at omega (rad/s) under the
 // engine's torque (rears, through a limited-slip differential), its brake, and the road's push back on the tyre; the
 // tyre's force comes from how much it slips: the slip ratio kappa (tread speed against road speed, along the wheel)
 // and the slip angle alpha (across it). The two are normalised by their peaks and combined, so a tyre spending its grip
 // on braking or wheelspin has less left for cornering (the friction ellipse) and the curve past the peak is what a
 // slide, a lock-up or a burnout is. Loads move between the four corners with the car's acceleration and fall with the
 // suspension's load in the air. ABS and traction control act per wheel on its slip ratio.
 const omega=new Float64Array(4),wheelSkid=new Float64Array(4),wheelSqueal=new Float64Array(4),fz=new Float64Array(4),kappa=new Float64Array(4),alpha=new Float64Array(4);
 const abs=[1,1,1,1];let tc=1;
 const WHEEL_I=[1.1,1.1,1.3,1.3],ENGINE_I=.18,KAPPA_PEAK=.1,TAN_ALPHA_PEAK=Math.tan(.26),FZ_NOMINAL=m*G/4,LOW_SPEED=1.5;
 const curve=(sn:number)=>Math.sin(1.35*Math.atan(2.36*sn));          // 1 at the peak, ~0.85 sliding: broad and forgiving
 const tyre={fx:0,fy:0,sn:0,k:0,t:0};
 // The tyre's force (in the wheel's frame: fx along, fy across, left positive) at wheel speed om.
 function tyreForce(i:number,om:number,vx:number,vy:number,load:number){
  const R=WHEELS[i][2],den=Math.max(Math.abs(vx),LOW_SPEED);
  const k=(om*R-vx)/den,t=vy/den,sx=k/KAPPA_PEAK,sy=t/TAN_ALPHA_PEAK,sn=Math.hypot(sx,sy);
  const mu=SPEC.tyreMu*(1-.08*(load/FZ_NOMINAL-1));                   // load sensitivity: grip grows less than load
  const f=mu*load*curve(sn);
  tyre.fx=sn>1e-9?f*sx/sn:0;tyre.fy=sn>1e-9?-f*sy/sn:0;tyre.sn=sn;tyre.k=k;tyre.t=t;return tyre;
 }
 function step(hdt:number,input:CarInput){
  const {u,v,r}=s;const absU=Math.abs(u);
  // Steering: rate limited, and the wheels turn less at speed, as a driver would.
  // Speed-sensitive lock; a snapping input (keys) winds on at a paced rate and unwinds faster, so the yaw response
  // is smooth however the input arrives.
  const maxSteer=SPEC.maxSteerRad/(1+(absU/11)**2);let target=input.steer*maxSteer;
  // Steering assist: at speed the front wheels are not turned past the angle that puts the front tyres at their grip
  // peak for the way the car is actually moving, so full lock from a key bends the car at the limit instead of ploughing
  // straight on with the fronts scrubbing. It limits only the turn-in side; countersteer in a slide is untouched.
  if(absU>5&&!s.reverse&&!input.handbrake){const flow=Math.atan((v+a*r)/absU),peak=.2;target=Math.max(flow-peak,Math.min(flow+peak,target));}const rate=(Math.abs(target)<Math.abs(s.steer)?8:3.8)/(1+absU/30);
  s.steer+=Math.max(-rate*hdt,Math.min(rate*hdt,target-s.steer));const delta=s.steer;
  // Gearbox: automatic shifts by engine speed, reverse engaged from rest with the brake pedal held.
  if(!s.reverse&&absU<.4&&input.brake>0&&input.throttle===0&&!input.handbrake){
   reverseHold+=hdt;
   if(reverseHold>=.35){s.reverse=true;s.gear=1;shiftTimer=0;}
  }else reverseHold=0;
  if(s.reverse&&absU<.4&&input.throttle>0&&input.brake===0){s.reverse=false;s.gear=1;shiftTimer=0;}
  const drivePedal=s.reverse?input.brake:input.throttle, brakePedal=s.reverse?input.throttle:input.brake;
  const ratio=(s.reverse?SPEC.gears[0]:SPEC.gears[s.gear-1])*SPEC.finalDrive;
  // The engine turns with the driven wheels: a spinning rear revs it, a locked one stalls it to idle.
  const rearOmega=(omega[2]+omega[3])*.5,wheelRpm=Math.abs(rearOmega)*ratio*60/(2*Math.PI);let rpm=wheelRpm;
  shiftTimer=Math.max(0,shiftTimer-hdt);
  const flying=ride.air>.05,grip=ride.grip;
  // The box holds its gear when the driver lifts (engine braking; it used to short-shift up at 2,800 rpm the moment the
  // throttle was released), shifts down as the revs fall, sooner under braking, and with the throttle open it shifts
  // later the further the pedal is down: 3,400 rpm on a light pedal, the full upshift speed on the floor.
  const coasting=drivePedal<.05,slowing=brakePedal>.05;
  // In a slide or with the rears spinning it keeps the gear and lets the engine sit on the limiter: an upshift there
  // drops the wheel speed and ends the drift. The hold outlasts the slide by 0.6 s so it does not shift mid-transition.
  const spin=(Math.abs(rearOmega)*Rw-absU)/Math.max(4,absU);
  slideHold=input.handbrake||Math.abs(v-b*r)>2.2||spin>.18?.6:Math.max(0,slideHold-hdt);
  const upAt=coasting?SPEC.limiterRpm-150:slideHold>0?Infinity:3400+(SPEC.upshiftRpm-3400)*drivePedal;
  const downAt=coasting?(slowing?2600:1500):1400+(SPEC.downshiftRpm-1400)*drivePedal;
  if(!s.reverse&&shiftTimer===0&&!flying){if(rpm>upAt&&s.gear<SPEC.gears.length){s.gear++;shiftTimer=.25;}else if(rpm<downAt&&s.gear>1){s.gear--;shiftTimer=.25;}}
  rpm=Math.max(SPEC.idleRpm,Math.min(SPEC.limiterRpm,rpm));
  // Pulling away the clutch slips: the engine runs at a speed set by the pedal, above what the wheels ask for, and passes
  // its full torque; while it slips the engine's inertia is not on the wheels.
  let clutchSlipping=false;
  if(drivePedal>0){const launch=SPEC.idleRpm+drivePedal*1800*Math.max(0,1-Math.abs(rearOmega)*Rw/8);if(launch>rpm){rpm=launch;clutchSlipping=true;}}
  s.rpm+=(rpm-s.rpm)*Math.min(1,hdt*12);
  const reverseGovernor=s.reverse?THREE.MathUtils.clamp((7-absU)/1.2,0,1):1;
  const rearSlipNow=Math.abs(Math.atan((v-b*r)/Math.max(.6,absU)));const electronics=aids&&!input.handbrake;const stmTrim=(electronics&&rearSlipNow>.28)?.55:1;
  let axleTorque=engineTorqueNm(rpm)*drivePedal*reverseGovernor*(shiftTimer>0?.15:1)*ratio*SPEC.drivelineEfficiency*stmTrim*(electronics?tc:1)*(s.reverse?-1:1);
  // Engine braking: the engine's own friction, through the gearbox, while it is driven by the wheels above idle.
  if(drivePedal===0&&wheelRpm>SPEC.idleRpm*1.15&&!flying)axleTorque-=Math.sign(rearOmega)*(16+wheelRpm*.0055)*ratio*SPEC.drivelineEfficiency;
  const coupled=!clutchSlipping&&shiftTimer===0;
  // Limited-slip differential: torque moves from the faster rear wheel to the slower one, up to a locking limit.
  const lsd=THREE.MathUtils.clamp(45*(omega[3]-omega[2]),-500,500);
  // Loads: static split, longitudinal transfer from the tyres' last acceleration, lateral transfer half to each axle,
  // all scaled by what the springs carry (nothing in the air, more on a landing).
  const axleF=Math.max(400,m*G*b/L-m*s.ax*h/L),axleR=Math.max(400,m*G*a/L+m*s.ax*h/L);
  // lateral transfer split between the axles as their static loads are (a neutral roll-stiffness balance)
  const latF=(b/L)*m*s.ay*h/SPEC.trackFrontM,latR=(a/L)*m*s.ay*h/SPEC.trackRearM;   // s.ay > 0 (left) loads the right wheels
  fz[0]=Math.max(0,axleF/2+latF)*grip;fz[1]=Math.max(0,axleF/2-latF)*grip;fz[2]=Math.max(0,axleR/2+latR)*grip;fz[3]=Math.max(0,axleR/2-latR)*grip;
  // Brakes: up to 1.3 g, 62 % front, as torque at each wheel; ABS scales each wheel back while it locks. The handbrake
  // clamps the rears hard, with no ABS, and the stability electronics hand the car to the driver.
  const brakeForce=brakePedal*SPEC.maxBrakeG*m*G;
  const cosD=Math.cos(delta),sinD=Math.sin(delta);
  let sumX=0,sumY=0,mz=0,fyFrontBody=0,fxFront=0,fxRear=0,fyFront=0,fyRear=0,aF=0,aR=0,maxDriveSlip=0;
  for(let i=0;i<4;i++){
   const front=i<2,wx=front?a:-b,wy=WHEELS[i][1],R=WHEELS[i][2];
   const vxB=u-r*wy,vyB=v+r*wx,c=front?cosD:1,sn=front?sinD:0;
   const vx=c*vxB+sn*vyB,vy=-sn*vxB+c*vyB;
   let brakeT=(front?SPEC.brakeFrontShare:1-SPEC.brakeFrontShare)*.5*brakeForce*R*(!input.handbrake||front?abs[i]:1);
   if(!front&&input.handbrake)brakeT+=3500;
   const driveT=front?0:axleTorque*.5+(i===2?lsd:-lsd);
   const I=WHEEL_I[i]+(!front&&coupled?.5*ENGINE_I*ratio*ratio:0);
   // Wheel speed. The tyre pushes back on the wheel very stiffly near rest (a small change of wheel speed is a large
   // change of slip ratio), so the wheel is integrated in substeps short enough for that stiffness, each also implicit
   // against the tyre's local slope; the brake can stop the wheel but not turn it backwards. The body's velocity is held
   // through the substeps and the tyre's force on it is their average.
   const kLin=SPEC.tyreMu*Math.max(fz[i],1)*3.19/(KAPPA_PEAK*Math.max(Math.abs(vx),LOW_SPEED));
   const n=Math.max(1,Math.min(64,Math.ceil(hdt*R*R*kLin/I/.6))),dh=hdt/n;
   let next=omega[i],avgX=0,avgY=0;
   for(let k=0;k<n;k++){
    const F0=tyreForce(i,next,vx,vy,fz[i]).fx,F1=tyreForce(i,next+.05,vx,vy,fz[i]).fx,stiff=Math.max(0,(F1-F0)/.05)*R;
    const Ieff=I+dh*stiff*R;
    next+=dh*(driveT-R*F0)/Ieff;
    const stop=dh*brakeT/Ieff;next=Math.abs(next)<=stop?0:next-Math.sign(next)*stop;
    const tk=tyreForce(i,next,vx,vy,fz[i]);avgX+=tk.fx/n;avgY+=tk.fy/n;
   }
   omega[i]=next;
   const t=tyreForce(i,next,vx,vy,fz[i]);kappa[i]=t.k;alpha[i]=Math.atan(t.t);t.fx=avgX;t.fy=avgY;
   const fxB=c*t.fx-sn*t.fy,fyB=sn*t.fx+c*t.fy;
   sumX+=fxB;sumY+=fyB;mz+=wx*fyB-wy*fxB;
   if(front){fyFrontBody+=fyB;fxFront+=fxB;fyFront+=t.fy;aF+=alpha[i]*.5;}else{fxRear+=fxB;fyRear+=t.fy;aR+=alpha[i]*.5;}
   // ABS: back the brake off while the wheel locks past the tyre's peak; let it back on as the wheel recovers.
   abs[i]=brakePedal>0&&t.k*Math.sign(vx||1)<-1.15*KAPPA_PEAK?Math.max(.1,abs[i]-hdt*20):Math.min(1,abs[i]+hdt*4);
   // (the slip that traction control watches: a driven tyre's combined slip while the engine is what pushes it past its peak)
   if(!front&&t.k*Math.sign(axleTorque||1)>.03)maxDriveSlip=Math.max(maxDriveSlip,t.sn);
   // Rubber laid down and squeal: past the tyre's peak with the contact patch actually sliding over the road.
   const slide=Math.hypot(next*R-vx,vy);
   wheelSkid[i]=fz[i]>50?THREE.MathUtils.clamp((t.sn-1.3)/.6,0,1)*THREE.MathUtils.clamp((slide-1.5)/2.5,0,1):0;
   wheelSqueal[i]=fz[i]>50?THREE.MathUtils.clamp((t.sn-.8)/.5,0,1)*THREE.MathUtils.clamp((slide-.8)/3,0,1):0;
  }
  // Traction control: cut the engine while a driven tyre's combined slip is past its peak, restore it as it grips.
  tc=maxDriveSlip>1.02&&drivePedal>0?Math.max(.15,tc-hdt*30):Math.min(1,tc+hdt*2.5);
  s.skidF=Math.max(wheelSkid[0],wheelSkid[1]);s.skidR=Math.max(wheelSkid[2],wheelSkid[3]);s.squeal=Math.max(wheelSqueal[0],wheelSqueal[1],wheelSqueal[2],wheelSqueal[3]);
  s.slip=Math.max(Math.abs(aF),Math.abs(aR));
  const drag=-.5*1.2*SPEC.dragArea*u*absU;const rolling=absU>.1?-Math.sign(u)*SPEC.rollingResistance*m*G*Math.min(1,grip):0;
  const axT=(sumX+drag+rolling)/m,ayT=sumY/m;
  const ax=axT+v*r,ay=ayT-u*r;
  // Gravity along the road. The road plane under the four tyres is pitched (front up positive) and rolled (left up
  // positive); on the ground the weight's component in that plane pulls the car downhill, forward/back and sideways. It
  // moves the car but is not a tyre force, so it stays out of the load transfer (s.ax, s.ay) and the suspension's pitch.
  const onGround=ride.air<.02;
  const gx=onGround?-G*Math.sin(ride.roadPitch):0,gy=onGround?-G*Math.sin(ride.roadRoll):0;
  let rDot=mz/Iz;
  // Stability control: yaw beyond what the steering asks for at this speed is damped, as the brakes of a
  // stability system would; the handbrake switches it off.
  if(electronics&&absU>2&&grip>.3){const rGrip=.9*SPEC.tyreMu*G/absU;const rRef=Math.max(-rGrip,Math.min(rGrip,u*Math.tan(delta)/L));const excess=r-rRef;if(Math.abs(excess)>.08){const e=excess-Math.sign(excess)*.08;rDot-=e*9;rDot-=Math.sign(e)*Math.min(Math.abs(e),.5)*a*Math.abs(fyFrontBody)*.35/Iz;}}
  const alphaF=aF,alphaR=aR,fzFront=fz[0]+fz[1],fzRear=fz[2]+fz[3];
  trace={u,v,r,delta,alphaF,alphaR,fyFront,fyRear,fxRear,fxFront,ax,ay,rDot,fzFront,fzRear,gear:s.gear,rpm:s.rpm,shift:shiftTimer,sax:s.ax,contacts:0,rideZ:ride.z};
  s.u+=(ax+gx)*hdt;s.v+=(ay+gy)*hdt;s.r+=rDot*hdt;s.ax+=(axT-s.ax)*.3;s.ay+=(ayT-s.ay)*.3;
  // Parking speeds: the slip-angle tyre has no meaning near rest, so the body is blended toward the kinematic
  // turn the steering asks for (yaw rate u tan(delta)/L, rear axle tracking), fully at walking pace.
  if(absU<3&&grip>.3){const k=Math.min(1,(3-absU)/2.5)*Math.min(1,hdt*14)*Math.min(1,grip);const rKin=s.u*Math.tan(delta)/L;s.r+=(rKin-s.r)*k;s.v+=(b*s.r-s.v)*k;if(drivePedal===0&&Math.abs(s.u)<.15&&brakePedal>0){s.u=0;omega.fill(0);}}
  // Hold at rest: stopped with neither the drive pedal nor the brake moving it, the brakes hold the car on a hill (as an
  // automatic's auto-hold does) instead of letting gravity roll it away. Grades steeper than the brakes can hold slide.
  if(onGround&&drivePedal===0&&Math.abs(s.u)<.15&&Math.abs(s.v)<.3&&Math.hypot(gx,gy)<SPEC.maxBrakeG*G*.9){s.u=0;s.v=0;s.r*=.5;omega.fill(0);}
  s.yaw+=s.r*hdt;
  const [fx,fz2]=forward(),[lx,lz]=left();
  s.x+=(fx*s.u+lx*s.v)*hdt;s.z+=(fz2*s.u+lz*s.v)*hdt;
  s.throttle=drivePedal;
  // The tyres load the suspension through their forces. Body-frame velocity derivatives also include yaw.
  rideStep(hdt,ax-v*r,ay+u*r);collide();
  trace.contacts=contactCount;trace.rideZ=ride.z;
 }
 // World velocity of a point of the body at offset (px, pz) from the origin, and an impulse applied there.
 function pointVelocity(px:number,pz:number):[number,number]{const [fx,fz]=forward(),[lx,lz]=left();return [fx*s.u+lx*s.v+s.r*pz,fz*s.u+lz*s.v-s.r*px];}
 function impulse(jx:number,jz:number,px:number,pz:number){const [fx,fz]=forward(),[lx,lz]=left();const vx=fx*s.u+lx*s.v+jx/m,vz=fz*s.u+lz*s.v+jz/m;s.u=vx*fx+vz*fz;s.v=vx*lx+vz*lz;s.r+=(pz*jx-px*jz)/Iz;}
 function sample(){const [fx,fz]=forward(),[lx,lz]=left();for(let i=0;i<4;i++){const [wx,wy]=WHEELS[i];const x=s.x+fx*wx+lx*wy,z=s.z+fz*wx+lz*wy;wheelH[i]=surfaceAt(x,z);detail[i]=wheelH[i]-groundAt(x,z);}}
 // Fit one support plane to the final drivable surface. A road above uneven underlying terrain
 // still has the same suspension response as that road would have on level ground.
 function supportHeight(){
  const [fr,fl,rr,rl]=wheelH;
  ride.roadPitch=Math.atan2((fr+fl-rr-rl)*.5,L);
  ride.roadRoll=Math.atan2((fl+rl-fr-rr)*.5,(SPEC.trackFrontM+SPEC.trackRearM)*.5);
  return (fr+fl+rr+rl)*.25;
 }
 function settle(){
  sample();ride.z=supportHeight();s.pitch=ride.roadPitch;s.roll=ride.roadRoll;
  ride.zv=ride.pv=ride.qv=0;ride.load=ride.grip=1;ride.air=0;
 }
 // Uphill direction of the surface's height above the terrain grid (a kerb's face), or none where it is smooth.
 function stepDirection(x:number,z:number):[number,number]{const d=.3,e=(p:number,q:number)=>surfaceAt(p,q)-groundAt(p,q);const gx=(e(x+d,z)-e(x-d,z))/(2*d),gz=(e(x,z+d)-e(x,z-d))/(2*d),g=Math.hypot(gx,gz);return g>.06?[gx/g,gz/g]:[0,0];}
 function rideStep(hdt:number,forwardLoad:number,lateralLoad:number){
  const [fx,fz]=forward(),[lx,lz]=left();
  for(let i=0;i<4;i++){
   const [wx,wy,R]=WHEELS[i];const px=fx*wx+lx*wy,pz=fz*wx+lz*wy,x=s.x+px,z=s.z+pz;const hNow=surfaceAt(x,z),dNow=hNow-groundAt(x,z),rise=dNow-detail[i];wheelH[i]=hNow;detail[i]=dNow;
   // A wheel meeting a step has to climb it: the car loses speed at that corner along the step's uphill direction, in
   // proportion to the rise against the wheel's radius, and the impulse at the corner turns the car a little.
   // (only a wheel on the ground meets the step's face; one in the air passes over it)
   if(rise>.002&&ride.load>0){const [nx,nz]=stepDirection(x,z);if(nx||nz){const [vx,vz]=pointVelocity(px,pz);const vn=vx*nx+vz*nz;if(vn>0){const j=KERB*m/4*Math.min(1,rise/R)*vn;impulse(-nx*j,-nz*j,px,pz);frame.kerb=Math.max(frame.kerb,vn*Math.min(1,rise/R));}}}
  }
  const roadHeight=supportHeight();
  // How fast the road under the car rises at the car's speed, and the spring's compression and its rate.
  const climb=s.u*Math.tan(ride.roadPitch)+s.v*Math.tan(ride.roadRoll);
  const comp=roadHeight-ride.z,compRate=climb-ride.zv;
  // Upward acceleration the tyres give the body. On the ground and within travel this is exactly the critically damped
  // spring toward the road the car always had (G cancels); it cannot pull, so a road dropping away leaves the car to
  // gravity, and past full extension the wheels hang free.
  let support=0;
  if(comp>-SAG){
   support=Math.max(0,G+HEAVE_RATE**2*comp+2*HEAVE_RATE*compRate);
   if(comp>TRAVEL){support+=BUMP_RATE**2*(comp-TRAVEL)+Math.max(0,2*BUMP_RATE*compRate);frame.bottom=Math.max(frame.bottom,compRate);}
  }
  ride.zv+=(support-G)*hdt;ride.z+=ride.zv*hdt;
  if(roadHeight-ride.z>MAX_COMP){ride.z=roadHeight-MAX_COMP;if(ride.zv<climb){frame.bottom=Math.max(frame.bottom,climb-ride.zv);ride.zv=climb;}}
  ride.load=support/G;ride.air=support>0?0:ride.air+hdt;
  ride.grip+=(Math.min(1.8,ride.load)-ride.grip)*Math.min(1,hdt*40);
  if(ride.air>.06){
   // Airborne: nothing turns the body but what it took off with, and the air slowly takes that.
   ride.pv*=Math.exp(-hdt*.8);ride.qv*=Math.exp(-hdt*1.5);
  }else{
   // Load transfer displaces the spring equilibrium. A pedal or steering change cannot kick the
   // damper as if the road itself had moved. The tyre solver still takes the full load.
   const pitchTarget=ride.roadPitch+THREE.MathUtils.clamp(.0023*forwardLoad,-.026,.026);
   const rollTarget=ride.roadRoll+THREE.MathUtils.clamp(.0032*lateralLoad,-.037,.037);
   ride.pv+=(TILT_RATE**2*(pitchTarget-s.pitch)-2*TILT_RATE*ride.pv)*hdt;
   ride.qv+=(TILT_RATE**2*(rollTarget-s.roll)-2*TILT_RATE*ride.qv)*hdt;
  }
  s.pitch+=ride.pv*hdt;s.roll+=ride.qv*hdt;
 }
 function collide(){
  const [fx,fz]=forward(),[lx,lz]=left();contactCount=0;
  const add=(nx:number,nz:number,px:number,pz:number,pen:number,kind=0)=>{let c=contacts[contactCount];if(!c){c={nx:0,nz:0,px:0,pz:0,pen:0,bias:0,jn:0,jt:0,kind:0};contacts.push(c);}c.nx=nx;c.nz=nz;c.px=px;c.pz=pz;c.pen=pen;c.bias=0;c.jn=0;c.jt=0;c.kind=kind;contactCount++;};
  // Buildings, steps and high rises (the distance field) and the world's edge, against probes along the outline;
  // each contact pushes the car along the field's gradient, away from what it touches.
  for(const [bx,by] of OUTLINE){
   const px=fx*bx+lx*by,pz=fz*bx+lz*by,x=s.x+px,z=s.z+pz;
   const radial=Math.hypot(x,z);if(radial>worldRadiusM){add(-x/radial,-z/radial,px,pz,radial-worldRadiusM,1);continue;}
   if(!field)continue;const d=field.at(x,z);if(d>=MARGIN)continue;
   let [nx,nz]=field.gradient(x,z);
   if(!(nx||nz)){const [vx,vz]=pointVelocity(px,pz),sp=Math.hypot(vx,vz),pr=Math.hypot(px,pz)||1;[nx,nz]=sp>.01?[-vx/sp,-vz/sp]:[-px/pr,-pz/pr];}
   add(nx,nz,px,pz,MARGIN-d);
  }
  // Street objects: circles against the rounded outline (a core box grown by the corner radius).
  if(ground){const o=ground.obstacles,cx=BODY_L-BODY_R,cy=BODY_W-BODY_R;o.near(s.x,s.z,BODY_L,nearby);
   for(const k of nearby){
    const ox=o.x[k]-s.x,oz=o.z[k]-s.z,bx=ox*fx+oz*fz,by=ox*lx+oz*lz,qx=Math.max(-cx,Math.min(cx,bx)),qy=Math.max(-cy,Math.min(cy,by));
    const dx=bx-qx,dy=by-qy,d=Math.hypot(dx,dy);let ex:number,ey:number,pen:number;
    // the push runs from the object toward the car; an object whose centre is inside the core leaves by the nearer side
    if(d>1e-6){pen=BODY_R+o.r[k]-d;if(pen<=0)continue;ex=-dx/d;ey=-dy/d;}
    else if(cx-Math.abs(bx)<cy-Math.abs(by)){ex=-Math.sign(bx)||-1;ey=0;pen=BODY_R+o.r[k]+cx-Math.abs(bx);}
    else{ex=0;ey=-Math.sign(by)||-1;pen=BODY_R+o.r[k]+cy-Math.abs(by);}
    const tx=bx+ex*o.r[k],ty=by+ey*o.r[k];add(fx*ex+lx*ey,fz*ex+lz*ey,fx*tx+lx*ty,fz*tx+lz*ty,pen,2);
   }}
  if(!contactCount)return;
  // Out of the overlap first: each contact moves the car only by what the moves before it left.
  let mx=0,mz=0;for(let it=0;it<3;it++)for(let i=0;i<contactCount;i++){const c=contacts[i];const rest=c.pen-(mx*c.nx+mz*c.nz);if(rest>0){mx+=c.nx*rest;mz+=c.nz*rest;}}
  s.x+=mx;s.z+=mz;
  // Then the velocities: sequential impulses at the contact points, a little restitution on a hit, Coulomb friction
  // (a scraping contact scrubs speed), so the car slides along a wall, stops against it head on and turns about a post.
  // The contact that strikes or slides hardest becomes this step's hit for the sparks and sounds.
  let best=-1,bestScore=0,bestVx=0,bestVz=0,bestVt=0;
  for(let i=0;i<contactCount;i++){const c=contacts[i];const [vx,vz]=pointVelocity(c.px,c.pz);const vn=vx*c.nx+vz*c.nz;c.bias=vn<-.5?-RESTITUTION*vn:0;if(-vn>impact.speed){impact.speed=-vn;impact.kind=c.kind;}
   if(c.kind!==1){const vt=Math.abs(vz*c.nx-vx*c.nz),score=Math.max(0,-vn)+.3*vt;if(score>bestScore){best=i;bestScore=score;bestVx=vx;bestVz=vz;bestVt=vt;}
    if(-vn>frame.impact){frame.impact=-vn;frame.impactKind=c.kind;}frame.scrape=Math.max(frame.scrape,vt);}}
  if(best>=0&&hits.length<32){const c=contacts[best];hits.push({x:s.x+c.px,y:ride.z+(c.kind===2?.42:.34),z:s.z+c.pz,nx:c.nx,nz:c.nz,vx:bestVx,vz:bestVz,vn:Math.max(0,-(bestVx*c.nx+bestVz*c.nz)),speed:bestVt,kind:c.kind,dt:STEP});}
  for(let it=0;it<4;it++)for(let i=0;i<contactCount;i++){
   const c=contacts[i];let [vx,vz]=pointVelocity(c.px,c.pz);
   const rn=c.pz*c.nx-c.px*c.nz,jn=Math.max(0,c.jn+(c.bias-(vx*c.nx+vz*c.nz))/(1/m+rn*rn/Iz)),dn=jn-c.jn;c.jn=jn;impulse(c.nx*dn,c.nz*dn,c.px,c.pz);
   [vx,vz]=pointVelocity(c.px,c.pz);const tx=-c.nz,tz=c.nx,rt=c.pz*tx-c.px*tz,lim=FRICTION*c.jn,jt=Math.max(-lim,Math.min(lim,c.jt-(vx*tx+vz*tz)/(1/m+rt*rt/Iz))),dt=jt-c.jt;c.jt=jt;impulse(tx*dt,tz*dt,c.px,c.pz);
  }
  // A glancing hit turns the car, but it does not leave it spinning: some yaw goes while it is in contact.
  s.r*=.985;
 }
 function lerpAngle(a:number,b:number,k:number){let d=b-a;while(d>Math.PI)d-=2*Math.PI;while(d<-Math.PI)d+=2*Math.PI;return a+d*k;}
 let camYaw=0,camDist=5.25,camGrade=0,reverseView=false;
 // The clearance grid bounds the boom only when it actually meets a wall. On open road the
 // camera can use any distance, including the fraction between the last two sample points.
 function freeDistance(bx:number,bz:number,want:number){
  if(!field)return want;
  const clearance=(d:number)=>field.at(rs.x-bx*d,rs.z-bz*d);
  let clear=2.4;
  while(clear<want){
   let end=Math.min(want,clear+.25);
   if(clearance(end)<.7){
    for(let i=0;i<5;i++){const mid=(clear+end)*.5;if(clearance(mid)<.7)end=mid;else clear=mid;}
    return clear;
   }
   clear=end;
  }
  return want;
 }
 function syncCamera(dt:number,snap=false){
  const speed=Math.abs(s.u),y=rs.y,fx=-Math.sin(rs.yaw),fz=-Math.cos(rs.yaw);
  if(cameraMode==='bonnet'){const ly=rs.yaw+lookYaw;camera.position.set(rs.x+fx*.25,y+1.1+rs.pitch*.3,rs.z+fz*.25);camera.up.set(0,1,0);camera.lookAt(rs.x-Math.sin(ly)*30,y+1.02,rs.z-Math.cos(ly)*30);camera.fov=64+Math.min(10,speed*.15);camera.updateProjectionMatrix();return;}
  if(s.u<-.8)reverseView=true;else if(s.u>.5||snap)reverseView=false;
  const ahead=reverseView?-1:1;
  const slide=Math.atan2(s.v,Math.max(1.5,speed))*ahead*Math.min(1,speed/6);
  const targetYaw=rs.yaw+(reverseView?Math.PI:0)+slide*.45+lookYaw;
  camYaw=snap?targetYaw:lerpAngle(camYaw,targetYaw,1-Math.exp(-dt*6));
  const bx=-Math.sin(camYaw),bz=-Math.cos(camYaw);
  const wanted=6.9+Math.min(.8,speed*.026);   // a longer lens from further back: at 5.25 m and 53 degrees the tail read twice the size of the cabin
  const extension=snap?wanted:THREE.MathUtils.lerp(camDist,wanted,1-Math.exp(-dt*2.5));
  camDist=freeDistance(bx,bz,extension);
  // Transport the boom and its aim with the *same* interpolated chassis position. Only orbit
  // angle, extension and road grade have inertia. Braking cannot make the camera chase an old
  // world-space aim point. The grade comes from the tyres, independent of body dive and roll.
  const angle=camYaw-rs.yaw;
  const roadGrade=Math.tan(ride.roadPitch)*Math.cos(angle)+Math.tan(ride.roadRoll)*Math.sin(angle);
  camGrade=snap?roadGrade:THREE.MathUtils.lerp(camGrade,roadGrade,1-Math.exp(-dt*3));
  const lead=1.6,aimY=y+1.07;
  camLook.set(rs.x+bx*lead,aimY+camGrade*lead,rs.z+bz*lead);
  camPos.set(rs.x-bx*camDist,aimY+.092*(camDist+lead)-camGrade*camDist,rs.z-bz*camDist);
  camera.position.copy(camPos);camera.up.set(0,1,0);camera.lookAt(camLook);
  camera.fov=41+Math.min(8,speed*.17);camera.updateProjectionMatrix();
 }
 function pose(dt:number){
  // The body at its sprung height, pitch and roll (rideStep); each wheel stands on the surface under it within the
  // suspension's travel. A hub's height in the world is y + x sin(pitch) + (hub cos(roll) - z sin(roll)) cos(pitch),
  // model z being the car's right.
  const fx=-Math.sin(rs.yaw),fz=-Math.cos(rs.yaw),lx=fz,lz=-fx;
  car.position.set(rs.x,rs.y,rs.z);car.rotation.set(0,rs.yaw+Math.PI/2,0);car.rotateZ(rs.pitch);car.rotateX(rs.roll);
  const sp=Math.sin(rs.pitch),cp=Math.cos(rs.pitch),sr=Math.sin(rs.roll),c=cp*Math.cos(rs.roll);
    wheels.forEach((w,i)=>{
   const [wx,wy,R]=WHEELS[i];const hNow=surfaceAt(rs.x+fx*wx+lx*wy,rs.z+fz*wx+lz*wy);
   const dy=(hNow+R-rs.y-wx*sp-wy*sr*cp)/c-R;w.hub.position.y=R+Math.max(-TRAVEL,Math.min(TRAVEL,dy));wheelTouch[i]=dy>-TRAVEL-.012;
   w.spin.rotation.z-=omega[i]*dt;if(i<2)w.hub.rotation.y=s.steer;
  });
 }
 function update(dt:number,input:CarInput){
  // Fixed physics steps, as many as the frame's time holds; the frame shows the state between the last two.
  acc+=dt;let n=0;
  hits.length=0;frame.impact=0;frame.impactKind=-1;frame.scrape=0;frame.bottom=0;frame.kerb=0;
  while(acc>=STEP-1e-9&&n<16){prev.x=s.x;prev.z=s.z;prev.yaw=s.yaw;prev.y=ride.z;prev.pitch=s.pitch;prev.roll=s.roll;step(STEP,input);acc-=STEP;n++;}
  if(n===16)acc=0;
  const k=acc/STEP;rs.x=prev.x+(s.x-prev.x)*k;rs.z=prev.z+(s.z-prev.z)*k;rs.yaw=lerpAngle(prev.yaw,s.yaw,k);rs.y=prev.y+(ride.z-prev.y)*k;rs.pitch=prev.pitch+(s.pitch-prev.pitch)*k;rs.roll=prev.roll+(s.roll-prev.roll)*k;
  pose(dt);syncCamera(dt);
  // brake lights: the tail lamps' LED bars and the third light in the spoiler brighten under braking
  const braking=(s.reverse?input.throttle:input.brake)>0;
  tailOn.emissiveIntensity=braking?5:.50;stopLamp.emissiveIntensity=braking?5:0;for(const [lamp,rest] of brakeLamps)lamp.emissiveIntensity=braking?rest*7:rest;
  // Tyre marks under sliding wheels that are on the ground; sparks where the body hit or scraped this frame.
  {const fx=-Math.sin(rs.yaw),fz=-Math.cos(rs.yaw),lx=fz,lz=-fx;
   for(let i=0;i<4;i++){const [wx,wy]=WHEELS[i],t=tracks[i];t.x=rs.x+fx*wx+lx*wy;t.z=rs.z+fz*wx+lz*wy;t.touching=wheelTouch[i]&&ride.air<.03;t.skid=wheelSkid[i];}
   effects.track(tracks,-lx,-lz);
   for(const hit of hits)effects.hit(hit);
   if(frame.bottom>1.6&&Math.abs(s.u)>4)effects.scrapeFloor(rs.x,surfaceAt(rs.x,rs.z),rs.z,fx*s.u+lx*s.v,fz*s.u+lz*s.v,Math.min(2,(frame.bottom-1.2)/1.5));
   effects.update(dt,camera);}
  if(input.throttle>0||input.brake>0)sound.start();
  sound.update({rpm:s.rpm,throttle:s.throttle,speed:Math.abs(s.u),gear:s.reverse?-1:s.gear,limiterRpm:SPEC.limiterRpm,skid:s.squeal,scrape:frame.scrape,airborne:ride.air>.08},dt);
  if(frame.impact>.9)sound.impact(frame.impact,frame.impactKind);
  const thud=Math.max(frame.kerb/4,(frame.bottom-.8)/3);if(thud>.05)sound.bump(Math.min(1,thud));
  if(Math.abs(s.u)>2&&driven<8){driven+=dt;if(driven>=8&&helpEl&&helpEl!==(hud as unknown))helpEl.hidden=true;}
  gauge();hudTimer+=dt;
  if(hudTimer>.1){
   hudTimer=0;
   const speed=String(Math.round(Math.abs(s.u)*3.6));
   const gear=s.reverse?'R':(Math.abs(s.u)<.3&&s.throttle===0?'N':String(s.gear));
   if(speedEl.textContent!==speed)speedEl.textContent=speed;
   if(gearEl.textContent!==gear)gearEl.textContent=gear;
  }
 }
 function telemetry():CarTelemetry{return {speedKmh:Math.abs(s.u)*3.6,gear:s.reverse?'R':String(s.gear),rpm:s.rpm,throttle:s.throttle,slip:s.slip};}
 return {group:car,body,update,placeAt,telemetry,get forwardSpeedMps(){return s.u;},get yawRateRadPerSec(){return s.r;},
  look:(deltaYaw:number)=>{lookYaw=Math.max(-2.6,Math.min(2.6,lookYaw+deltaYaw));},
  relax:(dt:number)=>{lookYaw*=Math.max(0,1-dt*2.2);},
  /** Toggles (or sets) the car's sound; remembered across visits. Returns whether it is now muted. */
  /** Toggles (or sets) traction and stability control; remembered across visits. Returns whether they are now on. */
  aids:(on?:boolean)=>{aids=on??!aids;tc=1;try{globalThis.localStorage?.setItem(AIDS_KEY,aids?'on':'off');}catch{}aidsBadge();return aids;},
  mute:(on?:boolean)=>{sound.setMuted(on??!sound.muted);return sound.muted;},
  effects,
  toggleCamera:()=>{cameraMode=cameraMode==='chase'?'bonnet':'chase';syncCamera(0,true);},
  position:()=>[s.x,s.z,s.yaw] as [number,number,number],
  debug:()=>({...trace,y:ride.z,zv:ride.zv,load:ride.load,air:ride.air,skid:[s.skidF,s.skidR],omega:[...omega],kappa:[...kappa],alpha:[...alpha],fz:[...fz],tc,abs:[...abs],squeal:s.squeal,wheelTouch:[...wheelTouch],sparks:effects.sparkCount,liveSparks:effects.liveSparks,markSegments:effects.markSegments,pitch:s.pitch,roll:s.roll,wheelH:[...wheelH],hubs:wheels.map(w=>w.hub.position.y),impact:{...impact},contactList:contacts.slice(0,contactCount).map(c=>({kind:c.kind,pen:+c.pen.toFixed(3),n:[+c.nx.toFixed(2),+c.nz.toFixed(2)],at:[+c.px.toFixed(2),+c.pz.toFixed(2)],jn:+c.jn.toFixed(1)}))}),
  stop:()=>{sound.stop();hud.hidden=true;},
  dispose:()=>{sound.close();effects.dispose();hud.remove();scene.remove(car);car.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(mm=>mm.dispose());}});}};
}
export type Car=ReturnType<typeof createCar>;
