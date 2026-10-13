import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

// A drivable mid-engined roadster with a planar vehicle model: rigid body (position, yaw, longitudinal and
// lateral velocity, yaw rate), bicycle tyre model with slip angles and a magic-formula lateral force, friction
// circle for combined slip, longitudinal load transfer, engine torque curve through a six-speed gearbox to the
// rear axle, aerodynamic drag and rolling resistance, stepped at a fixed rate. Buildings are a distance field
// rasterised from the scene's foundation outlines, street objects are circles (car-ground.ts), and the body's
// rounded outline meets both through contact impulses, so the car slides along walls and glances off posts. The
// wheels stand on the drivable surface (car-ground.ts): a kerb is a step they climb. Nothing here is location specific.

import type {CarGround} from './car-ground';

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
function lateralForce(slipRad:number,load:number,mu:number){
 // Pacejka-style with a broad peak near 14 degrees and almost no fall-off past it: slides are holdable and
 // recoverable, the way an arcade-simulation tyre behaves, rather than snapping away.
 const B=9,C=1.35,D=mu*Math.max(load,0);
 return -D*Math.sin(C*Math.atan(B*slipRad));
}

// ---- collision field -------------------------------------------------------------------------------------
// Distance (m, up to FIELD_CAP) to the nearest cell the car cannot enter. Cells are marked at load; the distances are an
// exact squared-distance transform (Felzenszwalb-Huttenlocher, rows then columns) made tile by tile where the car or the
// chase camera first asks, over the tile and a margin as wide as the cap, so a tile costs well under a millisecond and
// nothing is computed for streets never driven.
const FIELD_TILE=64,FIELD_CAP=4;
class DistanceField{
 readonly cell:number;readonly n:number;readonly origin:number;readonly mask:Uint8Array;
 private readonly per:number;private readonly pad:number;private readonly tiles=new Map<number,Float32Array>();
 constructor(extentM:number,cellM:number){this.cell=cellM;this.n=Math.ceil(extentM*2/cellM)+1;this.origin=-extentM;this.mask=new Uint8Array(this.n*this.n);this.per=Math.ceil(this.n/FIELD_TILE);this.pad=Math.ceil(FIELD_CAP/cellM);}
 private index(x:number,z:number){const i=Math.floor((x-this.origin)/this.cell),j=Math.floor((z-this.origin)/this.cell);return (i<0||j<0||i>=this.n||j>=this.n)?-1:j*this.n+i;}
 markTriangle(ax:number,az:number,bx:number,bz:number,cx:number,cz:number){
  const i0=Math.max(0,Math.floor((Math.min(ax,bx,cx)-this.origin)/this.cell)),i1=Math.min(this.n-1,Math.ceil((Math.max(ax,bx,cx)-this.origin)/this.cell));
  const j0=Math.max(0,Math.floor((Math.min(az,bz,cz)-this.origin)/this.cell)),j1=Math.min(this.n-1,Math.ceil((Math.max(az,bz,cz)-this.origin)/this.cell));
  const det=(bx-ax)*(cz-az)-(cx-ax)*(bz-az);if(Math.abs(det)<1e-9)return;
  for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){const px=this.origin+(i+.5)*this.cell,pz=this.origin+(j+.5)*this.cell;
   const w0=((bx-px)*(cz-pz)-(cx-px)*(bz-pz))/det,w1=((cx-px)*(az-pz)-(ax-px)*(cz-pz))/det,w2=1-w0-w1;
   if(w0>=-1e-6&&w1>=-1e-6&&w2>=-1e-6)this.mask[j*this.n+i]=1;}
 }
 markPoint(x:number,z:number){const idx=this.index(x,z);if(idx>=0)this.mask[idx]=1;}
 markSegment(x0:number,z0:number,x1:number,z1:number){const len=Math.hypot(x1-x0,z1-z0);const steps=Math.max(1,Math.ceil(len/(this.cell*.7)));for(let s=0;s<=steps;s++){const k=s/steps;const idx=this.index(x0+(x1-x0)*k,z0+(z1-z0)*k);if(idx>=0)this.mask[idx]=1;}}
 finish(){this.tiles.clear();}
 get tilesMade(){return this.tiles.size;}
 private tile(ti:number,tj:number){
  const key=tj*this.per+ti;let t=this.tiles.get(key);if(t)return t;
  const T=FIELD_TILE,P=this.pad,n=this.n,i0=Math.max(0,ti*T-P),j0=Math.max(0,tj*T-P),i1=Math.min(n,ti*T+T+P),j1=Math.min(n,tj*T+T+P),W=i1-i0,H=j1-j0;
  const d=new Float64Array(W*H);for(let j=0;j<H;j++)for(let i=0;i<W;i++)d[j*W+i]=this.mask[(j0+j)*n+i0+i]?0:1e12;
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
 try{const response=await fetch(url,{cache:'no-cache'});if(!response.ok)return null;return (await new GLTFLoader().parseAsync(await response.arrayBuffer(),'/')).scene;}catch{return null;}
}
function finishes(){
 const std=(o:THREE.MeshStandardMaterialParameters)=>new THREE.MeshStandardMaterial(o);
 const phys=(o:THREE.MeshPhysicalMaterialParameters)=>new THREE.MeshPhysicalMaterial(o);
 const f={
  // solid white under a clear coat: the base is a slightly rough pigment layer, the coat a near-mirror
  paint:phys({color:'#f2f2ee',metalness:0,roughness:.38,clearcoat:1,clearcoatRoughness:.03}),
  hoop:phys({color:'#c6c9cd',metalness:.3,roughness:.34,clearcoat:1,clearcoatRoughness:.05}),
  black_gloss:phys({color:'#070808',metalness:0,roughness:.42,clearcoat:.12,clearcoatRoughness:.2}),
  black_satin:std({color:'#121315',metalness:0,roughness:.55}),
  black_matte:std({color:'#0d0d0e',metalness:0,roughness:.85}),
  gap:std({color:'#2a2b2c',metalness:0,roughness:.85}),
  glass:phys({color:'#10161b',metalness:0,roughness:.02,transparent:true,opacity:.42,side:THREE.DoubleSide,depthWrite:false}),
  lens:phys({color:'#ffffff',metalness:0,roughness:.02,transparent:true,opacity:.14,depthWrite:false,clearcoat:1,clearcoatRoughness:.02}),
  lamp_chrome:std({color:'#9aa0a6',metalness:1,roughness:.18}),
  lamp_black:std({color:'#16181a',metalness:.4,roughness:.35}),
  drl:std({color:'#f4f7ff',emissive:'#eaf1ff',emissiveIntensity:2.2,roughness:.3}),
  tail_lens:phys({color:'#2a0406',metalness:0,roughness:.06,clearcoat:1,clearcoatRoughness:.02}),
  tail_led:std({color:'#5a0508',emissive:'#ff1d18',emissiveIntensity:1.4,roughness:.4}),
  reflector:phys({color:'#6d0a0c',metalness:0,roughness:.2,clearcoat:1,clearcoatRoughness:.05}),
  tyre:std({color:'#161616',metalness:0,roughness:.9}),
  rim:std({color:'#1b1c1e',metalness:.55,roughness:.4,side:THREE.DoubleSide}),   // the barrel is an open surface
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
function buildModel(source:THREE.Object3D){
 const car=new THREE.Group();const materials=finishes();
 const wheelParts={front:[] as THREE.Mesh[],rear:[] as THREE.Mesh[]},hubParts={front:[] as THREE.Mesh[],rear:[] as THREE.Mesh[]};
 let body:THREE.Mesh|null=null;
 const meshes:THREE.Mesh[]=[];source.traverse(o=>{if(o instanceof THREE.Mesh)meshes.push(o);});
 for(const o of meshes){
  const [part,raw='']=o.name.split('__');const finish=raw.replace(/_\d+$/,'');const material=materials[(finish in materials?finish:'black_satin') as Finish];
  o.material=material;o.castShadow=material.name!=='glass'&&material.name!=='lens';o.receiveShadow=true;
  if(!o.geometry.attributes.normal)o.geometry.computeVertexNormals();
  const axle=part.endsWith('_front')?'front':'rear';
  if(part.startsWith('wheel_'))wheelParts[axle].push(o);
  else if(part.startsWith('hub_'))hubParts[axle].push(o);
  else{o.removeFromParent();car.add(o);if(part==='body'&&finish==='paint')body=o;}
 }
 const wheels:{hub:THREE.Group;spin:THREE.Group}[]=[];
 const clone=(m:THREE.Mesh,side:number)=>{const c=new THREE.Mesh(m.geometry,m.material);c.scale.z=side;c.castShadow=m.castShadow;c.receiveShadow=true;c.name=m.name;return c;};
 const makeWheel=(axle:'front'|'rear',x:number,z:number,radius:number)=>{
  const hub=new THREE.Group();hub.position.set(x,radius,z);const spin=new THREE.Group();hub.add(spin);const side=Math.sign(z);
  for(const m of wheelParts[axle])spin.add(clone(m,side));for(const m of hubParts[axle])hub.add(clone(m,side));
  car.add(hub);wheels.push({hub,spin});
 };
 makeWheel('front',AXLE,SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM);makeWheel('front',AXLE,-SPEC.trackFrontM/2,SPEC.wheelRadiusFrontM);
 makeWheel('rear',-AXLE,SPEC.trackRearM/2,SPEC.wheelRadiusM);makeWheel('rear',-AXLE,-SPEC.trackRearM/2,SPEC.wheelRadiusM);
 return {car,wheels,body,tailOn:materials.tail_led,materials};
}
export function buildRoadster(source:THREE.Object3D){return buildModel(source);}

// ---- engine sound ---------------------------------------------------------------------------------------
class EngineSound{
 // A flat-six voice: firing-order harmonics through a soft clipper and a throttle-opened low-pass, an exhaust rasp
 // and wind from shaped noise, an irregular idle lope that fades out by 2000 rpm, a limiter stutter, and on every
 // gear change a throttle cut with an exhaust pop (up) or a blip (down).
 private ctx:AudioContext|null=null;private master:GainNode|null=null;private tone:GainNode|null=null;private filter:BiquadFilterNode|null=null;
 private osc:{o:OscillatorNode;ratio:number}[]=[];private noise:AudioBuffer|null=null;
 private exhaust:{src:AudioBufferSourceNode;band:BiquadFilterNode;gain:GainNode}|null=null;private wind:{src:AudioBufferSourceNode;band:BiquadFilterNode;gain:GainNode}|null=null;
 private lastGear=0;private lope=0;private t=0;private engaged=false;private idle=0;
 // The page's first AudioContext blocks the main thread for about 0.2 s while the browser starts its audio stack.
 // Listing the audio devices starts that stack in the background, so the context costs a few ms when it is made.
 constructor(){void navigator.mediaDevices?.enumerateDevices?.().catch(()=>{});}
 start(){if(this.engaged&&this.ctx)return;this.engaged=true;clearTimeout(this.idle);if(this.ctx){void this.ctx.resume();return;}this.prepare();}   // called on every frame a pedal is down
 // Builds the voice ahead of the first throttle (on entering Drive) and leaves it suspended, silent, until start().
 prepare(){if(this.ctx)return;try{const ctx=new AudioContext();this.ctx=ctx;
  const master=ctx.createGain();master.gain.value=0;master.connect(ctx.destination);this.master=master;
  const shaper=ctx.createWaveShaper();const curve=new Float32Array(1024);for(let i=0;i<1024;i++){const x=i/511.5-1;curve[i]=Math.tanh(2.4*x);}shaper.curve=curve;shaper.oversample='2x';
  const filter=ctx.createBiquadFilter();filter.type='lowpass';filter.frequency.value=500;filter.Q.value=1.1;this.filter=filter;
  const tone=ctx.createGain();tone.gain.value=.5;this.tone=tone;tone.connect(shaper);shaper.connect(filter);filter.connect(master);
  for(const [type,ratio,detune,level] of [['sawtooth',1,0,.55],['square',.5,0,.3],['sawtooth',2,7,.16],['triangle',3,-5,.1],['sawtooth',1,-11,.3]] as [OscillatorType,number,number,number][]){const o=ctx.createOscillator();o.type=type;o.detune.value=detune;const g=ctx.createGain();g.gain.value=level;o.connect(g);g.connect(tone);o.start();this.osc.push({o,ratio});}
  const noise=ctx.createBuffer(1,ctx.sampleRate*2,ctx.sampleRate);const data=noise.getChannelData(0);for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;this.noise=noise;
  const shaped=(type:BiquadFilterType,freq:number,q:number)=>{const src=ctx.createBufferSource();src.buffer=noise;src.loop=true;const band=ctx.createBiquadFilter();band.type=type;band.frequency.value=freq;band.Q.value=q;const gain=ctx.createGain();gain.gain.value=0;src.connect(band);band.connect(gain);gain.connect(master);src.start();return {src,band,gain};};
  this.exhaust=shaped('bandpass',260,.8);this.wind=shaped('lowpass',420,.6);
  if(!this.engaged)void ctx.suspend();else if(ctx.state==='suspended')void ctx.resume();}catch{this.ctx=null;}}
 update(rpm:number,throttle:number,speed:number,gear:number,dt:number){
  if(!this.engaged||!this.ctx||!this.master||!this.filter)return;const now=this.ctx.currentTime;
  this.t+=dt;this.lope+=(Math.sin(this.t*9.2)+.6*Math.sin(this.t*13.7+1.3)+.3*(Math.random()-.5)-this.lope)*Math.min(1,dt*8);
  const idleness=Math.max(0,1-(rpm-SPEC.idleRpm)/1200);const f0=rpm/20*(1+.018*this.lope*idleness);
  for(const {o,ratio} of this.osc)o.frequency.setTargetAtTime(f0*ratio,now,.025);
  this.filter.frequency.setTargetAtTime(320+throttle*2600+rpm*.14,now,.04);
  const limiter=rpm>SPEC.limiterRpm-150&&throttle>.5?(Math.floor(this.t*22)%2?.35:1):1;
  const level=(.02+throttle*.045+rpm/SPEC.limiterRpm*.02)*(1+.1*this.lope*idleness)*limiter;
  this.master.gain.setTargetAtTime(level,now,.06);
  if(this.exhaust){this.exhaust.band.frequency.setTargetAtTime(220+rpm*.09,now,.05);this.exhaust.gain.gain.setTargetAtTime(.15+.6*throttle,now,.08);}
  if(this.wind){this.wind.band.frequency.setTargetAtTime(300+speed*14,now,.2);this.wind.gain.gain.setTargetAtTime(Math.min(.9,(speed/38)**2*1.2),now,.2);}
  if(gear!==this.lastGear){if(this.lastGear>0&&gear>0)this.shift(gear>this.lastGear);this.lastGear=gear;}
 }
 shift(up:boolean){
  if(!this.ctx||!this.master||!this.noise)return;const ctx=this.ctx,now=ctx.currentTime;const g=this.master.gain;const current=g.value;
  g.cancelScheduledValues(now);g.setValueAtTime(current,now);g.linearRampToValueAtTime(current*.3,now+.04);g.setTargetAtTime(current,now+.16,.05);
  const pop=ctx.createBufferSource();pop.buffer=this.noise;const band=ctx.createBiquadFilter();band.type='bandpass';band.frequency.value=up?160:420;band.Q.value=1.4;
  const pg=ctx.createGain();pg.gain.setValueAtTime(0,now);pg.gain.linearRampToValueAtTime(up?.35:.2,now+.012);pg.gain.exponentialRampToValueAtTime(.001,now+(up?.22:.14));
  pop.connect(band);band.connect(pg);pg.connect(this.master);pop.start(now);pop.stop(now+.3);
 }
 // Leaving Drive fades the voice out and suspends it, so the next drive resumes it rather than building a new one.
 stop(){this.engaged=false;if(!this.ctx)return;const ctx=this.ctx;this.master?.gain.setTargetAtTime(0,ctx.currentTime,.1);
  clearTimeout(this.idle);this.idle=window.setTimeout(()=>{if(!this.engaged)void ctx.suspend();},400);}
 close(){this.stop();clearTimeout(this.idle);const ctx=this.ctx,osc=this.osc,noises=[this.exhaust,this.wind];if(!ctx)return;this.ctx=null;this.osc=[];this.exhaust=null;this.wind=null;
  setTimeout(()=>{for(const {o} of osc){try{o.stop()}catch{}}for(const n of noises){try{n?.src.stop()}catch{}}void ctx.close();},400);}
}

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
const EDGE=262;                            // the world's edge, a soft wall
const RESTITUTION=.12,FRICTION=.3;         // a hit keeps a little of its approach speed; a scraping contact scrubs speed
const KERB=.3;                             // share of its approach speed a wheel loses climbing a step as tall as its radius
const TRAVEL=.09;                          // suspension travel either way from the ride height (m)
const HEAVE={w:2*Math.PI*1.8,z:.55},TILT={w:2*Math.PI*1.6,z:.6};   // body springs: natural frequency, damping ratio
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
export function createCar(scene:THREE.Scene,host:HTMLElement,camera:THREE.PerspectiveCamera,groundAt:(x:number,z:number)=>number,field:DistanceField|null,source:THREE.Object3D,ground:CarGround|null=null){
 const {car,wheels,body,tailOn}=buildModel(source);scene.add(car);
 const sound=new EngineSound();
 const hud=document.createElement('div');hud.className='car-hud';hud.setAttribute('aria-live','off');
 hud.innerHTML=gaugeMarkup()+'<div class="car-help">W / S throttle and brake · A / D steer · Space handbrake · C camera · R reset · drag to look</div>';
 host.appendChild(hud);const speedEl=hud.querySelector('.tach-speed')!,gearEl=hud.querySelector('.tach-gear')!,needle=hud.querySelector('.tach-needle') as SVGLineElement,fill=hud.querySelector('.tach-fill') as SVGPathElement;
 function gauge(){const frac=Math.min(1,Math.max(0,s.rpm/TACH.max));needle.setAttribute('transform',`rotate(${(TACH.start+TACH.sweep*frac).toFixed(2)} ${TACH.cx} ${TACH.cy})`);fill.setAttribute('stroke-dasharray',`${(frac*1000).toFixed(1)} 1000`);fill.classList.toggle('is-hot',s.rpm>=TACH.red);}
 // State in the body frame: u forward, v left, r yaw rate (positive turns left). Heading: forward=(-sin yaw, -cos yaw).
 const s={x:0,z:0,yaw:0,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,ay:0,throttle:0,rpm:SPEC.idleRpm,slip:0,pitch:0,roll:0};
 let trace:Record<string,number>={};
 let cameraMode:'chase'|'bonnet'='chase';let lookYaw=0;let hudTimer=0;let reverseHold=0;let shiftTimer=0;
 const camPos=new THREE.Vector3(),camLook=new THREE.Vector3();
 const {massKg:m,wheelbaseM:L,cgToFrontM:a,cgHeightM:h,yawInertia:Iz,wheelRadiusM:Rw}=SPEC;const b=L-a;
 const surfaceAt=ground?(x:number,z:number)=>ground.surface.at(x,z):groundAt;
 // The body rides on its four wheels: the surface under each (front right, front left, rear right, rear left) and its
 // height above the terrain grid (a kerb is a jump in it, a hill is not); the body's height, pitch and roll are springs
 // toward the plane of the four contacts.
 const wheelH=new Float64Array(4),detail=new Float64Array(4);
 const ride={z:0,zv:0,pv:0,qv:0,t:[0,0,0,0,0,0] as number[]};
 // The frame shows the physics state interpolated between the last two steps.
 const prev={x:0,z:0,yaw:0,y:0},rs={x:0,z:0,yaw:0,y:0};let acc=0;
 const contacts:Contact[]=[];let contactCount=0;const nearby:number[]=[];
 // the hardest hit since the car was placed: approach speed along the contact normal (m/s) and what was hit (kind)
 const impact={speed:0,kind:-1};
 function placeAt(x:number,z:number,yaw:number){Object.assign(s,{x,z,yaw,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,ay:0,rpm:SPEC.idleRpm});settle();acc=0;impact.speed=0;impact.kind=-1;Object.assign(prev,{x,z,yaw,y:ride.z});Object.assign(rs,prev);lookYaw=0;hud.style.display='';syncCamera(0,true);if(navigator.userActivation?.hasBeenActive)sound.prepare();}
 function forward(){return [-Math.sin(s.yaw),-Math.cos(s.yaw)];}
 function left(){const [fx,fz]=forward();return [fz,-fx];}
 function step(hdt:number,input:CarInput){
  const {u,v,r}=s;const absU=Math.abs(u);
  // Steering: rate limited, and the wheels turn less at speed, as a driver would.
  // Speed-sensitive lock; a snapping input (keys) winds on at a paced rate and unwinds faster, so the yaw response
  // is smooth however the input arrives.
  const maxSteer=SPEC.maxSteerRad/(1+(absU/11)**2);const target=input.steer*maxSteer;const rate=(Math.abs(target)<Math.abs(s.steer)?8:3.8)/(1+absU/30);
  s.steer+=Math.max(-rate*hdt,Math.min(rate*hdt,target-s.steer));const delta=s.steer;
  // Gearbox: automatic shifts by engine speed, reverse engaged from rest with the brake pedal held.
  if(absU<.4&&input.brake>0&&input.throttle===0){reverseHold+=hdt;if(reverseHold>.35)s.reverse=true;}else if(absU<.4&&input.throttle>0&&s.reverse&&input.brake===0){s.reverse=false;}
  if(input.throttle===0&&input.brake===0)reverseHold=0;
  const drivePedal=s.reverse?(absU>7?0:input.brake):input.throttle;const brakePedal=s.reverse?input.throttle:input.brake;
  const ratio=(s.reverse?SPEC.gears[0]:SPEC.gears[s.gear-1])*SPEC.finalDrive;
  let rpm=Math.abs(u)/Rw*ratio*60/(2*Math.PI);
  shiftTimer=Math.max(0,shiftTimer-hdt);
  if(!s.reverse&&shiftTimer===0){if(rpm>SPEC.upshiftRpm&&s.gear<SPEC.gears.length){s.gear++;shiftTimer=.25;}else if(rpm<SPEC.downshiftRpm&&s.gear>1){s.gear--;shiftTimer=.25;}}
  rpm=Math.max(SPEC.idleRpm,Math.min(SPEC.limiterRpm,rpm));s.rpm+=(rpm-s.rpm)*Math.min(1,hdt*12);
  const clutch=Math.min(1,absU/2.5+.25);   // slipping clutch from rest
  const rearSlipNow=Math.abs(Math.atan((v-b*r)/(absU<.6?.6:u)));const stmTrim=(!input.handbrake&&rearSlipNow>.28)?.55:1;
  const wheelTorque=engineTorqueNm(rpm)*drivePedal*(shiftTimer>0?.15:1)*ratio*SPEC.drivelineEfficiency*clutch*stmTrim;
  const engineBrake=drivePedal===0&&absU>.5?-Math.sign(u)*(30+rpm*.012)*ratio*SPEC.drivelineEfficiency:0;
  let fxRear=(wheelTorque*(s.reverse?-1:1)+engineBrake)/Rw;
  // Loads with longitudinal transfer from the last step's acceleration.
  const fzFront=Math.max(1000,m*G*b/L-m*s.ax*h/L),fzRear=Math.max(1000,m*G*a/L+m*s.ax*h/L);
  // Brakes: 1.2 g total, split to the front, clamped to what each axle can hold (ABS).
  const brakeForce=brakePedal*SPEC.maxBrakeG*m*G*(absU>.05?-Math.sign(u):0);
  let fxFront=brakeForce*SPEC.brakeFrontShare;fxRear+=brakeForce*(1-SPEC.brakeFrontShare);
  // Tyres: slip angles from the body velocities at each axle; the handbrake locks the rears.
  const uEff=absU<.6?(u<0?-.6:.6):u;
  const alphaF=Math.atan((v+a*r)/uEff)-delta*Math.sign(uEff),alphaR=Math.atan((v-b*r)/uEff);
  const muRear=input.handbrake?SPEC.lockedTyreMu:SPEC.tyreMu;
  if(input.handbrake)fxRear=-Math.sign(u)*muRear*fzRear*(absU>.3?.85:0);
  let fyFront=lateralForce(alphaF,fzFront,SPEC.tyreMu),fyRear=lateralForce(alphaR,fzRear,muRear);
  // Friction circle: an axle cannot give more than mu times its load in any direction. With the stability
  // electronics on (everything but the handbrake), the lateral force keeps priority and the longitudinal
  // demand is cut to what is left: traction control on the driven axle, ABS on a braking one, so a floored
  // throttle or a stamped brake in a bend does not throw the car. The handbrake hands the rear to the driver.
  const circle=(fx:number,fy:number,limit:number,keepLateral:boolean)=>{
   if(keepLateral){
    // Traction control that lets a third of the excess through: the tyre can run a little over its circle,
    // so a floored throttle in a bend loosens the rear progressively instead of either cutting all power or
    // throwing the car; the stability damping below catches the yaw.
    const fyc=Math.max(-limit,Math.min(limit,fy));const room=Math.sqrt(Math.max(0,limit*limit-fyc*fyc));
    const fxc=Math.abs(fx)>room?Math.sign(fx)*(room+(Math.abs(fx)-room)*.33):fx;const mag=Math.hypot(fxc,fyc);const cap=limit*1.08;
    return mag>cap?[fxc*cap/mag,fyc*cap/mag]:[fxc,fyc];}
   const mag=Math.hypot(fx,fy);return mag>limit?[fx*limit/mag,fy*limit/mag]:[fx,fy];};
  [fxFront,fyFront]=circle(fxFront,fyFront,SPEC.tyreMu*fzFront,true);[fxRear,fyRear]=circle(fxRear,fyRear,muRear*fzRear,!input.handbrake);
  s.slip=Math.max(Math.abs(alphaF),Math.abs(alphaR));
  const drag=-.5*1.2*SPEC.dragArea*u*absU;const rolling=absU>.1?-Math.sign(u)*SPEC.rollingResistance*m*G:0;
  const cosD=Math.cos(delta),sinD=Math.sin(delta);
  const ax=(fxFront*cosD-fyFront*sinD+fxRear+drag+rolling)/m+v*r;
  const ay=(fyFront*cosD+fxFront*sinD+fyRear)/m-u*r;
  let rDot=(a*(fyFront*cosD+fxFront*sinD)-b*fyRear)/Iz;
  // Stability control: yaw beyond what the steering asks for at this speed is damped, as the brakes of a
  // stability system would; the handbrake switches it off.
  if(!input.handbrake&&absU>2){const rGrip=.9*SPEC.tyreMu*G/absU;const rRef=Math.max(-rGrip,Math.min(rGrip,u*Math.tan(delta)/L));const excess=r-rRef;if(Math.abs(excess)>.08){const e=excess-Math.sign(excess)*.08;rDot-=e*9;rDot-=Math.sign(e)*Math.min(Math.abs(e),.5)*a*Math.abs(fyFront)*.35/Iz;}}
  trace={u,v,r,delta,alphaF,alphaR,fyFront,fyRear,fxRear,fxFront,ax,ay,rDot,fzFront,fzRear,gear:s.gear,rpm:s.rpm,shift:shiftTimer,sax:s.ax};
  s.u+=ax*hdt;s.v+=ay*hdt;s.r+=rDot*hdt;s.ax+=(ax-s.ax)*.3;s.ay=ay;
  // Parking speeds: the slip-angle tyre has no meaning near rest, so the body is blended toward the kinematic
  // turn the steering asks for (yaw rate u tan(delta)/L, rear axle tracking), fully at walking pace.
  if(absU<3){const k=Math.min(1,(3-absU)/2.5)*Math.min(1,hdt*14);const rKin=s.u*Math.tan(delta)/L;s.r+=(rKin-s.r)*k;s.v+=(b*s.r-s.v)*k;if(drivePedal===0&&Math.abs(s.u)<.15&&brakePedal>0)s.u=0;}
  s.yaw+=s.r*hdt;
  const [fx,fz]=forward(),[lx,lz]=left();
  s.x+=(fx*s.u+lx*s.v)*hdt;s.z+=(fz*s.u+lz*s.v)*hdt;
  s.throttle=drivePedal;
  rideStep(hdt);collide();
  trace.contacts=contactCount;trace.rideZ=ride.z;
 }
 // World velocity of a point of the body at offset (px, pz) from the origin, and an impulse applied there.
 function pointVelocity(px:number,pz:number):[number,number]{const [fx,fz]=forward(),[lx,lz]=left();return [fx*s.u+lx*s.v+s.r*pz,fz*s.u+lz*s.v-s.r*px];}
 function impulse(jx:number,jz:number,px:number,pz:number){const [fx,fz]=forward(),[lx,lz]=left();const vx=fx*s.u+lx*s.v+jx/m,vz=fz*s.u+lz*s.v+jz/m;s.u=vx*fx+vz*fz;s.v=vx*lx+vz*lz;s.r+=(pz*jx-px*jz)/Iz;}
 function sample(){const [fx,fz]=forward(),[lx,lz]=left();for(let i=0;i<4;i++){const [wx,wy]=WHEELS[i];const x=s.x+fx*wx+lx*wy,z=s.z+fz*wx+lz*wy;wheelH[i]=surfaceAt(x,z);detail[i]=wheelH[i]-groundAt(x,z);}}
 // Ride targets from the four contacts: the terrain grid under them (hills, followed exactly) and the surface's height above
 // it (kerbs, which the springs smooth), plus squat under power, dive on the brakes and a lean out of the bend (clamped).
 function plane(h:ArrayLike<number>):[number,number,number]{const hF=(h[0]+h[1])/2,hR=(h[2]+h[3])/2,hL=(h[1]+h[3])/2,hRt=(h[0]+h[2])/2;return [(hF+hR)/2,Math.atan2(hF-hR,L),Math.atan2(hL-hRt,(SPEC.trackFrontM+SPEC.trackRearM)/2)];}
 function targets():[number,number,number,number,number,number]{
  const base=[wheelH[0]-detail[0],wheelH[1]-detail[1],wheelH[2]-detail[2],wheelH[3]-detail[3]];const [bz,bp,bq]=plane(base),[dz,dp,dq]=plane(detail);
  return [bz,bp,bq,dz,dp+Math.max(-.07,Math.min(.07,s.ax*.0065)),dq+Math.max(-.08,Math.min(.08,s.ay*.011))];
 }
 function settle(){sample();const t=targets();ride.z=t[0]+t[3];s.pitch=t[1]+t[4];s.roll=t[2]+t[5];ride.zv=ride.pv=ride.qv=0;ride.t=t;}
 // Uphill direction of the surface's height above the terrain grid (a kerb's face), or none where it is smooth.
 function stepDirection(x:number,z:number):[number,number]{const d=.3,e=(p:number,q:number)=>surfaceAt(p,q)-groundAt(p,q);const gx=(e(x+d,z)-e(x-d,z))/(2*d),gz=(e(x,z+d)-e(x,z-d))/(2*d),g=Math.hypot(gx,gz);return g>.06?[gx/g,gz/g]:[0,0];}
 function rideStep(hdt:number){
  const [fx,fz]=forward(),[lx,lz]=left();
  for(let i=0;i<4;i++){
   const [wx,wy,R]=WHEELS[i];const px=fx*wx+lx*wy,pz=fz*wx+lz*wy,x=s.x+px,z=s.z+pz;const hNow=surfaceAt(x,z),dNow=hNow-groundAt(x,z),rise=dNow-detail[i];wheelH[i]=hNow;detail[i]=dNow;
   // A wheel meeting a step has to climb it: the car loses speed at that corner along the step's uphill direction, in
   // proportion to the rise against the wheel's radius, and the impulse at the corner turns the car a little.
   if(rise>.002){const [nx,nz]=stepDirection(x,z);if(nx||nz){const [vx,vz]=pointVelocity(px,pz);const vn=vx*nx+vz*nz;if(vn>0){const j=KERB*m/4*Math.min(1,rise/R)*vn;impulse(-nx*j,-nz*j,px,pz);}}}
  }
  // Springs with dampers on the body's motion relative to the contacts: the terrain's own rise is fed forward whole, so a
  // hill is followed without lag, and a kerb's (the surface above the terrain) at a capped rate, so it is a jolt, not a kick.
  const t=targets(),t0=ride.t;
  const spring=(x:number,v:number,i:number,cap:number,k:{w:number;z:number}):[number,number]=>{const rate=(t[i]-t0[i])/hdt+Math.max(-cap,Math.min(cap,(t[i+3]-t0[i+3])/hdt));v+=(k.w*k.w*(t[i]+t[i+3]-x)+2*k.z*k.w*(rate-v))*hdt;return [x+v*hdt,v];};
  [ride.z,ride.zv]=spring(ride.z,ride.zv,0,.6,HEAVE);[s.pitch,ride.pv]=spring(s.pitch,ride.pv,1,.35,TILT);[s.roll,ride.qv]=spring(s.roll,ride.qv,2,.35,TILT);
  ride.t=t;
 }
 function collide(){
  const [fx,fz]=forward(),[lx,lz]=left();contactCount=0;
  const add=(nx:number,nz:number,px:number,pz:number,pen:number,kind=0)=>{let c=contacts[contactCount];if(!c){c={nx:0,nz:0,px:0,pz:0,pen:0,bias:0,jn:0,jt:0,kind:0};contacts.push(c);}c.nx=nx;c.nz=nz;c.px=px;c.pz=pz;c.pen=pen;c.bias=0;c.jn=0;c.jt=0;c.kind=kind;contactCount++;};
  // Buildings, steps and high rises (the distance field) and the world's edge, against probes along the outline;
  // each contact pushes the car along the field's gradient, away from what it touches.
  for(const [bx,by] of OUTLINE){
   const px=fx*bx+lx*by,pz=fz*bx+lz*by,x=s.x+px,z=s.z+pz;
   const radial=Math.hypot(x,z);if(radial>EDGE){add(-x/radial,-z/radial,px,pz,radial-EDGE,1);continue;}
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
  for(let i=0;i<contactCount;i++){const c=contacts[i];const [vx,vz]=pointVelocity(c.px,c.pz);const vn=vx*c.nx+vz*c.nz;c.bias=vn<-.5?-RESTITUTION*vn:0;if(-vn>impact.speed){impact.speed=-vn;impact.kind=c.kind;}}
  for(let it=0;it<4;it++)for(let i=0;i<contactCount;i++){
   const c=contacts[i];let [vx,vz]=pointVelocity(c.px,c.pz);
   const rn=c.pz*c.nx-c.px*c.nz,jn=Math.max(0,c.jn+(c.bias-(vx*c.nx+vz*c.nz))/(1/m+rn*rn/Iz)),dn=jn-c.jn;c.jn=jn;impulse(c.nx*dn,c.nz*dn,c.px,c.pz);
   [vx,vz]=pointVelocity(c.px,c.pz);const tx=-c.nz,tz=c.nx,rt=c.pz*tx-c.px*tz,lim=FRICTION*c.jn,jt=Math.max(-lim,Math.min(lim,c.jt-(vx*tx+vz*tz)/(1/m+rt*rt/Iz))),dt=jt-c.jt;c.jt=jt;impulse(tx*dt,tz*dt,c.px,c.pz);
  }
  // A glancing hit turns the car, but it does not leave it spinning: some yaw goes while it is in contact.
  s.r*=.985;
 }
 function lerpAngle(a:number,b:number,k:number){let d=b-a;while(d>Math.PI)d-=2*Math.PI;while(d<-Math.PI)d+=2*Math.PI;return a+d*k;}
 let camYaw=0,camDist=4.9;
 // How far the camera may trail the car before it stands inside a wall: the trailing ray is marched through the
 // distance field and stops at the last free point.
 function freeDistance(bx:number,bz:number,want:number){if(!field)return want;let d=2.4;for(let t=2.4;t<=want;t+=.25){if(field.at(rs.x-bx*t,rs.z-bz*t)<.7)break;d=t;}return d;}
 function syncCamera(dt:number,snap=false){
  // the cameras follow the rendered state: the body's own height (so a kerb lifts them with it) and its interpolated place
  const speed=Math.abs(s.u);const y=rs.y;const fx=-Math.sin(rs.yaw),fz=-Math.cos(rs.yaw);
  if(cameraMode==='bonnet'){const ly=rs.yaw+lookYaw;camera.position.set(rs.x+fx*.25,y+1.1+s.pitch*.3,rs.z+fz*.25);camera.up.set(0,1,0);camera.lookAt(rs.x-Math.sin(ly)*30,y+1.02,rs.z-Math.cos(ly)*30);camera.fov=64+Math.min(10,speed*.15);camera.updateProjectionMatrix();return;}
  // Chase camera in the arcade-simulation idiom: level at all times, trailing a heading that lags the car's and
  // leans toward the velocity direction in a slide (so a drift is seen sideways) and a little into a turn, sitting
  // further back with speed, swinging to the front when reversing, and easing in when a wall stands behind it
  // while keeping its downward angle, so nothing pitches or rolls with the steering. Every smoothing is time based.
  const reversing=s.u<-.8;const ahead=reversing?-1:1;
  const slide=Math.atan2(s.v,Math.max(1.5,speed))*ahead*Math.min(1,speed/6);
  const lead=Math.max(-.22,Math.min(.22,s.r*.16))*ahead;
  const targetYaw=rs.yaw+(reversing?Math.PI:0)+slide*.55+lead+lookYaw;
  camYaw=snap?targetYaw:lerpAngle(camYaw,targetYaw,1-Math.exp(-dt*4.2));
  const bx=-Math.sin(camYaw),bz=-Math.cos(camYaw);
  const allowed=freeDistance(bx,bz,6.6+speed*.055);
  camDist=snap?allowed:camDist+(allowed-camDist)*(1-Math.exp(-dt*(allowed<camDist?9:2.5)));
  // The camera is placed from its smoothed heading and distance, so when it swings round the car (a reverse, a
  // spin) it travels the arc and never cuts across the body; only its height and its aim are eased separately.
  // A fixed downward angle of nine degrees to the aim point whatever the geometry: the height follows the
  // horizontal distance to the aim, so neither speed, a wall behind, a slide nor a spin changes the pitch.
  const look=new THREE.Vector3(rs.x+fx*1.6*ahead,y+.9,rs.z+fz*1.6*ahead);
  if(snap)camLook.copy(look);else camLook.lerp(look,1-Math.exp(-dt*14));
  camPos.set(rs.x-bx*camDist,0,rs.z-bz*camDist);
  camPos.y=camLook.y+.158*Math.max(3,Math.hypot(camPos.x-camLook.x,camPos.z-camLook.z));
  camera.position.copy(camPos);camera.up.set(0,1,0);camera.lookAt(camLook);
  camera.fov=58+Math.min(12,speed*.25);camera.updateProjectionMatrix();
 }
 function pose(dt:number){
  // The body at its sprung height, pitch and roll (rideStep); each wheel stands on the surface under it within the
  // suspension's travel. A hub's height in the world is y + x sin(pitch) + (hub cos(roll) - z sin(roll)) cos(pitch),
  // model z being the car's right.
  const fx=-Math.sin(rs.yaw),fz=-Math.cos(rs.yaw),lx=fz,lz=-fx;
  car.position.set(rs.x,rs.y,rs.z);car.rotation.set(0,rs.yaw+Math.PI/2,0);car.rotateZ(s.pitch);car.rotateX(s.roll);
  const sp=Math.sin(s.pitch),cp=Math.cos(s.pitch),sr=Math.sin(s.roll),c=cp*Math.cos(s.roll);
  const spin=s.u/Rw*dt;
  wheels.forEach((w,i)=>{
   const [wx,wy,R]=WHEELS[i];const hNow=surfaceAt(rs.x+fx*wx+lx*wy,rs.z+fz*wx+lz*wy);
   const dy=(hNow+R-rs.y-wx*sp-wy*sr*cp)/c-R;w.hub.position.y=R+Math.max(-TRAVEL,Math.min(TRAVEL,dy));
   w.spin.rotation.z-=spin;if(i<2)w.hub.rotation.y=s.steer;
  });
 }
 function update(dt:number,input:CarInput){
  // Fixed physics steps, as many as the frame's time holds; the frame shows the state between the last two.
  acc+=dt;let n=0;
  while(acc>=STEP-1e-9&&n<16){prev.x=s.x;prev.z=s.z;prev.yaw=s.yaw;prev.y=ride.z;step(STEP,input);acc-=STEP;n++;}
  if(n===16)acc=0;
  const k=acc/STEP;rs.x=prev.x+(s.x-prev.x)*k;rs.z=prev.z+(s.z-prev.z)*k;rs.yaw=lerpAngle(prev.yaw,s.yaw,k);rs.y=prev.y+(ride.z-prev.y)*k;
  pose(dt);syncCamera(dt);
  // brake lights: the tail lamps' LED bars and the third light in the spoiler brighten under braking
  tailOn.emissiveIntensity=(s.reverse?input.throttle:input.brake)>0?5:1.4;
  if(input.throttle>0||input.brake>0)sound.start();sound.update(s.rpm,s.throttle,Math.abs(s.u),s.reverse?-1:s.gear,dt);
  gauge();hudTimer+=dt;if(hudTimer>.1){hudTimer=0;speedEl.textContent=String(Math.round(Math.abs(s.u)*3.6));gearEl.textContent=s.reverse?'R':(Math.abs(s.u)<.3&&s.throttle===0?'N':String(s.gear));}
 }
 function telemetry():CarTelemetry{return {speedKmh:Math.abs(s.u)*3.6,gear:s.reverse?'R':String(s.gear),rpm:s.rpm,throttle:s.throttle,slip:s.slip};}
 return {group:car,body,update,placeAt,telemetry,
  look:(deltaYaw:number)=>{lookYaw=Math.max(-2.6,Math.min(2.6,lookYaw+deltaYaw));},
  relax:(dt:number)=>{lookYaw*=Math.max(0,1-dt*2.2);},
  toggleCamera:()=>{cameraMode=cameraMode==='chase'?'bonnet':'chase';syncCamera(0,true);},
  position:()=>[s.x,s.z,s.yaw] as [number,number,number],
  debug:()=>({...trace,y:ride.z,pitch:s.pitch,roll:s.roll,wheelH:[...wheelH],hubs:wheels.map(w=>w.hub.position.y),impact:{...impact},contactList:contacts.slice(0,contactCount).map(c=>({kind:c.kind,pen:+c.pen.toFixed(3),n:[+c.nx.toFixed(2),+c.nz.toFixed(2)],at:[+c.px.toFixed(2),+c.pz.toFixed(2)],jn:+c.jn.toFixed(1)}))}),
  stop:()=>{sound.stop();hud.style.display='none';},
  dispose:()=>{sound.close();hud.remove();scene.remove(car);car.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(mm=>mm.dispose());}});}};
}
export type Car=ReturnType<typeof createCar>;
