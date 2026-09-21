import * as THREE from 'three';

// A drivable mid-engined roadster with a planar vehicle model: rigid body (position, yaw, longitudinal and
// lateral velocity, yaw rate), bicycle tyre model with slip angles and a magic-formula lateral force, friction
// circle for combined slip, longitudinal load transfer, engine torque curve through a six-speed gearbox to the
// rear axle, aerodynamic drag and rolling resistance. Buildings are a signed distance field rasterised from the
// scene's foundation outlines, so the car slides along walls. Nothing here is location specific.

export type CarInput={throttle:number;brake:number;steer:number;handbrake:boolean};
export type CarTelemetry={speedKmh:number;gear:string;rpm:number;throttle:number;slip:number};

const G=9.81;
const SPEC={
 massKg:1485,           // 1,405 kg unladen plus a driver
 wheelbaseM:2.475,
 cgToFrontM:1.36,       // 45/55 mid-engine weight split
 cgHeightM:0.45,
 yawInertia:2600,
 trackFrontM:1.515,
 trackRearM:1.532,
 wheelRadiusM:0.345,    // 265/35 R20
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
const FAR=1e9;   // squared cell distance for unmarked cells; Infinity would make the envelope arithmetic NaN
class DistanceField{
 readonly cell:number;readonly n:number;readonly origin:number;readonly data:Float32Array;
 constructor(extentM:number,cellM:number){this.cell=cellM;this.n=Math.ceil(extentM*2/cellM)+1;this.origin=-extentM;this.data=new Float32Array(this.n*this.n).fill(FAR);}
 private index(x:number,z:number){const i=Math.floor((x-this.origin)/this.cell),j=Math.floor((z-this.origin)/this.cell);return (i<0||j<0||i>=this.n||j>=this.n)?-1:j*this.n+i;}
 markTriangle(ax:number,az:number,bx:number,bz:number,cx:number,cz:number){
  const i0=Math.max(0,Math.floor((Math.min(ax,bx,cx)-this.origin)/this.cell)),i1=Math.min(this.n-1,Math.ceil((Math.max(ax,bx,cx)-this.origin)/this.cell));
  const j0=Math.max(0,Math.floor((Math.min(az,bz,cz)-this.origin)/this.cell)),j1=Math.min(this.n-1,Math.ceil((Math.max(az,bz,cz)-this.origin)/this.cell));
  const det=(bx-ax)*(cz-az)-(cx-ax)*(bz-az);if(Math.abs(det)<1e-9)return;
  for(let j=j0;j<=j1;j++)for(let i=i0;i<=i1;i++){const px=this.origin+(i+.5)*this.cell,pz=this.origin+(j+.5)*this.cell;
   const w0=((bx-px)*(cz-pz)-(cx-px)*(bz-pz))/det,w1=((cx-px)*(az-pz)-(ax-px)*(cz-pz))/det,w2=1-w0-w1;
   if(w0>=-1e-6&&w1>=-1e-6&&w2>=-1e-6)this.data[j*this.n+i]=0;}
 }
 markSegment(x0:number,z0:number,x1:number,z1:number){const len=Math.hypot(x1-x0,z1-z0);const steps=Math.max(1,Math.ceil(len/(this.cell*.7)));for(let s=0;s<=steps;s++){const k=s/steps;const idx=this.index(x0+(x1-x0)*k,z0+(z1-z0)*k);if(idx>=0)this.data[idx]=0;}}
 finish(){
  // Felzenszwalb-Huttenlocher squared distance transform, rows then columns; distances in metres.
  const n=this.n,d=this.data,f=new Float32Array(n),out=new Float32Array(n),v=new Int32Array(n),zb=new Float32Array(n+1);
  const pass=(read:(k:number)=>number,write:(k:number,val:number)=>void)=>{
   for(let k=0;k<n;k++)f[k]=read(k);
   let q=0;v[0]=0;zb[0]=-Infinity;zb[1]=Infinity;
   for(let k=1;k<n;k++){let s=((f[k]+k*k)-(f[v[q]]+v[q]*v[q]))/(2*k-2*v[q]);while(s<=zb[q]){q--;s=((f[k]+k*k)-(f[v[q]]+v[q]*v[q]))/(2*k-2*v[q]);}q++;v[q]=k;zb[q]=s;zb[q+1]=Infinity;}
   q=0;for(let k=0;k<n;k++){while(zb[q+1]<k)q++;out[k]=(k-v[q])*(k-v[q])+f[v[q]];}
   for(let k=0;k<n;k++)write(k,out[k]);
  };
  for(let j=0;j<n;j++)pass(k=>d[j*n+k],(k,val)=>{d[j*n+k]=val;});
  for(let i=0;i<n;i++)pass(k=>d[k*n+i],(k,val)=>{d[k*n+i]=val;});
  for(let k=0;k<d.length;k++)d[k]=Math.min(60,Math.sqrt(d[k])*this.cell);
 }
 at(x:number,z:number){
  const px=(x-this.origin)/this.cell,pz=(z-this.origin)/this.cell;const i=Math.floor(px),j=Math.floor(pz);
  if(i<0||j<0||i>=this.n-1||j>=this.n-1)return 50;
  const fx=px-i,fz=pz-j,d=this.data,n=this.n;const a=d[j*n+i],b=d[j*n+i+1],c=d[(j+1)*n+i],e=d[(j+1)*n+i+1];
  return (a*(1-fx)+b*fx)*(1-fz)+(c*(1-fx)+e*fx)*fz;
 }
 gradient(x:number,z:number){const h=this.cell;const gx=(this.at(x+h,z)-this.at(x-h,z))/(2*h),gz=(this.at(x,z+h)-this.at(x,z-h))/(2*h);const l=Math.hypot(gx,gz)||1;return [gx/l,gz/l];}
}
export function buildDistanceField(objects:THREE.Object3D[],layerOf:(o:THREE.Object3D)=>string,extentM=275,cellM=.5){
 const field=new DistanceField(extentM,cellM);const p=new THREE.Vector3(),q=new THREE.Vector3(),r=new THREE.Vector3();
 for(const o of objects){
  if(!(o instanceof THREE.Mesh)||layerOf(o)!=='foundations')continue;
  const pos=o.geometry.getAttribute('position');const idx=o.geometry.getIndex();const count=idx?idx.count:pos.count;
  const at=(k:number,v:THREE.Vector3)=>v.fromBufferAttribute(pos,idx?idx.getX(k):k).applyMatrix4(o.matrixWorld);
  for(let k=0;k<count;k+=3){at(k,p);at(k+1,q);at(k+2,r);field.markSegment(p.x,p.z,q.x,q.z);field.markSegment(q.x,q.z,r.x,r.z);field.markSegment(r.x,r.z,p.x,p.z);if(Math.abs(p.y-q.y)<.05&&Math.abs(p.y-r.y)<.05)field.markTriangle(p.x,p.z,q.x,q.z,r.x,r.z);}
 }
 field.finish();return field;
}

// ---- model ------------------------------------------------------------------------------------------------
// A two-seat mid-engined roadster built as a lofted body: at every station along the length a closed section is
// drawn from the plan width, the belt line and the crown of the deck, with the wheel wells cut where the arches are
// and the cockpit tub sunk between the belt lines; the sections are stitched into one smooth surface and capped.
// Lamps, intakes, screen, hoops, mirrors, seats and wheels are added as parts. Dimensions follow a compact sports
// roadster: 4.38 m long, 1.80 m wide, about 1.28 m high, 2.475 m wheelbase, forward is +x, up is +y.
function smooth(ctrl:[number,number][]){ // Catmull-Rom through the control points, held flat outside them
 return (x:number)=>{const n=ctrl.length;if(x<=ctrl[0][0])return ctrl[0][1];if(x>=ctrl[n-1][0])return ctrl[n-1][1];
  let i=0;while(i<n-2&&x>ctrl[i+1][0])i++;const p0=ctrl[Math.max(0,i-1)],p1=ctrl[i],p2=ctrl[i+1],p3=ctrl[Math.min(n-1,i+2)];
  const t=(x-p1[0])/(p2[0]-p1[0]),t2=t*t,t3=t2*t;
  return .5*(2*p1[1]+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3);};
}
const AXLE=SPEC.wheelbaseM/2;
const halfWidth=smooth([[-2.2,.56],[-2.1,.78],[-2.0,.85],[-1.5,.91],[-1.24,.915],[-.6,.89],[.3,.885],[1.0,.885],[1.6,.86],[2.0,.78],[2.12,.66],[2.2,.42]]);
const beltLine=smooth([[-2.2,.70],[-2.1,.78],[-2.0,.82],[-1.6,.845],[-1.24,.845],[-.9,.82],[-.5,.80],[.3,.80],[.55,.79],[.9,.78],[1.24,.76],[1.6,.68],[2.0,.56],[2.12,.50],[2.2,.40]]);
const floorLine=smooth([[-2.2,.30],[-2.1,.20],[-2.0,.15],[-1.7,.13],[1.7,.13],[2.05,.15],[2.12,.20],[2.2,.28]]);
const TUB_FROM=.40,TUB_TO=-1.05,TUB_W=.60,TUB_FLOOR=.50;
function crown(x:number){return x>TUB_FROM?(x>2.05?.03:.05):(x<TUB_TO?.03:0);}
function archTop(x:number){for(const [xa,r,ra] of [[AXLE,.335,.355],[-AXLE,.345,.36]] as [number,number,number][]){const d=Math.abs(x-xa);if(d<ra)return r+Math.sqrt(ra*ra-d*d);}return 0;}
function flank(w:number,h:number){return w-.04*((h-.45)/.35)**2;}   // tumblehome: the side is widest at 0.45 m
function halfSection(x:number):[number,number][]{ // (lateral, height) from the bottom centre to the top centre
 const w=halfWidth(x),zs=beltLine(x),z0=floorLine(x);const za=Math.max(z0+.006,archTop(x));const wIn=Math.max(.2,w-.40);
 const pts:[number,number][]=[[0,z0],[wIn,z0],[wIn,za],[flank(w,za)-.02,za]];
 for(let k=1;k<=5;k++){const h=za+(zs-za)*k/5;pts.push([flank(w,h),h]);}
 const ws=flank(w,zs);
 if(x<TUB_FROM&&x>TUB_TO){ // the cockpit: a sill, the tub wall, the floor
  const path:[number,number][]=[[ws,zs],[ws-.05,zs+.012],[TUB_W,zs+.012],[TUB_W,TUB_FLOOR],[0,TUB_FLOOR]];
  const seg=path.slice(1).map((p,i)=>Math.hypot(p[0]-path[i][0],p[1]-path[i][1]));const total=seg.reduce((a,b)=>a+b,0);
  for(let k=1;k<=9;k++){let d=total*k/9,i=0;while(i<seg.length-1&&d>seg[i]){d-=seg[i];i++;}const t=seg[i]>0?d/seg[i]:1;pts.push([path[i][0]+(path[i+1][0]-path[i][0])*t,path[i][1]+(path[i+1][1]-path[i][1])*t]);}
 }else{ // a crowned deck: a quarter superellipse from the belt line to the centre
  const zt=zs+crown(x);for(let k=1;k<=9;k++){const th=k/9*Math.PI/2;pts.push([ws*Math.pow(Math.cos(th),.78),zs+(zt-zs)*Math.pow(Math.sin(th),.78)]);}
 }
 return pts;
}
function buildBody(material:THREE.Material){
 const stations:number[]=[];for(let x=-2.2;x<=2.2001;x+=.025)stations.push(+x.toFixed(4));
 const loops=stations.map(x=>{const half=halfSection(x);return [...half,...half.slice(1,-1).reverse().map(([l,h]):[number,number]=>[-l,h])];});
 const M=loops[0].length,n=stations.length;const pos:number[]=[];const idx:number[]=[];
 loops.forEach((loop,j)=>{for(const [l,h] of loop)pos.push(stations[j],h,l);});
 for(let j=0;j<n-1;j++)for(let i=0;i<M;i++){const i2=(i+1)%M;const A=j*M+i,B=j*M+i2,C=(j+1)*M+i2,D=(j+1)*M+i;idx.push(A,C,B,A,D,C);}
 const tail=pos.length/3;{const loop=loops[0];pos.push(stations[0],loop.reduce((s,p)=>s+p[1],0)/M,0);for(let i=0;i<M;i++)idx.push(tail,i,(i+1)%M);}
 const nose=pos.length/3;{const loop=loops[n-1];pos.push(stations[n-1],loop.reduce((s,p)=>s+p[1],0)/M,0);const o=(n-1)*M;for(let i=0;i<M;i++)idx.push(nose,o+(i+1)%M,o+i);}
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geometry.setIndex(idx);geometry.computeVertexNormals();
 return new THREE.Mesh(geometry,material);
}
function buildModel(){
 const car=new THREE.Group();
 const paint=new THREE.MeshPhysicalMaterial({color:'#7e0912',metalness:0,roughness:.28,clearcoat:1,clearcoatRoughness:.05});
 const black=new THREE.MeshStandardMaterial({color:'#141516',metalness:.2,roughness:.55});
 const glass=new THREE.MeshPhysicalMaterial({color:'#243744',metalness:0,roughness:.04,transmission:.4,transparent:true,opacity:.82,ior:1.5,side:THREE.DoubleSide});
 const lampGlass=new THREE.MeshPhysicalMaterial({color:'#1a1e22',metalness:.1,roughness:.12,clearcoat:1,clearcoatRoughness:.04});
 const rubber=new THREE.MeshStandardMaterial({color:'#111111',roughness:.92,side:THREE.DoubleSide});
 const rim=new THREE.MeshStandardMaterial({color:'#4a4d52',metalness:.85,roughness:.32,side:THREE.DoubleSide});
 const disc=new THREE.MeshStandardMaterial({color:'#8c8f93',metalness:.9,roughness:.4});
 const caliper=new THREE.MeshStandardMaterial({color:'#c8102e',metalness:.3,roughness:.5});
 const lampOn=new THREE.MeshStandardMaterial({color:'#f2f0e8',emissive:'#fff4d6',emissiveIntensity:1.2,roughness:.3});
 const tailOn=new THREE.MeshStandardMaterial({color:'#7a0a10',emissive:'#ff2a2a',emissiveIntensity:1.6,roughness:.3});
 const interior=new THREE.MeshStandardMaterial({color:'#1a1b1d',roughness:.72});
 const leather=new THREE.MeshStandardMaterial({color:'#232426',roughness:.62});
 const plate=new THREE.MeshStandardMaterial({color:'#e8e8e2',roughness:.6});
 const add=(m:THREE.Mesh,parent:THREE.Object3D=car)=>{m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;};
 const box=(w:number,h:number,d:number,mat:THREE.Material,x:number,y:number,z:number,parent?:THREE.Object3D)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.set(x,y,z);return add(m,parent);};
 const blob=(rx:number,ry:number,rz:number,mat:THREE.Material,x:number,y:number,z:number)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(1,24,16),mat);m.scale.set(rx,ry,rz);m.position.set(x,y,z);return add(m);};
 const body=add(buildBody(paint));
 // Windscreen: a raked, plan-curved pane from the cowl to the header, with pillars and rails.
 {const cols=13,gp:number[]=[];for(const [x0,y0,z0,bow] of [[.50,.83,.66,.05],[-.03,1.17,.60,.06]] as [number,number,number,number][])for(let c=0;c<cols;c++){const k=c/(cols-1)*2-1;gp.push(x0+bow*(1-k*k),y0,k*z0);}
  const gi:number[]=[];for(let c=0;c<cols-1;c++){const a=c,b=c+1,d=cols+c,e=cols+c+1;gi.push(a,b,e,a,e,d);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(gp,3));g.setIndex(gi);g.computeVertexNormals();const screen=new THREE.Mesh(g,glass);screen.castShadow=false;car.add(screen);
  for(const s of [-1,1]){const from=new THREE.Vector3(.50,.83,s*.66),to=new THREE.Vector3(-.03,1.17,s*.60);const dir=to.clone().sub(from);const pillar=new THREE.Mesh(new THREE.BoxGeometry(.05,dir.length(),.05),black);pillar.position.copy(from).add(to).multiplyScalar(.5);pillar.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize());add(pillar);}
  box(.05,.04,1.24,black,-.03,1.185,0);box(.06,.03,1.34,black,.50,.835,0);}
 // Cockpit: dash, console, seats, wheel, hoops and the fairings behind them.
 box(.40,.26,1.18,interior,.28,.70,0);box(.90,.18,.22,interior,-.32,.59,0);
 for(const s of [-1,1]){box(.50,.12,.48,leather,-.36,.56,s*.38);const back=box(.12,.46,.46,leather,-.66,.79,s*.38);back.rotation.z=.22;box(.12,.13,.26,leather,-.76,1.04,s*.38);
  const hoop=new THREE.Mesh(new THREE.TorusGeometry(.20,.024,10,24,Math.PI),black);hoop.position.set(-1.0,.855,s*.40);hoop.rotation.y=Math.PI/2;add(hoop);
  blob(.40,.13,.25,paint,-1.36,.86,s*.40);}
 {const wheel=new THREE.Mesh(new THREE.TorusGeometry(.165,.02,10,28),black);wheel.position.set(.05,.92,.38);wheel.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(Math.cos(.45),Math.sin(.45),0));add(wheel);}
 // Mirrors and the side intakes ahead of the rear wheels.
 for(const s of [-1,1]){box(.16,.10,.20,paint,.40,.88,s*1.0);box(.04,.06,.12,black,.40,.84,s*.92);box(.44,.24,.07,black,-.72,.60,s*.86);}
 // Front: lamps on the fender tops with four-point lights, intakes and the lip.
 for(const s of [-1,1]){blob(.13,.10,.21,lampGlass,2.02,.60,s*.58);for(const [dy,dz] of [[.035,.035],[.035,-.035],[-.035,.035],[-.035,-.035]])blob(.02,.02,.02,lampOn,2.135,.60+dy,s*.58+dz);box(.10,.24,.40,black,2.12,.36,s*.62);}
 box(.10,.20,.70,black,2.15,.30,0);box(.12,.03,1.62,black,2.14,.14,0);
 // Rear: light bar, lamps, spoiler lip, plate, diffuser and centre exhausts.
 box(.06,.075,1.46,black,-2.17,.86,0);box(.02,.015,1.30,tailOn,-2.20,.86,0);for(const s of [-1,1]){box(.03,.05,.30,tailOn,-2.20,.86,s*.62);}
 box(.20,.03,1.32,paint,-1.90,.868,0);box(.02,.11,.52,plate,-2.21,.55,0);box(.15,.16,1.40,black,-2.15,.30,0);
 for(const s of [-1,1]){const pipe=new THREE.Mesh(new THREE.CylinderGeometry(.045,.045,.16,14),black);pipe.rotation.z=Math.PI/2;pipe.position.set(-2.23,.32,s*.11);add(pipe);}
 // Wheels: 20 inch, rounded tyres, dished rims with five twin spokes, red calipers; the front pair steers, all spin.
 const wheels:{hub:THREE.Group;spin:THREE.Group}[]=[];
 const makeWheel=(x:number,z:number,radius:number,width:number)=>{
  const g=new THREE.Group();g.position.set(x,radius,z);const spin=new THREE.Group();g.add(spin);wheels.push({hub:g,spin});
  const lathe=new THREE.Group();lathe.rotation.x=Math.sign(z)*Math.PI/2;spin.add(lathe);const hw=width/2,Rr=.254;
  const tyre=new THREE.Mesh(new THREE.LatheGeometry([[Rr,-hw+.01],[radius-.04,-hw],[radius,-hw+.045],[radius,hw-.045],[radius-.04,hw],[Rr,hw-.01]].map(([r,a])=>new THREE.Vector2(r,a)),36),rubber);add(tyre,lathe);
  const rimMesh=new THREE.Mesh(new THREE.LatheGeometry([[Rr-.006,-hw*.9],[Rr,-hw*.9],[Rr,hw*.9],[Rr-.012,hw*.9],[.20,hw*.55],[.07,hw*.5],[0,hw*.5]].map(([r,a])=>new THREE.Vector2(r,a)),36),rim);add(rimMesh,lathe);
  for(let k=0;k<5;k++){const a=k*Math.PI*2/5;for(const o of [-1,1]){const spoke=new THREE.Mesh(new THREE.BoxGeometry(.028,.02,.20),rim);spoke.position.set(Math.sin(a)*.155+Math.cos(a)*o*.02,hw*.56,Math.cos(a)*.155-Math.sin(a)*o*.02);spoke.rotation.y=a;add(spoke,lathe);}}
  const brake=new THREE.Mesh(new THREE.CylinderGeometry(.19,.19,.03,24),disc);brake.rotation.x=Math.PI/2;add(brake,g);
  box(.06,.16,.05,caliper,.14,0,Math.sign(z)*.05,g);
  car.add(g);return g;
 };
 makeWheel(AXLE,SPEC.trackFrontM/2,.335,.235);makeWheel(AXLE,-SPEC.trackFrontM/2,.335,.235);makeWheel(-AXLE,SPEC.trackRearM/2,.345,.265);makeWheel(-AXLE,-SPEC.trackRearM/2,.345,.265);
 return {car,wheels,body,tailOn};
}

// ---- engine sound ---------------------------------------------------------------------------------------
class EngineSound{
 private ctx:AudioContext|null=null;private osc:OscillatorNode[]=[];private gain:GainNode|null=null;private filter:BiquadFilterNode|null=null;
 start(){if(this.ctx)return;try{const ctx=new AudioContext();this.ctx=ctx;this.gain=ctx.createGain();this.gain.gain.value=0;this.filter=ctx.createBiquadFilter();this.filter.type='lowpass';this.filter.frequency.value=600;this.filter.connect(this.gain);this.gain.connect(ctx.destination);
  for(const [type,detune,level] of [['sawtooth',0,.5],['square',1200,.18],['sawtooth',-1195,.3]] as [OscillatorType,number,number][]){const o=ctx.createOscillator();o.type=type;o.detune.value=detune;const g=ctx.createGain();g.gain.value=level;o.connect(g);g.connect(this.filter);o.start();this.osc.push(o);}
  if(ctx.state==='suspended')void ctx.resume();}catch{this.ctx=null;}}
 update(rpm:number,throttle:number){if(!this.ctx||!this.gain||!this.filter)return;const f=rpm/20;for(const o of this.osc)o.frequency.setTargetAtTime(f,this.ctx.currentTime,.03);this.filter.frequency.setTargetAtTime(400+throttle*1800+rpm*.12,this.ctx.currentTime,.05);this.gain.gain.setTargetAtTime(.012+throttle*.03+rpm/7800*.012,this.ctx.currentTime,.08);}
 stop(){if(!this.ctx)return;this.gain?.gain.setTargetAtTime(0,this.ctx.currentTime,.1);setTimeout(()=>{this.osc.forEach(o=>{try{o.stop()}catch{}});void this.ctx?.close();this.ctx=null;this.osc=[];},400);}
}

// ---- vehicle ----------------------------------------------------------------------------------------------
export function createCar(scene:THREE.Scene,host:HTMLElement,camera:THREE.PerspectiveCamera,groundAt:(x:number,z:number)=>number,field:DistanceField|null){
 const {car,wheels,body}=buildModel();scene.add(car);
 const sound=new EngineSound();
 const hud=document.createElement('div');hud.className='car-hud';hud.setAttribute('aria-live','off');
 hud.innerHTML='<div class="car-speed"><b>0</b><small>km/h</small></div><div class="car-gear"><span>N</span><i><em></em></i></div><div class="car-help">W / S throttle and brake · A / D steer · Space handbrake · C camera · R reset · drag to look</div>';
 host.appendChild(hud);const speedEl=hud.querySelector('b')!,gearEl=hud.querySelector('.car-gear span')!,rpmEl=hud.querySelector('.car-gear em') as HTMLElement;
 // State in the body frame: u forward, v left, r yaw rate (positive turns left). Heading: forward=(-sin yaw, -cos yaw).
 const s={x:0,z:0,yaw:0,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,ay:0,throttle:0,rpm:SPEC.idleRpm,slip:0,pitch:0,roll:0};
 let trace:Record<string,number>={};
 let cameraMode:'chase'|'bonnet'='chase';let lookYaw=0;let hudTimer=0;let reverseHold=0;let shiftTimer=0;
 const camPos=new THREE.Vector3(),camLook=new THREE.Vector3();
 const {massKg:m,wheelbaseM:L,cgToFrontM:a,cgHeightM:h,yawInertia:Iz,wheelRadiusM:Rw}=SPEC;const b=L-a;
 function placeAt(x:number,z:number,yaw:number){Object.assign(s,{x,z,yaw,u:0,v:0,r:0,steer:0,gear:1,reverse:false,ax:0,ay:0,rpm:SPEC.idleRpm});lookYaw=0;hud.style.display='';syncCamera(0,true);}
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
  collide();
 }
 function collide(){
  // Four corners against the building distance field; the world edge is a soft wall.
  const [fx,fz]=forward(),[lx,lz]=left();const corners:[number,number][]=[[2.1,.85],[2.1,-.85],[-2.1,.85],[-2.1,-.85]];
  for(const [cx,cy] of corners){
   const px=s.x+fx*cx+lx*cy,pz=s.z+fz*cx+lz*cy;let d=field?field.at(px,pz):50;let nx=0,nz=0;
   const radial=Math.hypot(px,pz);if(radial>262){d=Math.min(d,262-radial+.3);nx=-px/radial;nz=-pz/radial;}
   if(d>=.45)continue;
   if(!(nx||nz)){[nx,nz]=field!.gradient(px,pz);}
   const wx0=fx*s.u+lx*s.v,wz0=fz*s.u+lz*s.v;const speed0=Math.hypot(wx0,wz0);
   if(!(nx||nz)){if(speed0>.01){nx=-wx0/speed0;nz=-wz0/speed0;}else{nx=-fx;nz=-fz;}}
   const push=.45-d;s.x+=nx*push;s.z+=nz*push;
   const wx=fx*s.u+lx*s.v,wz=fz*s.u+lz*s.v;const into=wx*nx+wz*nz;
   // Impact: the normal component of the velocity is taken away with a little rebound, the tangential one
   // scrubbed by the contact; the corner that hit turns the body a little, then yaw is damped.
   if(into<0){const kx=wx-nx*into*1.2,kz=wz-nz*into*1.2;const speedLoss=.75;s.u=(kx*fx+kz*fz)*speedLoss;s.v=(kx*lx+kz*lz)*speedLoss;s.r+=-(cy*into)*.05;s.r*=.7;s.ax=Math.max(s.ax,-6);}
  }
 }
 function lerpAngle(a:number,b:number,k:number){let d=b-a;while(d>Math.PI)d-=2*Math.PI;while(d<-Math.PI)d+=2*Math.PI;return a+d*k;}
 let camYaw=0,camDist=4.9;
 // How far the camera may trail the car before it stands inside a wall: the trailing ray is marched through the
 // distance field and stops at the last free point.
 function freeDistance(bx:number,bz:number,want:number){if(!field)return want;let d=2.4;for(let t=2.4;t<=want;t+=.25){if(field.at(s.x-bx*t,s.z-bz*t)<.7)break;d=t;}return d;}
 function syncCamera(dt:number,snap=false){
  const speed=Math.abs(s.u);const y=groundAt(s.x,s.z);const [fx,fz]=forward();
  if(cameraMode==='bonnet'){const ly=s.yaw+lookYaw;camera.position.set(s.x+fx*.25,y+1.1+s.pitch*.3,s.z+fz*.25);camera.up.set(0,1,0);camera.lookAt(s.x-Math.sin(ly)*30,y+1.02,s.z-Math.cos(ly)*30);camera.fov=64+Math.min(10,speed*.15);camera.updateProjectionMatrix();return;}
  // Chase camera in the arcade-simulation idiom: level at all times, trailing a heading that lags the car's and
  // leans toward the velocity direction in a slide (so a drift is seen sideways) and a little into a turn, sitting
  // further back with speed, swinging to the front when reversing, and easing in when a wall stands behind it
  // while keeping its downward angle, so nothing pitches or rolls with the steering. Every smoothing is time based.
  const reversing=s.u<-.8;const ahead=reversing?-1:1;
  const slide=Math.atan2(s.v,Math.max(1.5,speed))*ahead*Math.min(1,speed/6);
  const lead=Math.max(-.22,Math.min(.22,s.r*.16))*ahead;
  const targetYaw=s.yaw+(reversing?Math.PI:0)+slide*.55+lead+lookYaw;
  camYaw=snap?targetYaw:lerpAngle(camYaw,targetYaw,1-Math.exp(-dt*4.2));
  const bx=-Math.sin(camYaw),bz=-Math.cos(camYaw);
  const allowed=freeDistance(bx,bz,7.6+speed*.06);
  camDist=snap?allowed:camDist+(allowed-camDist)*(1-Math.exp(-dt*(allowed<camDist?9:2.5)));
  // The camera is placed from its smoothed heading and distance, so when it swings round the car (a reverse, a
  // spin) it travels the arc and never cuts across the body; only its height and its aim are eased separately.
  // A fixed downward angle of nine degrees to the aim point whatever the geometry: the height follows the
  // horizontal distance to the aim, so neither speed, a wall behind, a slide nor a spin changes the pitch.
  const look=new THREE.Vector3(s.x+fx*1.6*ahead,y+.9,s.z+fz*1.6*ahead);
  if(snap)camLook.copy(look);else camLook.lerp(look,1-Math.exp(-dt*14));
  camPos.set(s.x-bx*camDist,0,s.z-bz*camDist);
  camPos.y=camLook.y+.158*Math.max(3,Math.hypot(camPos.x-camLook.x,camPos.z-camLook.z));
  camera.position.copy(camPos);camera.up.set(0,1,0);camera.lookAt(camLook);
  camera.fov=58+Math.min(12,speed*.25);camera.updateProjectionMatrix();
 }
 function pose(dt:number){
  const [fx,fz]=forward(),[lx,lz]=left();
  const yc=groundAt(s.x,s.z),yf=groundAt(s.x+fx*1.2,s.z+fz*1.2),yr=groundAt(s.x-fx*1.2,s.z-fz*1.2),yl=groundAt(s.x+lx*.8,s.z+lz*.8),yrt=groundAt(s.x-lx*.8,s.z-lz*.8);
  const groundPitch=Math.atan2(yf-yr,2.4),groundRoll=Math.atan2(yl-yrt,1.6);
  const dynPitch=Math.max(-.07,Math.min(.07,s.ax*.0065)),dynRoll=Math.max(-.08,Math.min(.08,s.ay*.011));   // squat under power, dive on the brakes, lean out of the bend; impacts do not tip the body
  s.pitch+=(groundPitch+dynPitch-s.pitch)*Math.min(1,dt*8);s.roll+=(groundRoll+dynRoll-s.roll)*Math.min(1,dt*8);
  car.position.set(s.x,yc+.02,s.z);car.rotation.set(0,s.yaw+Math.PI/2,0);car.rotateZ(s.pitch);car.rotateX(s.roll);
  const spin=s.u/Rw*dt;
  wheels.forEach((w,i)=>{w.spin.rotation.z-=spin;if(i<2)w.hub.rotation.y=s.steer;});
 }
 function update(dt:number,input:CarInput){
  const sub=4,hdt=dt/sub;for(let k=0;k<sub;k++)step(hdt,input);
  pose(dt);syncCamera(dt);
  if(input.throttle>0||input.brake>0)sound.start();sound.update(s.rpm,s.throttle);
  hudTimer+=dt;if(hudTimer>.1){hudTimer=0;const kmh=Math.round(Math.abs(s.u)*3.6);speedEl.textContent=String(kmh);gearEl.textContent=s.reverse?'R':(Math.abs(s.u)<.3&&s.throttle===0?'N':String(s.gear));rpmEl.style.width=Math.round((s.rpm-SPEC.idleRpm)/(SPEC.limiterRpm-SPEC.idleRpm)*100)+'%';rpmEl.classList.toggle('is-hot',s.rpm>7000);}
 }
 function telemetry():CarTelemetry{return {speedKmh:Math.abs(s.u)*3.6,gear:s.reverse?'R':String(s.gear),rpm:s.rpm,throttle:s.throttle,slip:s.slip};}
 return {group:car,body,update,placeAt,telemetry,
  look:(deltaYaw:number)=>{lookYaw=Math.max(-2.6,Math.min(2.6,lookYaw+deltaYaw));},
  relax:(dt:number)=>{lookYaw*=Math.max(0,1-dt*2.2);},
  toggleCamera:()=>{cameraMode=cameraMode==='chase'?'bonnet':'chase';syncCamera(0,true);},
  position:()=>[s.x,s.z,s.yaw] as [number,number,number],
  debug:()=>({...trace}),
  stop:()=>{sound.stop();hud.style.display='none';},
  dispose:()=>{sound.stop();hud.remove();scene.remove(car);car.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(mm=>mm.dispose());}});}};
}
export type Car=ReturnType<typeof createCar>;
