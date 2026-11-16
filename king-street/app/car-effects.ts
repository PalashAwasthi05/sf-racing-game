import * as THREE from 'three';

// What the car leaves on the world: tyre marks where a tyre slides, and sparks where the body grinds or strikes
// something. Both are single meshes with fixed-size buffers, drawn after the opaque scene; neither writes depth or
// touches the scene's alpha channel (the grade reads that channel), and neither casts shadows. Nothing here is
// location specific.

// ---- tyre marks -----------------------------------------------------------------------------------------------
// A ring buffer of quads laid on the drivable surface, one strip per wheel. Old marks fade over a minute and are
// overwritten once the buffer wraps.
const MARK_SEGMENTS=6000,MARK_STEP=.14,MARK_LIFT=.018,MARK_FADE_S=75;
const markVertex=/* glsl */`
attribute float strength;attribute float born;attribute vec2 tread;
uniform float now;varying float vAlpha;varying vec2 vTread;
void main(){
 float age=now-born;vAlpha=strength*(1.-smoothstep(MARK_FADE*.6,MARK_FADE,age));vTread=tread;
 gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);
}`.replace(/MARK_FADE/g,MARK_FADE_S.toFixed(1));
const markFragment=/* glsl */`
varying float vAlpha;varying vec2 vTread;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
void main(){
 // soft edges across the tyre, tread grooves along it, and a little grain so a long mark is not a flat band
 float across=vTread.x;float edge=smoothstep(0.,.18,across)*smoothstep(1.,.82,across);
 float grooves=.72+.28*smoothstep(.2,.8,abs(sin(across*18.85)));
 float grain=.8+.2*hash(floor(vec2(across*24.,vTread.y*30.)));
 float a=vAlpha*edge*grooves*grain;
 if(a<.004)discard;
 gl_FragColor=vec4(vec3(.012,.011,.01),a);
}`;

// ---- sparks ---------------------------------------------------------------------------------------------------
const SPARKS=700;
const sparkVertex=/* glsl */`
attribute float heat;attribute vec2 corner;varying float vHeat;varying vec2 vCorner;
void main(){vHeat=heat;vCorner=corner;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
const sparkFragment=/* glsl */`
varying float vHeat;varying vec2 vCorner;
void main(){
 // a hot white core cooling through yellow and orange to a dull red; brightness well above white so the bloom catches it
 float across=1.-abs(vCorner.x*2.-1.);float along=mix(.25,1.,vCorner.y);
 float shape=smoothstep(0.,.65,across)*along;
 vec3 hot=vec3(1.,.93,.75),warm=vec3(1.,.55,.14),cold=vec3(.8,.16,.03);
 vec3 c=vHeat>.5?mix(warm,hot,vHeat*2.-1.):mix(cold,warm,vHeat*2.);
 gl_FragColor=vec4(c*shape*(3.+40.*vHeat*vHeat),0.);
}`;

export type WheelTrack={x:number;z:number;touching:boolean;skid:number;halfWidth:number};
export type CarHit={x:number;y:number;z:number;nx:number;nz:number;vx:number;vz:number;vn:number;speed:number;kind:number;dt:number};

export class CarEffects{
 readonly group=new THREE.Group();
 private marks:THREE.Mesh;private markPos:THREE.BufferAttribute;private markStrength:THREE.BufferAttribute;private markBorn:THREE.BufferAttribute;private markTread:THREE.BufferAttribute;
 private markMaterial:THREE.ShaderMaterial;private markHead=0;private markCount=0;private dirtyLo=Infinity;private dirtyHi=-1;
 private trails:{on:boolean;lx:number;ly:number;lz:number;rx:number;ry:number;rz:number;x:number;z:number;a:number;dist:number}[]=[];
 private sparks:THREE.Mesh;private sparkPos:THREE.BufferAttribute;private sparkHeat:THREE.BufferAttribute;
 private p=new Float32Array(SPARKS*3);private v=new Float32Array(SPARKS*3);private life=new Float32Array(SPARKS);private age=new Float32Array(SPARKS);private live=0;
 private clock=0;private surfaceAt:(x:number,z:number)=>number;
 /** sparks spawned since the car was placed (for tests and debugging) */ sparkCount=0;
 /** tyre-mark segments laid since the car was placed */ markSegments=0;
 constructor(scene:THREE.Scene,surfaceAt:(x:number,z:number)=>number){
  this.surfaceAt=surfaceAt;this.group.name='car-effects';
  // tyre marks
  const mg=new THREE.BufferGeometry();
  this.markPos=new THREE.BufferAttribute(new Float32Array(MARK_SEGMENTS*4*3),3);this.markStrength=new THREE.BufferAttribute(new Float32Array(MARK_SEGMENTS*4),1);
  this.markBorn=new THREE.BufferAttribute(new Float32Array(MARK_SEGMENTS*4).fill(-1e6),1);this.markTread=new THREE.BufferAttribute(new Float32Array(MARK_SEGMENTS*4*2),2);
  for(const a of [this.markPos,this.markStrength,this.markBorn,this.markTread])a.setUsage(THREE.DynamicDrawUsage);
  mg.setAttribute('position',this.markPos);mg.setAttribute('strength',this.markStrength);mg.setAttribute('born',this.markBorn);mg.setAttribute('tread',this.markTread);
  const idx=new Uint32Array(MARK_SEGMENTS*6);for(let s=0;s<MARK_SEGMENTS;s++){const v=s*4,i=s*6;idx[i]=v;idx[i+1]=v+2;idx[i+2]=v+1;idx[i+3]=v+1;idx[i+4]=v+2;idx[i+5]=v+3;}
  mg.setIndex(new THREE.BufferAttribute(idx,1));mg.setDrawRange(0,0);
  this.markMaterial=new THREE.ShaderMaterial({name:'tyre-marks',vertexShader:markVertex,fragmentShader:markFragment,uniforms:{now:{value:0}},transparent:true,depthWrite:false,
   polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-4,side:THREE.DoubleSide,
   blending:THREE.CustomBlending,blendSrc:THREE.SrcAlphaFactor,blendDst:THREE.OneMinusSrcAlphaFactor,blendSrcAlpha:THREE.ZeroFactor,blendDstAlpha:THREE.OneFactor});
  this.marks=new THREE.Mesh(mg,this.markMaterial);this.marks.name='tyre-marks';this.marks.frustumCulled=false;this.marks.renderOrder=1;this.marks.castShadow=false;this.marks.receiveShadow=false;
  // sparks: four corners per spark, rebuilt each frame as camera-facing streaks along their velocity
  const sg=new THREE.BufferGeometry();
  this.sparkPos=new THREE.BufferAttribute(new Float32Array(SPARKS*4*3),3);this.sparkHeat=new THREE.BufferAttribute(new Float32Array(SPARKS*4),1);
  this.sparkPos.setUsage(THREE.DynamicDrawUsage);this.sparkHeat.setUsage(THREE.DynamicDrawUsage);
  const corner=new Float32Array(SPARKS*4*2);for(let s=0;s<SPARKS;s++)corner.set([0,0,1,0,0,1,1,1],s*8);
  sg.setAttribute('position',this.sparkPos);sg.setAttribute('heat',this.sparkHeat);sg.setAttribute('corner',new THREE.BufferAttribute(corner,2));
  const sidx=new Uint32Array(SPARKS*6);for(let s=0;s<SPARKS;s++){const v=s*4,i=s*6;sidx[i]=v;sidx[i+1]=v+1;sidx[i+2]=v+2;sidx[i+3]=v+2;sidx[i+4]=v+1;sidx[i+5]=v+3;}
  sg.setIndex(new THREE.BufferAttribute(sidx,1));sg.setDrawRange(0,0);
  const sm=new THREE.ShaderMaterial({name:'sparks',vertexShader:sparkVertex,fragmentShader:sparkFragment,transparent:true,depthWrite:false,side:THREE.DoubleSide,
   blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneFactor,blendSrcAlpha:THREE.ZeroFactor,blendDstAlpha:THREE.OneFactor});
  this.sparks=new THREE.Mesh(sg,sm);this.sparks.name='sparks';this.sparks.frustumCulled=false;this.sparks.renderOrder=2;
  this.group.add(this.marks,this.sparks);scene.add(this.group);
  for(let i=0;i<4;i++)this.trails.push({on:false,lx:0,ly:0,lz:0,rx:0,ry:0,rz:0,x:0,z:0,a:0,dist:0});
 }
 /** Breaks every tyre trail (the car was moved). Marks already on the road stay. */
 reset(){for(const t of this.trails)t.on=false;this.sparkCount=0;this.markSegments=0;}
 clear(){this.reset();this.live=0;this.markHead=0;this.markCount=0;this.markBorn.array.fill(-1e6);this.markBorn.needsUpdate=true;this.marks.geometry.setDrawRange(0,0);}
 /** Lays tyre marks under sliding wheels. rx, rz: the car's right in the world. */
 track(wheels:WheelTrack[],rx:number,rz:number){
  for(let i=0;i<wheels.length;i++){
   const w=wheels[i],t=this.trails[i],a=Math.min(.92,.25+w.skid*.85);
   if(!w.touching||w.skid<.12){t.on=false;continue;}
   const y=this.surfaceAt(w.x,w.z)+MARK_LIFT,hw=w.halfWidth;
   const lx=w.x-rx*hw,lz=w.z-rz*hw,qx=w.x+rx*hw,qz=w.z+rz*hw;
   const moved=Math.hypot(w.x-t.x,w.z-t.z);
   if(!t.on||moved>4){Object.assign(t,{on:true,lx,ly:this.surfaceAt(lx,lz)+MARK_LIFT,lz,rx:qx,ry:this.surfaceAt(qx,qz)+MARK_LIFT,rz:qz,x:w.x,z:w.z,a:a*.3});continue;}
   if(moved<MARK_STEP)continue;
   const ly=Math.max(y-.06,this.surfaceAt(lx,lz)+MARK_LIFT),ry=Math.max(y-.06,this.surfaceAt(qx,qz)+MARK_LIFT);
   this.segment(t.lx,t.ly,t.lz,t.rx,t.ry,t.rz,lx,ly,lz,qx,ry,qz,t.a,a,t.dist,t.dist+moved);
   Object.assign(t,{lx,ly,lz,rx:qx,ry,rz:qz,x:w.x,z:w.z,a,dist:t.dist+moved});
  }
 }
 private segment(ax:number,ay:number,az:number,bx:number,by:number,bz:number,cx:number,cy:number,cz:number,dx:number,dy:number,dz:number,a0:number,a1:number,d0:number,d1:number){
  const s=this.markHead,v=s*4;this.markHead=(this.markHead+1)%MARK_SEGMENTS;this.markCount=Math.min(MARK_SEGMENTS,this.markCount+1);this.markSegments++;
  const P=this.markPos.array as Float32Array,S=this.markStrength.array as Float32Array,B=this.markBorn.array as Float32Array,T=this.markTread.array as Float32Array;
  P.set([ax,ay,az,bx,by,bz,cx,cy,cz,dx,dy,dz],v*3);S.set([a0,a0,a1,a1],v);B.set([this.clock,this.clock,this.clock,this.clock],v);T.set([0,d0,1,d0,0,d1,1,d1],v*2);
  this.dirtyLo=Math.min(this.dirtyLo,s);this.dirtyHi=Math.max(this.dirtyHi,s);
 }
 /** Sparks from a contact. A hit throws a burst; a sliding contact streams them in proportion to its speed. */
 hit(h:CarHit){
  if(h.kind===1)return;
  const metal=h.kind===2?1.6:1;
  let n=0;
  if(h.vn>2)n+=Math.min(90,(h.vn-1.4)*10*metal);
  if(h.speed>3)n+=(h.speed-2.5)*48*metal*h.dt;
  n=Math.floor(n)+(Math.random()<n%1?1:0);
  const sp=Math.hypot(h.vx,h.vz)||1;
  for(let k=0;k<n;k++){
   // thrown along the car's motion past the surface, off the surface, and up; a few go far and fast
   const fast=Math.random()<.2?1.8:1,along=(.35+.6*Math.random())*fast,off=.8+2.2*Math.random(),up=1.2+3.8*Math.random()*fast;
   const vx=h.vx*along+h.nx*off+(Math.random()-.5)*2.4,vz=h.vz*along+h.nz*off+(Math.random()-.5)*2.4;
   this.spawn(h.x+(Math.random()-.5)*.25,h.y+(Math.random()-.5)*.25,h.z+(Math.random()-.5)*.25,vx,up+Math.min(3,sp*.06)*Math.random(),vz,.35+Math.random()*.7*fast);
  }
 }
 /** The underbody striking the road on a hard landing or a kerb. */
 scrapeFloor(x:number,y:number,z:number,vx:number,vz:number,strength:number){
  const n=Math.min(80,Math.floor(strength*40));
  for(let k=0;k<n;k++){const along=.4+.5*Math.random(),a=Math.random()*Math.PI*2,r=1+3*Math.random();
   this.spawn(x+(Math.random()-.5)*1.4,y+.05,z+(Math.random()-.5)*1.4,vx*along+Math.cos(a)*r,.4+2*Math.random(),vz*along+Math.sin(a)*r,.3+Math.random()*.6);}
 }
 private spawn(x:number,y:number,z:number,vx:number,vy:number,vz:number,life:number){
  let i=this.live;if(i>=SPARKS){i=Math.floor(Math.random()*SPARKS);}else this.live++;
  this.p[i*3]=x;this.p[i*3+1]=y;this.p[i*3+2]=z;this.v[i*3]=vx;this.v[i*3+1]=vy;this.v[i*3+2]=vz;this.life[i]=life;this.age[i]=0;this.sparkCount++;
 }
 update(dt:number,camera:THREE.Camera){
  this.clock+=dt;this.markMaterial.uniforms.now.value=this.clock;
  if(this.dirtyHi>=0){
   const lo=this.dirtyLo,count=this.dirtyHi-lo+1;
   for(const [attr,size] of [[this.markPos,12],[this.markStrength,4],[this.markBorn,4],[this.markTread,8]] as [THREE.BufferAttribute,number][]){attr.clearUpdateRanges();attr.addUpdateRange(lo*size,count*size);attr.needsUpdate=true;}
   this.marks.geometry.setDrawRange(0,this.markCount*6);this.dirtyLo=Infinity;this.dirtyHi=-1;
  }
  // sparks: gravity, air drag, a bounce off the road that sheds most of their speed
  const P=this.p,V=this.v,drag=Math.exp(-dt*1.6);
  for(let i=0;i<this.live;){
   this.age[i]+=dt;
   if(this.age[i]>=this.life[i]){const j=--this.live;if(i!==j){P.copyWithin(i*3,j*3,j*3+3);V.copyWithin(i*3,j*3,j*3+3);this.life[i]=this.life[j];this.age[i]=this.age[j];}continue;}
   V[i*3+1]-=9.81*dt;V[i*3]*=drag;V[i*3+1]*=drag;V[i*3+2]*=drag;
   P[i*3]+=V[i*3]*dt;P[i*3+1]+=V[i*3+1]*dt;P[i*3+2]+=V[i*3+2]*dt;
   const floor=this.surfaceAt(P[i*3],P[i*3+2])+.01;
   if(P[i*3+1]<floor){P[i*3+1]=floor;if(V[i*3+1]<0)V[i*3+1]*=-.3;V[i*3]*=.6;V[i*3+2]*=.6;this.age[i]+=.08;}
   i++;
  }
  const cam=camera.position,pos=this.sparkPos.array as Float32Array,heat=this.sparkHeat.array as Float32Array;
  for(let i=0;i<this.live;i++){
   const x=P[i*3],y=P[i*3+1],z=P[i*3+2],vx=V[i*3],vy=V[i*3+1],vz=V[i*3+2];
   // a streak as long as the distance travelled in about 1/45 s (motion blur), at least a few centimetres
   const sp=Math.hypot(vx,vy,vz)||1,len=Math.max(.06,Math.min(.8,sp/35)),tx=-vx/sp*len,ty=-vy/sp*len,tz=-vz/sp*len;
   const ex=x-cam.x,ey=y-cam.y,ez=z-cam.z,dist=Math.hypot(ex,ey,ez)||1;
   let sx=vy*ez-vz*ey,sy=vz*ex-vx*ez,sz=vx*ey-vy*ex;const sl=Math.hypot(sx,sy,sz)||1;
   // at least ~1.3 px wide at 1080p so a spark does not vanish between the temporal jitters
   const w=Math.max(.016,dist*.0024)/sl;sx*=w;sy*=w;sz*=w;
   const h=Math.max(0,1-(this.age[i]/this.life[i])**1.5),o=i*12;
   pos[o]=x+tx-sx;pos[o+1]=y+ty-sy;pos[o+2]=z+tz-sz;pos[o+3]=x+tx+sx;pos[o+4]=y+ty+sy;pos[o+5]=z+tz+sz;
   pos[o+6]=x-sx;pos[o+7]=y-sy;pos[o+8]=z-sz;pos[o+9]=x+sx;pos[o+10]=y+sy;pos[o+11]=z+sz;
   heat[i*4]=heat[i*4+1]=heat[i*4+2]=heat[i*4+3]=h;
  }
  if(this.live){this.sparkPos.clearUpdateRanges();this.sparkPos.addUpdateRange(0,this.live*12);this.sparkPos.needsUpdate=true;this.sparkHeat.clearUpdateRanges();this.sparkHeat.addUpdateRange(0,this.live*4);this.sparkHeat.needsUpdate=true;}
  this.sparks.geometry.setDrawRange(0,this.live*6);
 }
 get liveSparks(){return this.live;}
 dispose(){this.group.removeFromParent();this.marks.geometry.dispose();this.markMaterial.dispose();this.sparks.geometry.dispose();(this.sparks.material as THREE.Material).dispose();}
}
