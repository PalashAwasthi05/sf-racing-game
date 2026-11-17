// Car sound: a flat-six engine synthesised sample by sample in an AudioWorklet, and the sounds of the car meeting the
// world around it (tyre squeal, road roar, wind, wall scrapes, impacts, landings, kerb thumps), all made from shaped
// noise and oscillators so nothing is downloaded. Nothing here is location specific.
//
// The engine is a model of what one hears from a car's exhaust and intake rather than a stack of oscillators. Each
// cylinder firing is a pressure pulse as long as the valve event (so it shortens as the revs climb), carrying
// band-limited gas noise, a little different for each cylinder and each firing. The two sides of the engine fire
// alternately into their own pipes (two delay-line combs, two muffler resonances, a throttle-opened low-pass) that
// differ slightly, and are heard left and right. The intake adds a tonal howl under load: each intake stroke rings two
// plenum resonances. A lifted throttle leaves some cylinders barely lit (the overrun burble) and a short window of
// afterfire pops; the limiter cuts the ignition in bursts; a gear change cuts it for a moment. A short synthetic street
// echo keeps the whole mix from sounding like it is played inside the listener's head.
// What made the earlier version sound robotic, measured from offline renders: one harmonic 30 dB above all the others
// (a whistle), perfectly periodic firing, white hiss from sample-rate noise, and a dry mono signal.

const ENGINE_WORKLET = /* js */`
const TAU=Math.PI*2;
function bandpass(f,q,sr,c){const w=TAU*f/sr,al=Math.sin(w)/(2*q),a0=1+al;c=c||[0,0,0,0,0];c[0]=al/a0;c[1]=0;c[2]=-al/a0;c[3]=-2*Math.cos(w)/a0;c[4]=(1-al)/a0;return c;}
function biquad(c,s,x){const y=c[0]*x+c[1]*s[0]+c[2]*s[1]-c[3]*s[2]-c[4]*s[3];s[1]=s[0];s[0]=x;s[3]=s[2];s[2]=y;return y;}
// One side of the engine: three cylinders into one pipe. The two sides differ a little in pipe length and resonance.
class Bank{
 constructor(sr,tail,header,f1,f2){
  this.imp=0;this.pa=0;this.pb=0;this.hp=0;this.hpx=0;this.nz=0;this.nz2=0;
  this.d1=new Float32Array(1024);this.d2=new Float32Array(512);this.i1=0;this.i2=0;this.lp1=0;this.lp2=0;
  this.D1=Math.min(1000,Math.round(sr*tail));this.D2=Math.min(500,Math.round(sr*header));
  this.c1=bandpass(f1,1.1,sr);this.s1=[0,0,0,0];this.c2=bandpass(f2,1.5,sr);this.s2=[0,0,0,0];this.m1=0;this.m2=0;
 }
}
class FlatSix extends AudioWorkletProcessor{
 static get parameterDescriptors(){return [
  {name:'rpm',defaultValue:900,minValue:0,maxValue:12000,automationRate:'k-rate'},
  {name:'throttle',defaultValue:0,minValue:0,maxValue:1,automationRate:'k-rate'},
  {name:'limiter',defaultValue:7800,minValue:1000,maxValue:12000,automationRate:'k-rate'},
 ];}
 constructor(){
  super();const sr=sampleRate;this.sr=sr;
  this.phase=0;this.k=0;this.rpm=900;this.load=0;this.lastThr=0;this.wander=0;this.idle=1;
  this.cyl=[1,.9,1.08,.94,1.05,.91];
  this.L=new Bank(sr,.0071,.0023,92,255);this.R=new Bank(sr,.00655,.00252,101,281);
  this.popEnv=0;this.popDecay=Math.exp(-1/(sr*.006));this.popWindow=0;this.popBank=this.L;
  this.cut=0;this.limitCut=0;
  this.ci1=bandpass(430,2.2,sr);this.si1=[0,0,0,0];this.ci2=bandpass(1050,1.8,sr);this.si2=[0,0,0,0];
  this.ri=bandpass(1200,1.1,sr);this.si=[0,0,0,0];
  this.an=0;this.env=0;this.mech=0;this.dc=[0,0,0,0];
  this.port.onmessage=e=>{const m=e.data||{};if(m.cut)this.cut=Math.max(this.cut,m.cut*sr);if(m.pops)this.popWindow=Math.max(this.popWindow,m.pops*sr);};
 }
 fire(){
  const k=this.k++%6,bank=k&1?this.R:this.L;
  // no two firings are alike, least of all at idle
  let a=this.cyl[k]*(.45+.55*this.load)*(1+(Math.random()-.5)*(.16+.3*this.idle));
  // overrun: with the throttle shut some cylinders barely light, which is the lumpy burble of a lifted engine
  if(this.load<.1&&this.rpm>2000&&Math.random()<.3)a*=.3;
  const cutting=this.cut>0||this.limitCut>0;
  if(cutting)a*=.06;
  // Unburnt fuel lighting in the hot exhaust: pops after a lift, and on a cut cylinder at the limiter.
  if((this.popWindow>0&&Math.random()<Math.min(.3,240/this.rpm))||(cutting&&this.load>.5&&Math.random()<.22)){this.popEnv=Math.max(this.popEnv,.6+Math.random()*.9);this.popBank=bank;}
  bank.imp+=a;
 }
 // exhaust pulse -> gas noise -> header and tail pipe -> muffler
 pipe(b,kd,kr,rough,a,pop){
  const w=Math.random()*2-1;
  b.pa=b.pa*kd+b.imp;b.pb=b.pb*kr+b.imp;b.imp=0;
  let x=b.pa-b.pb;
  b.nz+=(w-b.nz)*.32;b.nz2+=(b.nz-b.nz2)*.03;
  x*=1+rough*(b.nz-b.nz2)*3;x+=pop;
  const hp=x-b.hpx+.997*b.hp;b.hpx=x;b.hp=hp;
  // two combs with damped feedback (an open pipe end inverts the reflection)
  let j=(b.i2-b.D2)&511;b.lp2+=(b.d2[j]-b.lp2)*.5;const h=hp+.26*b.lp2;b.d2[b.i2]=h;b.i2=(b.i2+1)&511;
  j=(b.i1-b.D1)&1023;b.lp1+=(b.d1[j]-b.lp1)*.35;const p=h-.42*b.lp1;b.d1[b.i1]=p;b.i1=(b.i1+1)&1023;
  const y=.6*p+.9*biquad(b.c1,b.s1,p)+.5*biquad(b.c2,b.s2,p);
  b.m1+=(y-b.m1)*a;b.m2+=(b.m1-b.m2)*a;
  return b.m2;
 }
 process(inputs,outputs,params){
  const o=outputs[0],outL=o&&o[0];if(!outL)return true;const outR=o[1]||outL,n=outL.length,sr=this.sr;
  const rpmT=params.rpm[0],thrT=params.throttle[0],limit=params.limiter[0];
  // throttle lift at high revs opens the afterfire window
  if(this.lastThr>.55&&thrT<.08&&this.rpm>3600)this.popWindow=Math.max(this.popWindow,(.45+.5*Math.min(1,(this.rpm-3600)/3500))*sr);
  this.lastThr=thrT;
  // limiter: the ignition is cut in bursts while the revs sit on it under throttle
  if(this.rpm>limit-120&&thrT>.5&&this.limitCut<=0&&Math.random()<.5)this.limitCut=sr*(.025+Math.random()*.03);
  const rpm=this.rpm,load=this.load,idle=this.idle=Math.max(0,1-(rpm-900)/1800);
  // the revs never sit perfectly still: a slow wander, widest at idle
  this.wander+=(Math.random()*2-1-this.wander)*.02;const wob=1+this.wander*(.05+.2*idle),jit=.014+.05*idle;
  // The pulse is as long as the valve event, so it shortens as the revs climb; load sharpens its front.
  const T=40/Math.max(500,rpm),tau=.0012+.09*T,tr=Math.max(.00015,tau*(.3-.22*load));
  const kd=Math.exp(-1/(sr*tau)),kr=Math.exp(-1/(sr*tr)),rough=.1+.3*load;
  // muffler low-pass opens with throttle and revs; the intake resonances climb a little with revs
  const fc=500+2400*load+rpm*.3,a=1-Math.exp(-TAU*fc/sr);
  bandpass(430+rpm*.012,2.2,sr,this.ci1);bandpass(1050+rpm*.04,1.8,sr,this.ci2);bandpass(700+rpm*.28,1.1,sr,this.ri);
  const rise=Math.min(1,Math.max(0,(rpm-1800)/4500)),howl=1.3*Math.pow(load,1.5)*rise*rise*(3-2*rise);
  const rush=.16*load*Math.min(1,rpm/7000),mechLevel=.012+.03*rpm/8000;
  const gain=1.0*(1.25-.25*Math.min(1,rpm/3000)),drive=1.1+.8*load,norm=1/Math.tanh(drive),step=3/60/sr*wob;
  const L=this.L,R=this.R,dc=this.dc;
  for(let i=0;i<n;i++){
   this.rpm+=(rpmT-this.rpm)*.0016;this.load+=(thrT-this.load)*.0009;
   if(this.cut>0)this.cut--;if(this.limitCut>0)this.limitCut--;if(this.popWindow>0)this.popWindow--;
   // three firings per crank revolution, alternating sides
   this.phase+=this.rpm*step;
   if(this.phase>=1){this.phase-=1+(Math.random()-.5)*jit;this.fire();}
   let pop=0;if(this.popEnv>.001){pop=this.popEnv*(Math.random()*2-1);this.popEnv*=this.popDecay;}
   const yl=this.pipe(L,kd,kr,rough,a,this.popBank===L?pop*.7:0)*gain,yr=this.pipe(R,kd,kr,rough,a,this.popBank===R?pop*.7:0)*gain;
   this.env+=(Math.abs(yl+yr)-this.env)*.004;
   // induction: each intake stroke gulps air through the plenum, which rings; under load this is the howl
   const w=Math.random()*2-1,s=Math.sin(TAU*this.phase),g=s>0?s*s*s*s:0;
   this.an+=(w-this.an)*.2;
   const ind=howl*(biquad(this.ci1,this.si1,g-.19)+.8*biquad(this.ci2,this.si2,g-.19))+rush*biquad(this.ri,this.si,this.an*(.4+2.5*g));
   // valvetrain
   this.mech+=(w-this.mech)*.6;const common=ind+mechLevel*(w-this.mech)*(.4+3*this.env)+pop*.2;
   let y=Math.tanh((.8*yl+.45*yr+common)*drive)*norm,d=y-dc[0]+.996*dc[1];dc[0]=y;dc[1]=d;outL[i]=d*.42;
   if(outR!==outL){y=Math.tanh((.45*yl+.8*yr+common)*drive)*norm;d=y-dc[2]+.996*dc[3];dc[2]=y;dc[3]=d;outR[i]=d*.42;}
  }
  for(let c=2;c<o.length;c++)o[c].set(outL);
  return true;
 }
}
registerProcessor('flat-six',FlatSix);`;

// Tyres, from what makes the sound. Rolling: tread blocks slapping the road give a low roar (brown noise in a band that
// climbs with speed) with a grainy surface texture over a little high hiss. Sliding: rubber sticks and lets go many
// times a second; each release rings the tread and carcass, which have a few narrow resonances. So the squeal is noise
// bursts at the stick-slip rate through three high-Q resonators, pitched up as the slide grows, wavering in level.
const TYRE_WORKLET = /* js */`
class Tyres extends AudioWorkletProcessor{
 static get parameterDescriptors(){return [
  {name:'speed',defaultValue:0,minValue:0,maxValue:120,automationRate:'k-rate'},
  {name:'slip',defaultValue:0,minValue:0,maxValue:1,automationRate:'k-rate'},
  {name:'ground',defaultValue:1,minValue:0,maxValue:1,automationRate:'k-rate'},
 ];}
 constructor(){super();this.sr=sampleRate;this.sp=0;this.slip=0;this.ground=1;this.brown=0;this.tex=0;this.tex2=0;this.flut=0;this.flut2=0;
  this.lo=[0,0,0,0];this.mid=[0,0,0,0];this.hi=[0,0,0,0];this.r=[[0,0,0,0],[0,0,0,0],[0,0,0,0]];this.phase=0;this.pulse=0;this.pulseDecay=Math.exp(-1/(sampleRate*.0035));}
 biquad(c,s,x){const y=c[0]*x+c[1]*s[0]+c[2]*s[1]-c[3]*s[2]-c[4]*s[3];s[1]=s[0];s[0]=x;s[3]=s[2];s[2]=y;return y;}
 process(inputs,outputs,params){
  const out=outputs[0]&&outputs[0][0];if(!out)return true;const n=out.length,sr=this.sr;
  const spT=params.speed[0],slipT=params.slip[0],groundT=params.ground[0];
  this.sp+=(spT-this.sp)*.2;this.ground+=(groundT-this.ground)*.35;
  // fast onset, slower release, so a slide does not chatter on and off
  this.slip+=(slipT-this.slip)*(slipT>this.slip?.35:.08);
  const sp=this.sp,slip=this.slip,g=this.ground;
  const cLo=bandpass(55+sp*2.6,.75,sr),cMid=bandpass(240+sp*9,1.1,sr),cHi=bandpass(2200+sp*30,.7,sr);
  const roar=g*Math.min(.7,.02+.65*Math.pow(sp/32,1.15)),hiss=g*Math.min(.35,Math.pow(sp/50,1.6)*.3);
  const f0=780+330*slip+Math.min(160,sp*3.5),Q=[30,22,15],ratio=[1,1.59,2.31],lvl=[1,.42,.16];
  const res=ratio.map((k,i)=>bandpass(f0*k,Q[i],sr));
  const rate=38+95*slip,squeal=g*(slip>.01?Math.pow(slip,1.2):0);
  for(let i=0;i<n;i++){
   const w=Math.random()*2-1;
   this.brown=.996*this.brown+.06*w;
   // surface texture: slow random level changes as the road changes under the tread
   this.tex+=(w-this.tex)*.0009;this.tex2+=(this.tex-this.tex2)*.002;
   let y=roar*(1+3.5*this.tex2)*(this.biquad(cLo,this.lo,this.brown)*1.6+this.biquad(cMid,this.mid,this.brown)*.55)+hiss*this.biquad(cHi,this.hi,w)*.6;
   if(squeal>0){
    this.phase+=rate/sr;if(this.phase>=1){this.phase-=1+(Math.random()-.5)*.5;this.pulse=.5+Math.random();}
    const exc=w*(.3+this.pulse);this.pulse*=this.pulseDecay;
    this.flut+=(w-this.flut)*.0016;this.flut2+=(this.flut-this.flut2)*.003;
    const sq=lvl[0]*this.biquad(res[0],this.r[0],exc)+lvl[1]*this.biquad(res[1],this.r[1],exc)+lvl[2]*this.biquad(res[2],this.r[2],exc);
    y+=squeal*Math.max(.2,1+9*this.flut2)*sq*9;
   }
   out[i]=Math.tanh(y*1.2)*.5;
  }
  for(let c=1;c<outputs[0].length;c++)outputs[0][c].set(out);
  return true;
 }
}
registerProcessor('tyres',Tyres);`;

type Loop={src:AudioBufferSourceNode;filter:BiquadFilterNode;gain:GainNode};

// The sound coming back off the road and the house fronts: a few early reflections and a short dull tail, different in
// each ear, scaled to unit energy so the wet gain alone sets its level.
function streetEcho(ctx:BaseAudioContext){
 const sr=ctx.sampleRate,n=Math.round(sr*.42),buf=ctx.createBuffer(2,n,sr);
 for(let c=0;c<2;c++){
  const d=buf.getChannelData(c);
  for(let k=0;k<7;k++){const at=Math.round(sr*(.009+.011*k+Math.random()*.009));d[at]+=(Math.random()<.5?-1:1)*.9*Math.exp(-k*.38);}
  let lp=0,e=0;
  for(let i=Math.round(sr*.02);i<n;i++){lp+=(Math.random()*2-1-lp)*.22;d[i]+=lp*1.1*Math.exp(-i/(sr*.085));}
  for(let i=0;i<n;i++)e+=d[i]*d[i];
  const k=1/Math.sqrt(e);for(let i=0;i<n;i++)d[i]*=k;
 }
 return buf;
}

export type CarSoundState={rpm:number;throttle:number;speed:number;gear:number;limiterRpm:number;
 /** tyre slide, 0 to 1, loudest of the four */ skid:number;
 /** sliding speed (m/s) of the body along whatever it is scraping */ scrape:number;
 /** true while no wheel carries load */ airborne:boolean};

const MUTE_KEY='king-street-car-muted';

export class CarAudio{
 private ctx:AudioContext|null=null;private master:GainNode|null=null;private out:GainNode|null=null;
 private engine:AudioWorkletNode|null=null;private engineGain:GainNode|null=null;private noise:AudioBuffer|null=null;
 private tyres:AudioWorkletNode|null=null;private wind:Loop|null=null;private scrapeLoop:Loop|null=null;private grind:Loop|null=null;
 private lastGear=0;private engaged=false;private idle=0;private t=0;private lastImpact=-1;private lastBump=-1;private wasAirborne=false;private airTime=0;
 muted=false;
 // The page's first AudioContext blocks the main thread for about 0.2 s while the browser starts its audio stack.
 // Listing the audio devices starts that stack in the background, so the context costs a few ms when it is made.
 constructor(){
  void (globalThis.navigator as Navigator|undefined)?.mediaDevices?.enumerateDevices?.().catch(()=>{});
  try{this.muted=globalThis.localStorage?.getItem(MUTE_KEY)==='1';}catch{}
 }
 setMuted(on:boolean){this.muted=on;try{globalThis.localStorage?.setItem(MUTE_KEY,on?'1':'0');}catch{}
  if(this.out&&this.ctx)this.out.gain.setTargetAtTime(on?0:1,this.ctx.currentTime,.03);}
 start(){if(this.engaged&&this.ctx)return;this.engaged=true;clearTimeout(this.idle);if(this.ctx){void this.ctx.resume();return;}this.prepare();}   // called on every frame a pedal is down
 // Builds the voices ahead of the first throttle (on entering Drive) and leaves them suspended, silent, until start().
 prepare(){
  if(this.ctx||typeof AudioContext==='undefined')return;
  try{
   const ctx=new AudioContext({latencyHint:'interactive'});this.ctx=ctx;
   const limiter=ctx.createDynamicsCompressor();limiter.threshold.value=-10;limiter.knee.value=6;limiter.ratio.value=8;limiter.attack.value=.002;limiter.release.value=.18;
   const out=ctx.createGain();out.gain.value=this.muted?0:1;this.out=out;limiter.connect(out);out.connect(ctx.destination);
   const master=ctx.createGain();master.gain.value=0;master.connect(limiter);this.master=master;
   const echo=ctx.createConvolver();echo.normalize=false;echo.buffer=streetEcho(ctx);const wet=ctx.createGain();wet.gain.value=.22;master.connect(echo);echo.connect(wet);wet.connect(limiter);
   const noise=ctx.createBuffer(1,ctx.sampleRate*2,ctx.sampleRate);const data=noise.getChannelData(0);for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;this.noise=noise;
   const loop=(type:BiquadFilterType,freq:number,q:number):Loop=>{const src=ctx.createBufferSource();src.buffer=noise;src.loop=true;src.loopStart=Math.random();const filter=ctx.createBiquadFilter();filter.type=type;filter.frequency.value=freq;filter.Q.value=q;const gain=ctx.createGain();gain.gain.value=0;src.connect(filter);filter.connect(gain);gain.connect(master);src.start(0,Math.random()*1.5);return {src,filter,gain};};
   this.wind=loop('lowpass',420,.6);
   this.scrapeLoop=loop('bandpass',2600,.9);this.grind=loop('bandpass',640,5);
   const engineGain=ctx.createGain();engineGain.gain.value=1;engineGain.connect(master);this.engineGain=engineGain;
   const url=URL.createObjectURL(new Blob([ENGINE_WORKLET,TYRE_WORKLET],{type:'text/javascript'}));
   void ctx.audioWorklet.addModule(url).then(()=>{URL.revokeObjectURL(url);if(this.ctx!==ctx)return;
    const node=new AudioWorkletNode(ctx,'flat-six',{numberOfInputs:0,numberOfOutputs:1,outputChannelCount:[2]});node.connect(engineGain);this.engine=node;
    const tyres=new AudioWorkletNode(ctx,'tyres',{numberOfInputs:0,numberOfOutputs:1,outputChannelCount:[1]});tyres.connect(master);this.tyres=tyres;}).catch(()=>{});
   if(!this.engaged)void ctx.suspend();else if(ctx.state==='suspended')void ctx.resume();
  }catch{this.ctx=null;}
 }
 update(st:CarSoundState,dt:number){
  if(!this.engaged||!this.ctx||!this.master)return;const ctx=this.ctx,now=ctx.currentTime;this.t+=dt;
  this.master.gain.setTargetAtTime(.9,now,.08);
  if(this.engine){const p=this.engine.parameters;p.get('rpm')!.setTargetAtTime(st.rpm,now,.02);p.get('throttle')!.setTargetAtTime(st.throttle,now,.03);p.get('limiter')!.setValueAtTime(st.limiterRpm,now);}
  if(st.gear!==this.lastGear){if(this.lastGear>0&&st.gear>0&&this.engine)this.engine.port.postMessage(st.gear>this.lastGear?{cut:.085,pops:.12}:{cut:.04});this.lastGear=st.gear;}
  const sp=st.speed,ground=st.airborne?0:1;
  if(this.wind){this.wind.filter.frequency.setTargetAtTime(280+sp*16,now,.2);this.wind.gain.gain.setTargetAtTime(Math.min(.55,(sp/40)**2*.45),now,.2);}
  // Tyres: rolling roar with speed, squeal with the slide, silent in the air.
  if(this.tyres){const p=this.tyres.parameters;p.get('speed')!.setTargetAtTime(sp,now,.05);p.get('slip')!.setTargetAtTime(Math.min(1,st.skid),now,.02);p.get('ground')!.setTargetAtTime(ground,now,.02);}
  // Body against a wall: a hiss of grinding paint and metal, plus a rough low grind, jittered frame to frame.
  const sc=Math.min(1,Math.max(0,st.scrape-.3)/9);
  if(this.scrapeLoop){this.scrapeLoop.filter.frequency.setTargetAtTime(1800+sc*2400,now,.05);this.scrapeLoop.gain.gain.setTargetAtTime(sc>0?sc*(.35+.25*Math.random()):0,now,sc>0?.02:.06);}
  if(this.grind){this.grind.filter.frequency.setTargetAtTime(420+sc*500+Math.random()*120,now,.03);this.grind.gain.gain.setTargetAtTime(sc>0?sc*(.6+.5*Math.random()):0,now,sc>0?.02:.08);}
  // A landing after real air time.
  if(st.airborne)this.airTime+=dt;else{if(this.wasAirborne&&this.airTime>.12)this.thump(Math.min(1,.25+this.airTime*.9),55);this.airTime=0;}
  this.wasAirborne=st.airborne;
 }
 /** A collision: speed is the approach speed along the contact normal (m/s); kind 2 is a street object (post, bin). */
 impact(speed:number,kind:number){
  if(!this.engaged||!this.ctx||!this.master||!this.noise||speed<.9)return;const ctx=this.ctx,now=ctx.currentTime;
  if(now-this.lastImpact<.07)return;this.lastImpact=now;
  const s=Math.min(1,(speed-.6)/11);
  this.thump(.35+.65*s,kind===2?95:70);
  this.burst(1100+Math.random()*500,.7,.75*s+.08,.06+.28*s,0);           // crunch of the panel
  this.burst(4200,.9,.35*s,.025+.03*s,0);                                // crack
  if(kind===2){const base=360+Math.random()*180;for(const [r,l] of [[1,.22],[2.76,.12],[5.4,.07]])this.ring(base*r,l*(.4+.6*s),.35+.4*s);}
  if(s>.35){const n=4+Math.round(10*s);for(let i=0;i<n;i++)this.burst(3800+Math.random()*3800,6,.06+.1*Math.random()*s,.012+Math.random()*.03,.04+Math.random()*(.25+.4*s));} // debris and glass
 }
 /** A wheel hitting a kerb edge or the suspension reaching its stops: strength 0 to 1. */
 bump(strength:number){if(!this.engaged||!this.ctx||strength<.05)return;const now=this.ctx.currentTime;if(now-this.lastBump<.14)return;this.lastBump=now;this.thump(Math.min(1,strength),62);this.burst(320,1.4,.25*strength,.06,0);}
 private thump(level:number,freq:number){
  const ctx=this.ctx!,now=ctx.currentTime;const o=ctx.createOscillator();o.type='sine';o.frequency.setValueAtTime(freq*1.7,now);o.frequency.exponentialRampToValueAtTime(freq*.6,now+.16);
  const g=ctx.createGain();g.gain.setValueAtTime(0,now);g.gain.linearRampToValueAtTime(.95*level,now+.006);g.gain.exponentialRampToValueAtTime(.001,now+.28);
  o.connect(g);g.connect(this.master!);o.start(now);o.stop(now+.32);
 }
 private burst(freq:number,q:number,level:number,decay:number,delay:number){
  const ctx=this.ctx!,t=ctx.currentTime+delay;const src=ctx.createBufferSource();src.buffer=this.noise;const f=ctx.createBiquadFilter();f.type='bandpass';f.frequency.value=freq;f.Q.value=q;
  const g=ctx.createGain();g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(Math.max(.0011,level),t+.003);g.gain.exponentialRampToValueAtTime(.001,t+.003+decay);
  src.connect(f);f.connect(g);g.connect(this.master!);src.start(t,Math.random()*1.5);src.stop(t+decay+.05);
 }
 private ring(freq:number,level:number,decay:number){
  const ctx=this.ctx!,now=ctx.currentTime;const o=ctx.createOscillator();o.frequency.value=freq;const g=ctx.createGain();g.gain.setValueAtTime(0,now);g.gain.linearRampToValueAtTime(level,now+.003);g.gain.exponentialRampToValueAtTime(.001,now+decay);
  o.connect(g);g.connect(this.master!);o.start(now);o.stop(now+decay+.05);
 }
 // Leaving Drive fades the voices out and suspends them, so the next drive resumes them rather than building new ones.
 stop(){this.engaged=false;if(!this.ctx)return;const ctx=this.ctx;this.master?.gain.setTargetAtTime(0,ctx.currentTime,.1);
  clearTimeout(this.idle);this.idle=window.setTimeout(()=>{if(!this.engaged)void ctx.suspend();},400);}
 close(){this.stop();clearTimeout(this.idle);const ctx=this.ctx;if(!ctx)return;this.ctx=null;this.engine=null;this.tyres=null;
  setTimeout(()=>{void ctx.close();},400);}
}

export const ENGINE_WORKLET_SOURCE=ENGINE_WORKLET,TYRE_WORKLET_SOURCE=TYRE_WORKLET;
