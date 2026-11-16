// Car sound: a flat-six engine synthesised sample by sample in an AudioWorklet, and the sounds of the car meeting the
// world around it (tyre squeal, road roar, wind, wall scrapes, impacts, landings, kerb thumps), all made from shaped
// noise and oscillators so nothing is downloaded. Nothing here is location specific.
//
// The engine is a model of what one hears from a car's exhaust rather than a stack of oscillators: each cylinder
// firing is a pressure pulse (a short spike with combustion noise in it, larger with throttle, a little different for
// each of the six cylinders, which gives the half-order burble of a boxer), the pulses run through two delay-line combs
// standing in for the headers and the tail pipe, two muffler resonances and a throttle-opened low-pass, then a soft
// clipper. A lifted throttle at high revs leaves a short window of afterfire pops; the limiter cuts the ignition in
// bursts; a gear change cuts it for a moment.

const ENGINE_WORKLET = /* js */`
const TAU=Math.PI*2;
function bandpass(f,q,sr){const w=TAU*f/sr,al=Math.sin(w)/(2*q),a0=1+al;return [al/a0,0,-al/a0,-2*Math.cos(w)/a0,(1-al)/a0];}
class FlatSix extends AudioWorkletProcessor{
 static get parameterDescriptors(){return [
  {name:'rpm',defaultValue:900,minValue:0,maxValue:12000,automationRate:'k-rate'},
  {name:'throttle',defaultValue:0,minValue:0,maxValue:1,automationRate:'k-rate'},
  {name:'limiter',defaultValue:7800,minValue:1000,maxValue:12000,automationRate:'k-rate'},
 ];}
 constructor(){
  super();const sr=sampleRate;this.sr=sr;
  this.phase=0;this.k=0;this.rpm=900;this.load=0;this.lastThr=0;
  this.cyl=[1,.9,1.07,.95,1.04,.92];
  this.pulse=0;this.pulseDecay=Math.exp(-1/(sr*.0011));this.env=0;
  this.popEnv=0;this.popDecay=Math.exp(-1/(sr*.006));this.popWindow=0;
  this.cut=0;this.limitCut=0;
  this.d1=new Float32Array(8192);this.d2=new Float32Array(2048);this.i1=0;this.i2=0;this.lp1=0;this.lp2=0;
  this.D1=Math.round(sr*.0071);this.D2=Math.round(sr*.0023);
  this.r1=bandpass(92,1.4,sr);this.s1=[0,0,0,0];this.r2=bandpass(265,2.2,sr);this.s2=[0,0,0,0];
  this.ri=bandpass(1200,1.1,sr);this.si=[0,0,0,0];
  this.m1=0;this.m2=0;this.dcx=0;this.dcy=0;this.mech=0;
  this.port.onmessage=e=>{const m=e.data||{};if(m.cut)this.cut=Math.max(this.cut,m.cut*sr);if(m.pops)this.popWindow=Math.max(this.popWindow,m.pops*sr);};
 }
 biquad(c,s,x){const y=c[0]*x+c[1]*s[0]+c[2]*s[1]-c[3]*s[2]-c[4]*s[3];s[1]=s[0];s[0]=x;s[3]=s[2];s[2]=y;return y;}
 fire(){
  const k=this.k++%6,idle=Math.max(0,1-(this.rpm-900)/1600);
  let a=this.cyl[k]*(.32+.68*this.load)*(1+(Math.random()-.5)*(.1+.25*idle));
  const cutting=this.cut>0||this.limitCut>0;
  if(cutting)a*=.06;
  // Unburnt fuel lighting in the hot exhaust: pops after a lift, and on a cut cylinder at the limiter.
  if((this.popWindow>0&&Math.random()<.16)||(cutting&&this.load>.5&&Math.random()<.22))this.popEnv=Math.max(this.popEnv,.6+Math.random()*.9);
  this.pulse+=a;
 }
 process(inputs,outputs,params){
  const out=outputs[0]&&outputs[0][0];if(!out)return true;const n=out.length,sr=this.sr;
  const rpmT=params.rpm[0],thrT=params.throttle[0],limit=params.limiter[0];
  // throttle lift at high revs opens the afterfire window
  if(this.lastThr>.55&&thrT<.08&&this.rpm>3600)this.popWindow=Math.max(this.popWindow,(.45+.5*Math.min(1,(this.rpm-3600)/3500))*sr);
  this.lastThr=thrT;
  // limiter: the ignition is cut in bursts while the revs sit on it under throttle
  if(this.rpm>limit-120&&thrT>.5&&this.limitCut<=0&&Math.random()<.5)this.limitCut=sr*(.025+Math.random()*.03);
  // muffler low-pass opens with throttle and revs; the intake band climbs with revs
  const fc=650+2900*this.load+this.rpm*.32,a=1-Math.exp(-TAU*fc/sr);
  this.ri=bandpass(700+this.rpm*.28,1.1,sr);
  const intake=.22*this.load*Math.min(1,this.rpm/7000),mechLevel=.012+.03*this.rpm/8000;
  const drive=1.5+1.6*this.load,norm=1/Math.tanh(drive);
  for(let i=0;i<n;i++){
   this.rpm+=(rpmT-this.rpm)*.0016;this.load+=(thrT-this.load)*.0009;
   if(this.cut>0)this.cut--;if(this.limitCut>0)this.limitCut--;if(this.popWindow>0)this.popWindow--;
   // three firings per crank revolution
   this.phase+=this.rpm/60/sr*3;
   if(this.phase>=1){this.phase-=1+(Math.random()-.5)*.03*Math.max(0,1-(this.rpm-900)/1800);this.fire();}
   const noise=Math.random()*2-1;
   const roughness=.22+.38*this.load;
   let x=this.pulse*(1-roughness+roughness*noise);this.pulse*=this.pulseDecay;
   this.env+=(Math.abs(x)-this.env)*.004;
   // afterfire: a short noisy bang
   let pop=0;if(this.popEnv>.001){pop=this.popEnv*(Math.random()*2-1);this.popEnv*=this.popDecay;}
   x+=pop*.7;
   // headers and tail pipe: two combs with damped feedback (an open pipe end inverts the reflection)
   let j=this.i2-this.D2;if(j<0)j+=this.d2.length;this.lp2+=(this.d2[j]-this.lp2)*.45;const h=x+.38*this.lp2;this.d2[this.i2]=h;this.i2=(this.i2+1)%this.d2.length;
   j=this.i1-this.D1;if(j<0)j+=this.d1.length;this.lp1+=(this.d1[j]-this.lp1)*.3;const p=h-.6*this.lp1;this.d1[this.i1]=p;this.i1=(this.i1+1)%this.d1.length;
   let y=.55*p+1.1*this.biquad(this.r1,this.s1,p)+.5*this.biquad(this.r2,this.s2,p);
   this.m1+=(y-this.m1)*a;this.m2+=(this.m1-this.m2)*a;y=this.m2+pop*.25;
   // induction roar and valvetrain, both pulsed by the firings
   y+=intake*this.biquad(this.ri,this.si,noise*(.35+3*this.env));
   this.mech+=(noise-this.mech)*.6;y+=mechLevel*(noise-this.mech)*(.4+6*this.env);
   y=Math.tanh(y*drive)*norm;
   // DC blocker
   const dc=y-this.dcx+.996*this.dcy;this.dcx=y;this.dcy=dc;
   out[i]=dc*.42;
  }
  for(let c=1;c<outputs[0].length;c++)outputs[0][c].set(out);
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
   const noise=ctx.createBuffer(1,ctx.sampleRate*2,ctx.sampleRate);const data=noise.getChannelData(0);for(let i=0;i<data.length;i++)data[i]=Math.random()*2-1;this.noise=noise;
   const loop=(type:BiquadFilterType,freq:number,q:number):Loop=>{const src=ctx.createBufferSource();src.buffer=noise;src.loop=true;src.loopStart=Math.random();const filter=ctx.createBiquadFilter();filter.type=type;filter.frequency.value=freq;filter.Q.value=q;const gain=ctx.createGain();gain.gain.value=0;src.connect(filter);filter.connect(gain);gain.connect(master);src.start(0,Math.random()*1.5);return {src,filter,gain};};
   this.wind=loop('lowpass',420,.6);
   this.scrapeLoop=loop('bandpass',2600,.9);this.grind=loop('bandpass',640,5);
   const engineGain=ctx.createGain();engineGain.gain.value=1;engineGain.connect(master);this.engineGain=engineGain;
   const url=URL.createObjectURL(new Blob([ENGINE_WORKLET,TYRE_WORKLET],{type:'text/javascript'}));
   void ctx.audioWorklet.addModule(url).then(()=>{URL.revokeObjectURL(url);if(this.ctx!==ctx)return;
    const node=new AudioWorkletNode(ctx,'flat-six',{numberOfInputs:0,numberOfOutputs:1,outputChannelCount:[1]});node.connect(engineGain);this.engine=node;
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
