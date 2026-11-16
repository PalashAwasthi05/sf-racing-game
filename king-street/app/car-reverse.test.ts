import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {createCar,type CarInput} from './car.ts';

// The production vehicle, with only its DOM display replaced; no renderer or audio context.
function drive(){
 const hud={setAttribute(){},classList:{toggle(){}},querySelector(){return this;},remove(){}};
 Object.defineProperty(globalThis,'document',{configurable:true,value:{createElement:()=>hud}});
 const car=createCar(new THREE.Scene(),{appendChild(){}} as unknown as HTMLElement,
  new THREE.PerspectiveCamera(53,16/9,.1,1000),()=>0,null,new THREE.Group(),null,10000);
 car.placeAt(0,0,0);
 const tick=(input:Partial<CarInput>={})=>{
  car.update(1/60,{throttle:0,brake:0,steer:0,handbrake:false,...input});
  return car.debug();
 };
 const run=(frames:number,input:Partial<CarInput>)=>{let state=car.debug();for(let i=0;i<frames;i++)state=tick(input);return state;};
 return {car,tick,run};
}

test('holding S engages reverse; reverse steering remains bounded and unwinds in either direction',(t)=>{
 for(const steer of [-1,1]){
  const {car,run,tick}=drive();
  const entry=run(150,{brake:1});
  assert.equal(car.telemetry().gear,'R');assert.ok(entry.u < -3);
  let peak=0;
  for(let i=0;i<60;i++){const state=tick({brake:1,steer});peak=Math.max(peak,Math.abs(state.v));}
  // Bounded, not zero: reversing at the 7 m/s governor on full lock the centre of mass moves sideways by geometry
  // (about 1.35 m/s) plus the rear tyres' ordinary cornering slip; the old fault grew without limit (10 m/s).
  assert.ok(peak<2.25,`reverse lateral velocity grew to ${peak.toFixed(3)} m/s`);
  const recovered=run(120,{brake:1});
  assert.ok(recovered.u < -3);
  assert.ok(Math.abs(recovered.v)<.15,`unwound lateral speed ${recovered.v}`);
  assert.ok(Math.abs(recovered.r)<.05,`unwound yaw rate ${recovered.r}`);
  t.diagnostic(`steer ${steer}: peak lateral=${peak.toFixed(3)} m/s; recovered lateral=${recovered.v.toFixed(4)} m/s, yaw=${recovered.r.toFixed(4)} rad/s`);
  car.dispose();
 }
});

test('W brakes a reversing car to rest before driving forward',()=>{
 const {car,run,tick}=drive();run(150,{brake:1});
 const before=car.debug();assert.ok(before.u < -3);
 const first=tick({throttle:1});assert.equal(car.telemetry().gear,'R');assert.ok(first.u<0);assert.ok(first.u>before.u,'W did not reduce reverse speed');
 for(let i=0;i<180;i++){
  const state=tick({throttle:1});
  if(car.telemetry().gear!=='R'){assert.ok(Math.abs(state.u)<.4,'forward engaged before reverse braking reached walking speed');break;}
 }
 assert.notEqual(car.telemetry().gear,'R');assert.ok(run(60,{throttle:1}).u>2);car.dispose();
});

test('S while moving forward brakes first and requires a held stop before reverse',()=>{
 const {car,run,tick}=drive();const start=run(150,{throttle:1});assert.ok(start.u>5);
 for(let i=0;i<12;i++){const state=tick({brake:1});assert.ok(state.u>0);assert.notEqual(car.telemetry().gear,'R');}
 let nearRest=false;
 for(let i=0;i<240;i++){
  const state=tick({brake:1});
  if(state.u>=0&&state.u<.4)nearRest=true;
  if(car.telemetry().gear==='R'){assert.ok(nearRest,'reverse bypassed the braking stop');break;}
 }
 assert.equal(car.telemetry().gear,'R');assert.ok(run(60,{brake:1}).u < -1);car.dispose();
});

test('reset placement used by KeyR clears both reverse-hold and automatic-shift timers',()=>{
 const {car,run,tick}=drive();run(18,{brake:1});car.placeAt(0,0,0);
 run(8,{brake:1});assert.notEqual(car.telemetry().gear,'R','reset retained the partial reverse hold');
 run(18,{brake:1});assert.equal(car.telemetry().gear,'R');car.placeAt(0,0,0);
 let shifted=false;
 for(let i=0;i<1200;i++){if(tick({throttle:1}).shift>0){shifted=true;break;}}
 assert.ok(shifted,'fixture never exercised an active shift timer');
 car.placeAt(10,20,.3);const reset=tick();
 assert.equal(reset.shift,0);assert.equal(reset.gear,1);assert.equal(reset.u,0);assert.equal(reset.v,0);assert.equal(reset.r,0);
 assert.equal(car.telemetry().gear,'1');car.dispose();
});
