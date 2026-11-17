import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {createCar,type CarInput} from './car.ts';

function drive(){
 const hud={setAttribute(){},classList:{toggle(){}},querySelector(){return this;},remove(){}};
 Object.defineProperty(globalThis,'document',{configurable:true,value:{createElement:()=>hud}});
 const car=createCar(new THREE.Scene(),{appendChild(){}} as unknown as HTMLElement,
  new THREE.PerspectiveCamera(53,16/9,.1,1000),()=>0,null,new THREE.Group(),null,10000);
 car.placeAt(0,0,0);
 const run=(frames:number,input:Partial<CarInput>)=>{const gears:string[]=[];for(let i=0;i<frames;i++){car.update(1/60,{throttle:0,brake:0,steer:0,handbrake:false,...input});gears.push(car.telemetry().gear);}return gears;};
 return {car,run};
}

test('lifting off the accelerator holds the gear; it never shifts up while coasting',(t)=>{
 const {car,run}=drive();
 run(300,{throttle:1});const before=Number(car.telemetry().gear),rpm=car.telemetry().rpm;
 const coast=run(180,{}).map(Number);
 assert.ok(before>=2,`reached gear ${before}`);
 assert.ok(Math.max(...coast)<=before,`coasting shifted up from ${before} to ${Math.max(...coast)}`);
 t.diagnostic(`gear ${before} at ${rpm.toFixed(0)} rpm; after 3 s of coasting gear ${coast.at(-1)} at ${car.telemetry().rpm.toFixed(0)} rpm`);
 car.dispose();
});

test('braking shifts down through the gears and full throttle revs out before shifting',(t)=>{
 const {car,run}=drive();
 const up=run(420,{throttle:1}).map(Number);const top=up.at(-1)!;
 for(let i=1;i<up.length;i++)assert.ok(up[i]>=up[i-1],'no downshift under full throttle');
 const down=run(240,{brake:1}).filter(g=>g!=='R'&&g!=='N').map(Number);
 assert.ok(Math.min(...down)<top,'brakes shift down');
 t.diagnostic(`top gear ${top} at ${(car.telemetry().speedKmh).toFixed(0)} km/h after braking; lowest gear under braking ${Math.min(...down)}`);
 car.dispose();
});

test('holding the handbrake on the throttle keeps the gear at the limiter instead of shifting up',(t)=>{
 const {car,run}=drive();
 run(150,{throttle:1});const before=Number(car.telemetry().gear);
 const slide=run(150,{throttle:1,handbrake:true,steer:1}).map(Number);
 assert.ok(Math.max(...slide)<=before,`shifted up from ${before} to ${Math.max(...slide)} in a handbrake slide`);
 t.diagnostic(`gear ${before} held; rpm ${car.telemetry().rpm.toFixed(0)}`);
 car.dispose();
});
