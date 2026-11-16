import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {createCar, type CarInput} from './car.ts';

// Exercise the same vehicle and cameras as the viewer without creating a GPU or an audio context.
function drive(groundAt: (x: number, z: number) => number = () => 0) {
  const hud = {
    style: {}, setAttribute() {}, classList: {toggle() {}}, querySelector() {return this;}
  };
  const host = {appendChild() {}} as unknown as HTMLElement;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: {createElement: () => hud}});
  const camera = new THREE.PerspectiveCamera(53, 16/9, .1, 1000);
  const car = createCar(new THREE.Scene(), host, camera, groundAt, null, new THREE.Group(), null, 10000);
  car.placeAt(0, 0, 0);
  const tick = (input: Partial<CarInput>, dt = 1/60) => {
    car.update(dt, {throttle: 0, brake: 0, steer: 0, handbrake: false, ...input});
    return car.debug();
  };
  return {car, camera, tick};
}

test('full throttle and hard braking give a small, correctly signed body displacement on level ground', () => {
  const {tick} = drive();
  let peak = 0, dive = 0;
  for (let i=0; i<150; i++) peak = Math.max(peak, tick({throttle: 1}).pitch);
  for (let i=0; i<50; i++) dive = Math.min(dive, tick({brake: 1}).pitch);
  assert.ok(peak > .005 && peak < .027);
  assert.ok(dive < -.009 && dive > -.027);
});

test('real grades and camber survive the ride filter and the car settles immediately at reset', () => {
  const {car, camera, tick} = drive((x, z) => -.08*x - .15*z);
  tick({throttle: 1, steer: .6});
  car.placeAt(24, -6, 0);
  const state = tick({});
  assert.ok(Math.abs(state.pitch - Math.atan(.15)) < 1e-7);
  assert.ok(Math.abs(state.roll - Math.atan(.08)) < 1e-7);
  const start = camera.position.clone();
  for (let i=0; i<30; i++) tick({});
  assert.ok(camera.position.distanceTo(start) < 1e-6);
});

test('under left and right steering the chassis leans out of each turn and stays framed by the chase camera', () => {
  for (const sign of [1, -1]) {
    const {car, camera, tick} = drive();
    for (let i=0; i<150; i++) tick({throttle: 1});
    let lean = 0, maxOffset = 0;
    for (let i=0; i<72; i++) {
      const state = tick({steer: sign*.45, throttle: .18});
      lean = Math.max(lean, sign*state.roll);
      camera.updateMatrixWorld();
      const center = car.group.position.clone();
      center.y += .85;
      maxOffset = Math.max(maxOffset, Math.abs(center.project(camera).x));
    }
    assert.ok(lean > .009 && lean < .038);
    assert.ok(maxOffset < .002, 'the chase pivot moves sideways on the screen');
  }
});

test('a short raised strip still moves the wheels and the chassis settles after clearing it', () => {
  const height = .09;
  const {tick} = drive((_x,z) => z < -12 && z > -13 ? height : 0);
  let peakWheel = 0, peakBody = 0, finalHeight = 0;
  for (let i=0; i<210; i++) {
    const s = tick({throttle: .75});
    peakWheel = Math.max(peakWheel, ...s.wheelH);
    peakBody = Math.max(peakBody, s.y);
    finalHeight = s.y;
  }
  assert.equal(peakWheel, height);
  assert.ok(peakBody > .005 && peakBody < height);
  assert.ok(Math.abs(finalHeight) < .001);
});

test('the ride and chase view stay consistent at 30, 60 and 120 Hz', () => {
  function maneuvre(hz: number) {
    const {car, camera, tick} = drive((_x, z) => -.06*z);
    for (let i=0; i<3*hz; i++) {
      const steering = i >= 2*hz && i < 2.5*hz;
      tick({steer: steering ? .35 : 0, throttle: i < 2*hz ? 1 : .2}, 1/hz);
    }
    return {pose: car.group.position, orientation: car.group.quaternion, camera};
  }
  const base = maneuvre(60);
  for (const hz of [30,120]) {
    const other = maneuvre(hz);
    assert.ok(other.pose.distanceTo(base.pose) < .001);
    assert.ok(other.orientation.angleTo(base.orientation) < .001);
    assert.ok(other.camera.position.distanceTo(base.camera.position) < .03);
    assert.ok(other.camera.quaternion.angleTo(base.camera.quaternion) < .005);
  }
});
