import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {createCar, type CarInput} from './car.ts';

// The car heads along -z from the origin. A crest: 18 m climbing at 18 %, then the road falls away at 18 %
// (an intersection on a steep street, taken over the top).
function crest(_x: number, z: number) {const d = -z - 30; return d < 0 ? 0 : d < 18 ? .18*d : Math.max(-6, .18*18 - .18*(d - 18));}

function drive(groundAt: (x: number, z: number) => number) {
  const hud = {style: {}, setAttribute() {}, classList: {toggle() {}}, querySelector() {return this;}};
  const host = {appendChild() {}} as unknown as HTMLElement;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: {createElement: () => hud}});
  const camera = new THREE.PerspectiveCamera(53, 16/9, .1, 1000);
  const car = createCar(new THREE.Scene(), host, camera, groundAt, null, new THREE.Group(), null, 10000);
  car.placeAt(0, 0, 0);
  const tick = (input: Partial<CarInput>, dt = 1/60) => {car.update(dt, {throttle: 0, brake: 0, steer: 0, handbrake: false, ...input}); return car.debug();};
  return {car, tick};
}

function overCrest(launchMps: number) {
  const {car, tick} = drive(crest);
  car.placeAt(0, 0, 0);
  // run up to speed on the flat, then hold it with a light throttle over the crest
  let air = 0, peakGap = 0, landedLoad = 0, maxPitchDown = 0, minTouching = 4;
  for (let i=0; i<60*12; i++) {
    const [, z] = car.position();
    const speed = car.forwardSpeedMps;
    const s = tick({throttle: speed < launchMps ? 1 : 0, brake: speed > launchMps + 1 ? .3 : 0});
    if (s.air > 0) air = Math.max(air, s.air);
    peakGap = Math.max(peakGap, s.y - crest(0, z));
    if (s.air === 0 && air > 0) landedLoad = Math.max(landedLoad, s.load);
    maxPitchDown = Math.min(maxPitchDown, s.pitch);
    if (-z > 25) minTouching = Math.min(minTouching, s.wheelTouch.filter(Boolean).length);
    if (-z > 130) break;
  }
  return {air, peakGap, landedLoad, maxPitchDown, minTouching, final: car.debug()};
}

test('a crest taken fast throws the car into the air; it lands, loads its springs and settles', () => {
  const r = overCrest(26);
  assert.ok(r.air > .35, `airtime ${r.air.toFixed(2)} s`);
  assert.ok(r.peakGap > .5, `peak height above road ${r.peakGap.toFixed(2)} m`);
  assert.ok(r.landedLoad > 1.3, `landing load ${r.landedLoad.toFixed(2)} g`);
  assert.equal(r.final.air, 0, 'back on the road');
  assert.ok(Math.abs(r.final.zv) < .5, 'settled');
});

test('the same crest taken slowly only goes light: at least two wheels stay on the road throughout', () => {
  const r = overCrest(7);
  assert.ok(r.minTouching >= 2, `${r.minTouching} wheels touching`);
  assert.ok(r.air < .25, `unloaded for ${r.air.toFixed(3)} s`);
  assert.ok(r.peakGap < .1, `gap ${r.peakGap.toFixed(3)} m`);
});

test('no grip in the air: steering and throttle do not change the heading or the speed mid-flight', () => {
  const {car, tick} = drive((_x, z) => -z > 40 ? -3 : 0);  // drive off a 3 m drop
  for (let i=0; i<900 && car.debug().air === 0; i++) tick({throttle: car.forwardSpeedMps < 18 ? 1 : 0});
  let s = tick({});
  for (let i=0; i<6 && s.air < .08; i++) s = tick({});
  assert.ok(s.air > 0, 'the car left the edge');
  const [, , yaw0] = car.position(), u0 = car.forwardSpeedMps, rpm0 = s.rpm;
  for (let i=0; i<8; i++) s = tick({steer: 1, throttle: 1});
  assert.ok(s.air > 0, 'still in the air');
  const [, , yaw1] = car.position();
  assert.ok(Math.abs(yaw1 - yaw0) < .01, 'heading unchanged');
  assert.ok(Math.abs(car.forwardSpeedMps - u0) < .25, 'speed unchanged but for drag');
  assert.ok(s.rpm > rpm0 + 300, 'the free wheels let the engine rev');
});

test('a handbrake slide lays tyre marks; straight driving does not', () => {
  const {car, tick} = drive(() => 0);
  for (let i=0; i<240; i++) tick({throttle: 1});
  assert.equal(car.debug().markSegments, 0);
  for (let i=0; i<60; i++) tick({steer: .8, handbrake: true});
  const s = car.debug();
  assert.ok(s.markSegments > 20, `${s.markSegments} segments`);
  assert.ok(s.skid[1] > .3);
});

test('gravity on slopes: coasting downhill gains speed, coasting uphill stops and the car then holds without rolling back', () => {
  const coast = (grade: number) => {
    // heading -z; the road rises ahead at `grade`
    const {car, tick} = drive((_x, z) => -z * grade);
    for (let i=0; i<600 && car.forwardSpeedMps < 12; i++) tick({throttle: 1});
    const u0 = car.forwardSpeedMps; let minU = u0;
    for (let i=0; i<60*8; i++) { tick({}); minU = Math.min(minU, car.forwardSpeedMps); }
    return {u0, u1: car.forwardSpeedMps, minU};
  };
  const down = coast(-.15), flat = coast(0), up = coast(.15);
  assert.ok(down.u1 > down.u0 + 2, `downhill ${down.u0.toFixed(1)} -> ${down.u1.toFixed(1)} m/s`);
  assert.ok(flat.u1 < flat.u0 && flat.u1 > 3, `flat ${flat.u0.toFixed(1)} -> ${flat.u1.toFixed(1)} m/s`);
  assert.equal(up.u1, 0, 'uphill coast ends stopped');
  assert.ok(up.minU > -.2, `rolled back at ${up.minU.toFixed(2)} m/s`);
});

test('a 20 % grade: full throttle climbs it, and released at rest the car holds', () => {
  const {car, tick} = drive((_x, z) => -z * .2);
  for (let i=0; i<240; i++) tick({throttle: 1});
  assert.ok(car.forwardSpeedMps > 10);
  for (let i=0; i<600 && car.forwardSpeedMps > 0; i++) tick({brake: 1});
  const [, z0] = car.position();
  for (let i=0; i<300; i++) tick({});
  assert.ok(Math.abs(car.position()[1] - z0) < .01, 'held');
});

test('tyre marks are for real slides only: hard ABS braking, lane changes and full-lock turns at city speed leave none', () => {
  const run = (script: (t: number, u: number) => Partial<CarInput>, secs: number) => {
    const {car, tick} = drive(() => 0);
    for (let i=0; i<secs*60; i++) tick(script(i/60, car.forwardSpeedMps));
    return car.debug().markSegments;
  };
  assert.equal(run((t) => t < 3.5 ? {throttle: 1} : {brake: 1}, 6), 0, 'braking');
  assert.equal(run((t, u) => ({throttle: u < 14 ? 1 : 0, steer: t > 3 && t < 3.4 ? 1 : t > 3.4 && t < 3.8 ? -1 : 0}), 6), 0, 'lane change');
  assert.equal(run((t, u) => ({throttle: u < 14 ? 1 : 0, steer: t > 3 ? 1 : 0}), 7), 0, 'full lock at 50 km/h');
});

test('per-wheel tyres: wheels roll at road speed when cruising, ABS keeps them turning under full braking, the handbrake locks the rears, traction control holds wheelspin near the peak', () => {
  const {car, tick} = drive(() => 0);
  let maxLaunchSlip = 0;
  for (let i=0; i<180; i++) { const s = tick({throttle: 1}); maxLaunchSlip = Math.max(maxLaunchSlip, s.kappa[2], s.kappa[3]); }
  assert.ok(maxLaunchSlip > .02 && maxLaunchSlip < .15, `launch slip ratio ${maxLaunchSlip.toFixed(3)}`);
  for (let i=0; i<60; i++) tick({throttle: .25});
  let s = car.debug();
  for (let i=0; i<4; i++) assert.ok(Math.abs(s.omega[i] * (i < 2 ? .336 : .347) - s.u) < .4, `wheel ${i} rolls with the road`);
  // full braking from speed: every wheel still turning while the car is fast
  let minFrontOmega = Infinity;
  for (let i=0; i<40; i++) { s = tick({brake: 1}); if (s.u > 8) minFrontOmega = Math.min(minFrontOmega, s.omega[0], s.omega[1]); }
  assert.ok(minFrontOmega > 5, `ABS let a front wheel stop (${minFrontOmega.toFixed(1)} rad/s)`);
  // handbrake at speed: rears stopped, fronts rolling
  const h = drive(() => 0);
  for (let i=0; i<200; i++) h.tick({throttle: 1});
  for (let i=0; i<10; i++) s = h.tick({handbrake: true});
  assert.equal(s.omega[2], 0); assert.equal(s.omega[3], 0);
  assert.ok(s.omega[0] > 20);
});

test('with traction and stability control off, a floored throttle in a bend holds a power slide; with them on it does not', () => {
  const slide = (aidsOn: boolean) => {
    const {car, tick} = drive(() => 0);
    car.aids(aidsOn);
    let peak = 0;
    for (let i=0; i<60*8; i++) { const t = i/60, u = car.forwardSpeedMps; const s = tick({throttle: t < 4 ? (u < 12 ? 1 : 0) : 1, steer: t > 4 && t < 6 ? 1 : 0}); peak = Math.max(peak, Math.abs(s.alphaR)); }
    car.aids(true);
    return {peak, marks: car.debug().markSegments};
  };
  const on = slide(true), off = slide(false);
  assert.ok(off.peak > on.peak + .05, `rear slip on ${(on.peak*57.3).toFixed(1)}°, off ${(off.peak*57.3).toFixed(1)}°`);
  assert.ok(off.marks > on.marks, `marks on ${on.marks}, off ${off.marks}`);
});
