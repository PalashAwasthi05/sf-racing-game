import test from 'node:test';
import assert from 'node:assert/strict';
import {Box3, PerspectiveCamera, Vector3} from 'three';
import {detailScreenFraction, frontageScreenFraction} from './frontage-projection.ts';

const camera = new PerspectiveCamera(90, 1, 1, 500);
const extent = (x0:number,y0:number,z0:number,x1:number,y1:number,z1:number) =>
  new Box3(new Vector3(x0,y0,z0),new Vector3(x1,y1,z1));
const fraction = (box:Box3) => frontageScreenFraction(box,camera.projectionMatrix,camera.near);

test('a prominent building across an intersection merits detail, a sliver at the edge does not', () => {
  assert.ok(fraction(extent(-24,-18,-70,24,18,-60)) > detailScreenFraction);
  assert.ok(fraction(extent(58,-18,-70,106,18,-60)) < detailScreenFraction);
  assert.equal(fraction(extent(80,-18,-70,128,18,-60)),0);
});
test('clips at the near plane when the chase camera passes a wall', () => {
  assert.ok(Math.abs(fraction(extent(-.25,-.25,-3,.25,.25,2))-.0625)<1e-9);
  assert.equal(fraction(extent(-2,-2,-3,2,2,2)),1);
  assert.equal(fraction(extent(-10,-10,.1,10,10,2)),0);
});
