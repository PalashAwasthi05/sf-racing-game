import assert from 'node:assert/strict';
import test from 'node:test';
import {lazyBlockVariants} from './lazy-block-variants.ts';

function deferred<T>() {let resolve!:(value:T)=>void, reject!:(error:Error)=>void;const promise=new Promise<T>((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
function layer() {return {visible:false,disposed:0,show(value:boolean){this.visible=value;},dispose(){this.disposed++;}};}

test('baseline fetches nothing; rapid selections share requests and only show the latest', async()=>{
  const photo=deferred<ReturnType<typeof layer>>(),splats=deferred<ReturnType<typeof layer>>();let loads=0;
  const variants=lazyBlockVariants(variant=>{loads++;return variant==='photo'?photo.promise:splats.promise;});
  assert.equal(await variants.set('baseline'),true);assert.equal(loads,0);
  const first=variants.set('photo'),second=variants.set('splats'),last=variants.set('photo');
  assert.equal(loads,2);
  const p=layer(),s=layer();splats.resolve(s);photo.resolve(p);
  assert.deepEqual(await Promise.all([first,second,last]),[false,false,true]);
  assert.equal(p.visible,true);assert.equal(s.visible,false);
  await variants.set('baseline');assert.equal(p.visible,false);
  await variants.set('splats');assert.equal(s.visible,true);assert.equal(loads,2);
  variants.dispose();assert.equal(p.disposed,1);assert.equal(s.disposed,1);
});

test('returning to baseline cancels pending visibility and failed selection can retry', async()=>{
  const pending=deferred<ReturnType<typeof layer>>();let loads=0;
  const variants=lazyBlockVariants(()=>{loads++;return loads===1?pending.promise:Promise.resolve(layer());});
  const first=variants.set('photo');await variants.set('baseline');pending.reject(new Error('network'));
  assert.equal(await first,false);
  assert.equal(await variants.set('photo'),true);assert.equal(loads,2);
  variants.dispose();
});

test('current failure is reported and disposal aborts requests and releases late arrivals once', async()=>{
  const failed=lazyBlockVariants(async()=>{throw new Error('network');});
  await assert.rejects(failed.set('photo'),/network/);assert.equal(await failed.set('baseline'),true);failed.dispose();
  const pending=deferred<ReturnType<typeof layer>>();let signal:AbortSignal|undefined;
  const variants=lazyBlockVariants((_variant,requestSignal)=>{signal=requestSignal;return pending.promise;});
  const selected=variants.set('splats');variants.dispose();variants.dispose();assert.equal(signal?.aborted,true);
  const late=layer();pending.resolve(late);assert.equal(await selected,false);assert.equal(late.visible,false);assert.equal(late.disposed,1);
  assert.equal(await variants.set('photo'),false);
});
