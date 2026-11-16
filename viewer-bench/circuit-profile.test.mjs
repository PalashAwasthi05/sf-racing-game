import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import test from 'node:test';
import {startPresentation} from './circuit-profile.mjs';

function connection(events,{dataLossOccurred=false,startError=false}={}){
 const target=new EventEmitter();target.commands=[];
 target.send=async(method,params)=>{
  target.commands.push({method,params});
  if(method==='Tracing.start'&&startError)throw Error('Tracing unavailable');
  if(method==='Tracing.end'){
   target.emit('protocol-event',{method:'Tracing.dataCollected',params:{value:events}});
   target.emit('protocol-event',{method:'Tracing.tracingComplete',params:{dataLossOccurred}});
  }
  return {};
 };
 return target;
}
const report={startMs:10,turns:[],crossings:[]};
const markers=[{name:'sf-circuit-start',ts:1000000},{name:'sf-circuit-end',ts:1050000}];
const updates=[0,16667,33333,50000].map(t=>({name:'Display::FrameDisplayed',ph:'I',ts:1000000+t}));
const pipeline={name:'PipelineReporter',ph:'b',ts:1010000,args:{frame_reporter:{has_main_animation:true,layer_tree_host_id:1,frame_source:1,frame_sequence:1,state:'STATE_PRESENTED_ALL'}}};

test('display mode limits trace categories and buffer while retaining all route presentation updates',async()=>{
 const target=connection([...markers,...updates,pipeline]),finish=await startPresentation(target,{mode:'display'}),result=await finish(report);
 const config=target.commands[0].params;assert.equal(config.traceConfig.traceBufferSizeInKb,8192);assert.deepEqual(config.traceConfig.includedCategories,['disabled-by-default-display.framedisplayed','blink.user_timing']);assert.equal(config.transferMode,'ReportEvents');
 assert.equal(result.mode,'display');assert.equal(result.complete,true);assert.equal(result.pipelineReporter,false);assert.equal(result.reportedSequences,null);assert.equal(result.display.updates,4);assert.equal(result.display.fps,60);assert.equal(result.traceSpanMs,50);assert.equal(target.listenerCount('protocol-event'),0);
});

test('full mode preserves the original categories,32MiB buffer and PipelineReporter counters',async()=>{
 const target=connection([...markers,...updates,pipeline]),finish=await startPresentation(target,{mode:'full'}),result=await finish(report);
 assert.deepEqual(target.commands[0].params.traceConfig.includedCategories,['benchmark','blink.user_timing']);assert.equal(target.commands[0].params.traceConfig.traceBufferSizeInKb,32768);
 assert.equal(result.pipelineReporter,true);assert.equal(result.reportedSequences,1);assert.equal(result.presentedAll,1);assert.equal(result.incomplete,0);assert.equal(result.complete,true);
});

test('missing beginning/end markers, no updates or trace loss never claim complete coverage or presentation FPS',async()=>{
 for(const [events,settings] of [[updates,{}],[[markers[0],...updates],{}],[markers,{}],[[...markers,...updates],{dataLossOccurred:true}]]){
  const target=connection(events,settings),finish=await startPresentation(target,{mode:'display'}),result=await finish(report);assert.equal(result.complete,false);
  if(!result.hasBothMarkers||!updates.every(e=>events.includes(e)))assert.equal(result.display.fps,null);
 }
 const target=connection([],{startError:true});await assert.rejects(()=>startPresentation(target,{mode:'display'}),/unavailable/);assert.equal(target.listenerCount('protocol-event'),0);
});
