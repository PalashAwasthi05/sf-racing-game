// Separate from timing runs: inspect submitted canvas frames and the browser compositor.
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';

// A one-to-three-frame dark surface that returns to its former appearance can
// be invisible to a whole-screen average. Keep the surrounding camera motion
// small and require a coherent dark patch rather than a single shadow pixel.
export function createPatchTracker({browser=false}={}){
  const cols=24,rows=12,n=cols*rows;
  const history=[],stats={samples:0,confirmed:0,worstDrop:0,examples:[]};
  function sample(data,w,h,phase,stamp){
    const values=new Float32Array(n),dark=new Float32Array(n),counts=new Uint16Array(n);
    for(let y=Math.floor(h*.14);y<h*.86;y++)for(let x=Math.floor(w*.02);x<w*.98;x++){
      const col=Math.min(cols-1,Math.max(0,Math.floor((x/w-.02)*cols/.96)));
      const row=Math.min(rows-1,Math.max(0,Math.floor((y/h-.14)*rows/.72)));
      const j=row*cols+col,i=4*(y*w+x);
      values[j]+=(data[i]+2*data[i+1]+data[i+2])/4;
      dark[j]+=Math.max(data[i],data[i+1],data[i+2])<28?1:0;counts[j]++;
    }
    for(let j=0;j<n;j++){values[j]/=Math.max(1,counts[j]);dark[j]/=Math.max(1,counts[j]);}
    stats.samples++;let strongest=null;
    for(let duration=1;duration<=Math.min(3,history.length-1);duration++){
      const before=history.at(-duration-1),first=history.at(-duration);
      if(before.phase!==phase||stamp-before.stamp>280)continue;
      let motion=0;for(let j=0;j<n;j++)motion+=Math.abs(values[j]-before.values[j]);
      if(motion/n>10)continue;
      for(let j=0;j<n;j++){
        const bright=Math.min(before.values[j],values[j]),drop=bright-first.values[j];
        if(bright<47||drop<35||first.values[j]>bright*.38||first.dark[j]<.66||
           Math.abs(before.values[j]-values[j])>Math.max(17,.22*bright))continue;
        if(history.slice(-duration).some(frame=>frame.values[j]>bright*.48))continue;
        if(!strongest||drop>strongest.drop)strongest={drop:Math.round(drop),col:j%cols,row:Math.floor(j/cols),duration,phase,stamp,frame:first.frame};
      }
    }
    if(strongest){stats.confirmed++;stats.worstDrop=Math.max(stats.worstDrop,strongest.drop);if(stats.examples.length<20)stats.examples.push(strongest);}
    history.push({values,dark,stamp,phase,frame:stats.samples});if(history.length>4)history.shift();
    return strongest;
  }
  // The chase camera keeps the roof and rear deck in this central rectangle.
  // A whole vehicle can disappear into black for longer than the three-frame
  // region test. Count the full interval, including recovery, on both surfaces.
  const vehicle={samples:0,darkFrames:0,episodes:0,longestFrames:0,examples:[]};
  let darkRun=0;
  function sampleVehicle(data,w,h,phase,stamp){
    if(phase!=='drive')return false;
    let nearBlack=0,total=0;
    for(let y=Math.ceil(h*.55);y<h*.64;y++)for(let x=Math.ceil(w*.47);x<w*.53;x++){
      const i=(y*w+x)*4;
      nearBlack+=Number(Math.max(data[i],data[i+1],data[i+2])<=5);total++;
    }
    vehicle.samples++;
    if(nearBlack<.8*total){darkRun=0;return false;}
    vehicle.darkFrames++;darkRun++;vehicle.longestFrames=Math.max(vehicle.longestFrames,darkRun);
    if(darkRun===1){vehicle.episodes++;if(vehicle.examples.length<8)vehicle.examples.push({stamp,sample:vehicle.samples});}
    return darkRun===1&&vehicle.episodes<=4;
  }
  const tracker={stats,sample,vehicle,sampleVehicle};
  if(browser){window.__streetPatchTracker=tracker;return;}
  return tracker;
}

export async function watchDriveFrames(page,cdp,decode,output){
  const presented={samples:0,black:0,abruptDrops:0,worstMean:255,examples:[],phases:{}};
  const patches=createPatchTracker(),pictures=[],timeline=[],pending=[];
  presented.patches=patches.stats;presented.vehicle=patches.vehicle;presented.swaps=[];
  const comparisonPoses=[];
  if(process.env.STREAM_FLASH_BASELINE){
    const prior=JSON.parse(readFileSync(process.env.STREAM_FLASH_BASELINE,'utf8'));
    for(const [index,event] of prior.flashReview.presented.vehicle.examples.entries()){
      const after=prior.samples.findIndex(s=>s.epochMs>=event.stamp);
      if(after<1)continue;
      const a=prior.samples[after-1],b=prior.samples[after],t=(event.stamp-a.epochMs)/(b.epochMs-a.epochMs);
      comparisonPoses.push({index:index+1,e:a.positionEN[0]+t*(b.positionEN[0]-a.positionEN[0]),
        n:a.positionEN[1]+t*(b.positionEN[1]-a.positionEN[1]),yaw:a.yaw});
    }
  }
  presented.matchedViews=[];
  const saveSwaps=()=>{
    for(let i=pending.length-1;i>=0;i--){
      const swap=pending[i],after=timeline.find(frame=>frame.stamp>=swap.stamp);
      const before=timeline.findLast(frame=>frame.stamp<swap.stamp);
      if(!after||!before)continue;
      if(swap.type==='comparison'){
        const file=path.basename(output,'.json')+'-matched-'+swap.index+'.jpg';
        writeFileSync(path.join(path.dirname(output),file),after.bytes);
        presented.matchedViews.push({...swap,file,lagMs:Math.round(after.stamp-swap.stamp)});
        pending.splice(i,1);continue;
      }
      const file=path.basename(output,'.json')+'-swap-'+String(presented.swaps.length+1).padStart(2,'0');
      writeFileSync(path.join(path.dirname(output),file+'-before.jpg'),before.bytes);
      writeFileSync(path.join(path.dirname(output),file+'-after.jpg'),after.bytes);
      presented.swaps.push({...swap,file,intervalMs:Math.round(after.stamp-before.stamp)});
      pending.splice(i,1);
    }
  };
  let phase='parked',last=0,lastImage=null;
  const receive=event=>{
    void cdp.send('Page.screencastFrameAck',{sessionId:event.sessionId}).catch(()=>{});
    const bytes=Buffer.from(event.data,'base64');
    timeline.push({stamp:event.metadata.timestamp*1000,bytes});if(timeline.length>32)timeline.shift();
    saveSwaps();
    const {width,height,data}=decode(bytes,{useTArray:true});
    if(patches.sampleVehicle(data,width,height,phase,event.metadata.timestamp*1000)){
      writeFileSync(output.replace(/\.json$/,'-vehicle-display-'+patches.vehicle.episodes+'.jpg'),bytes);
    }
    let count=0,dark=0,sum=0;
    for(let y=Math.floor(height*.24);y<height*.66;y++){
      for(let x=Math.floor(width*.04);x<width*.96;x++){
        const i=(y*width+x)*4;
        sum+=(data[i]+2*data[i+1]+data[i+2])/4;
        if(Math.max(data[i],data[i+1],data[i+2])<18)dark++;
        count++;
      }
    }
    const candidate=patches.sample(data,width,height,phase,event.metadata.timestamp*1000);
    if(candidate&&patches.stats.confirmed<=6){
      const base=path.join(path.dirname(output),path.basename(output,'.json')+'-patch-'+patches.stats.confirmed);
      const frames=[pictures.at(-candidate.duration-1),pictures.at(-candidate.duration),bytes];
      frames.forEach((data,i)=>{if(data)writeFileSync(base+['-before.jpg','-dark.jpg','-after.jpg'][i],data);});
    }
    pictures.push(bytes);if(pictures.length>4)pictures.shift();
    const mean=sum/count, black=dark/count>.90,drop=last>32&&mean<last*.32;
    presented.samples++;presented.black+=Number(black);presented.abruptDrops+=Number(drop);
    presented.worstMean=Math.min(presented.worstMean,mean);
    const group=presented.phases[phase]??={samples:0,black:0};group.samples++;group.black+=Number(black);
    if((black||drop)&&presented.examples.length<8){
      const file=path.basename(output,'.json')+'-frame-'+presented.samples;
      writeFileSync(path.join(path.dirname(output),file+'.jpg'),bytes);
      if(lastImage)writeFileSync(path.join(path.dirname(output),file+'-previous.jpg'),lastImage);
      presented.examples.push({file,phase,mean,previous:last,darkRatio:dark/count,timestamp:event.metadata.timestamp});
    }
    last=mean;lastImage=bytes;
  };
  await page.exposeFunction('__checkDisplayedSwap',swap=>{
    if(presented.swaps.length+pending.length<30){pending.push(swap);saveSwaps();}
  });
  await page.exposeFunction('__matchedVehicleView',sample=>{pending.push(sample);saveSwaps();});
  await page.exposeFunction('__storeCanvasPatch',({number,frames})=>{
    const base=path.join(path.dirname(output),path.basename(output,'.json')+'-canvas-'+number);
    frames.forEach((uri,i)=>{if(uri)writeFileSync(base+['-before.png','-dark.png','-after.png'][i],Buffer.from(uri.split(',')[1],'base64'));});
  });
  await page.evaluate(createPatchTracker,{browser:true});
  await page.evaluate(comparisonPoses=>{
    const canvas=window.__view.renderer.domElement,probe=document.createElement('canvas');
    probe.width=96;probe.height=54;
    const ctx=probe.getContext('2d',{willReadFrequently:true,alpha:false});
    const record={paint:{samples:0,black:0,minMean:255},afterLayout:{samples:0,black:0,minMean:255},contextLosses:0,
      patches:window.__streetPatchTracker.stats,vehicle:window.__streetPatchTracker.vehicle,stageSamples:[],qualitySteps:[],lastWidth:canvas.width,failures:[]};
    const ring=[],snap=document.createElement('canvas');snap.width=96;snap.height=54;
    const save=snap.getContext('2d');
    const sample=(source,now)=>{
      ctx.drawImage(canvas,0,0,96,54);
      const {data}=ctx.getImageData(2,14,92,24);
      if(source==='paint'){
        const image=ctx.getImageData(0,0,96,54),tracker=window.__streetPatchTracker;
        const darkVehicle=tracker.sampleVehicle(image.data,96,54,window.__flashPhase,now);
        if(window.__flashPhase==='drive'&&(darkVehicle||record.stageSamples.length===0)){
          const {renderer,composer}=window.__view,gl=renderer.getContext();
          const prev=renderer.getRenderTarget(),face=renderer.getActiveCubeFace(),mip=renderer.getActiveMipmapLevel();
          const stages={};
          try{
            for(const pass of composer.passes){
              const target=pass.target??pass.frames?.[pass.previous]??pass.output;
              if(!target?.texture||!target.width)continue;
              renderer.setRenderTarget(target);
              const value=new Float32Array(4);
              gl.readPixels(Math.floor(target.width*.5),Math.floor(target.height*.40),1,1,gl.RGBA,gl.FLOAT,value);
              stages[target.texture.name||pass.constructor.name]=Array.from(value,String);
            }
            let atlas;
            window.__car.car.group.traverse(o=>{if(!atlas&&o.material?.envMap?.renderTarget)atlas=o.material.envMap.renderTarget;});
            if(atlas){
              renderer.setRenderTarget(atlas);
              const row=new Float32Array(atlas.width*4);
              let invalid=0,finite=0,max=0;
              for(let y=3;y<atlas.height;y+=16){
                gl.readPixels(0,y,atlas.width,1,gl.RGBA,gl.FLOAT,row);
                for(let x=0;x<row.length;x+=4){
                  if(Number.isFinite(row[x]+row[x+1]+row[x+2])){finite++;max=Math.max(max,row[x],row[x+1],row[x+2]);}
                  else invalid++;
                }
              }
              stages.reflectionAtlas={invalid,finite,max};
            }
          }finally{renderer.setRenderTarget(prev,face,mip);}
          record.stageSamples.push({now,darkVehicle,stages});
        }
        const hit=tracker.sample(image.data,96,54,window.__flashPhase,now);
        if(hit&&tracker.stats.confirmed<=6){
          const frames=[ring.at(-hit.duration-1),ring.at(-hit.duration),image].map(frame=>{
            if(!frame)return null;save.putImageData(frame,0,0);return snap.toDataURL('image/png');
          });
          window.__storeCanvasPatch({number:tracker.stats.confirmed,frames}).catch(()=>{});
        }
        ring.push(image);if(ring.length>4)ring.shift();
      }let sum=0,dark=0;
      for(let i=0;i<data.length;i+=4){
        sum+=(data[i]+2*data[i+1]+data[i+2])/4;
        if(Math.max(data[i],data[i+1],data[i+2])<18)dark++;
      }
      const mean=sum/(data.length/4),group=record[source];
      group.samples++;group.minMean=Math.min(group.minMean,mean);
      if(dark/(data.length/4)>.90){group.black++;if(record.failures.length<16)record.failures.push({source,now,mean,width:canvas.width,phase:window.__flashPhase});}
    };
    let lastEvent=window.__view.stream.status().events.at(-1)?.ms??0;
    const tick=now=>{sample('paint',now);
      if(comparisonPoses.length&&window.__flashPhase==='drive'){
        const [e,z,yaw]=window.__car.car.position();
        for(let i=comparisonPoses.length-1;i>=0;i--){
          const pose=comparisonPoses[i],distance=Math.hypot(e-pose.e,-z-pose.n);
          if(distance>.4||Math.cos(yaw-pose.yaw)<.999)continue;
          window.__matchedVehicleView({type:'comparison',index:pose.index,distance,stamp:performance.timeOrigin+now}).catch(()=>{});
          comparisonPoses.splice(i,1);
        }
      }
      const events=window.__view.stream.status().events;
      for(const event of events)if(event.ms>lastEvent&&['attached','evicted'].includes(event.type)){
        window.__checkDisplayedSwap({type:event.type,id:event.id,stamp:performance.timeOrigin+event.ms}).catch(()=>{});
      }
      lastEvent=events.at(-1)?.ms??lastEvent;if(record.lastWidth!==canvas.width){record.qualitySteps.push({now,width:canvas.width});record.lastWidth=canvas.width;}id=requestAnimationFrame(tick);};
    const onLoss=()=>record.contextLosses++;
    canvas.addEventListener('webglcontextlost',onLoss);
    const observer=new ResizeObserver(()=>sample('afterLayout',performance.now()));
    observer.observe(canvas.parentElement);
    let id=requestAnimationFrame(tick);
    window.__flashStats=record;
    window.__endFlashReview=()=>{cancelAnimationFrame(id);observer.disconnect();canvas.removeEventListener('webglcontextlost',onLoss);return record;};
  },comparisonPoses);
  cdp.on('Page.screencastFrame',receive);
  await cdp.send('Page.startScreencast',{format:'jpeg',quality:65,maxWidth:256,maxHeight:144,everyNthFrame:1});
  return {
    presented,
    async phase(name){phase=name;await page.evaluate(p=>{window.__flashPhase=p;},name);},
    async snapshot(){return {presented,canvas:await page.evaluate(()=>window.__flashStats)};},
    async finish(){await cdp.send('Page.stopScreencast');cdp.off('Page.screencastFrame',receive);return {presented,canvas:await page.evaluate(()=>window.__endFlashReview())};}
  };
}
