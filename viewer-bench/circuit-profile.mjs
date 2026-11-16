// Optional diagnostics on the actual circuit. The basic drive does not enable tracing.
// Count each backing store once, including archive slices retained by a tiny attribute.
// Take these snapshots with the car stopped, outside the measured lap.
export function geometryMemory() {
  const view=window.__view, geometries=new Set(), buffers=new Map();
  function attribute(attr,owner) {
    const array=attr?.isInterleavedBufferAttribute?attr.data.array:attr?.array;
    if(!ArrayBuffer.isView(array)||!array.byteLength)return;
    let row=buffers.get(array.buffer);
    if(!row){row={owner,slices:[]};buffers.set(array.buffer,row);}
    row.slices.push([array.byteOffset,array.byteOffset+array.byteLength]);
  }
  function visit(mesh) {
    const geometry=mesh.geometry;
    if(!geometry)return;
    const owner=(mesh.name||mesh.parent?.name||mesh.type).slice(0,96);
    attribute(mesh.instanceMatrix,owner);attribute(mesh.instanceColor,owner);
    if(geometries.has(geometry))return;
    geometries.add(geometry);
    attribute(geometry.index,owner);
    for(const attr of Object.values(geometry.attributes))attribute(attr,owner);
    for(const list of Object.values(geometry.morphAttributes))for(const attr of list)attribute(attr,owner);
  }
  view.scene.traverse(visit);
  for(const original of view.objects)visit(original);
  let backingBytes=0,referencedBytes=0;
  const largest=[];
  for(const [buffer,{owner,slices}] of buffers){
    backingBytes+=buffer.byteLength;
    let end=0,bytes=0;
    for(const [lo,hi] of slices.sort((a,b)=>a[0]-b[0])){
      bytes+=Math.max(0,hi-Math.max(end,lo));end=Math.max(hi,end);
    }
    referencedBytes+=bytes;largest.push({firstOwner:owner,backingBytes:buffer.byteLength,referencedBytes:bytes});
  }
  return {backingBytes,referencedBytes,buffers:buffers.size,geometries:geometries.size,
    largest:largest.sort((a,b)=>b.backingBytes-a.backingBytes).slice(0,8),
    uploadedCounts:{...view.renderer.info.memory},imagePool:view.imagePool?.()};
}

export function sampleGpu() {
  const {renderer,composer}=window.__view,gl=renderer.getContext();
  const ext=gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const labels=['street and shadows','contact occlusion','temporal resolve','exposure meter','composite','bloom','display grade'];
  const rows=labels.map(()=>({gpu:[],cpu:[],triangles:[],calls:[]}));
  const pending=new Map();
  let frame=0;
  function collect(){
    if(!ext)return;
    const disjoint=gl.getParameter(ext.GPU_DISJOINT_EXT);
    for(const [q,index] of pending){
      if(!disjoint&&!gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE))continue;
      if(!disjoint)rows[index].gpu.push(gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6);
      gl.deleteQuery(q);pending.delete(q);
    }
  }
  for(const [i,pass] of composer.passes.entries()){
    const render=pass.render.bind(pass);
    if(!rows[i])throw Error('Unexpected render passes');
    pass.render=function(...args){
      if(i===0)frame++;
      if(frame%11||Math.floor(frame/11)%rows.length!==i)return render(...args);
      collect();
      const q=ext&&pending.size<3?gl.createQuery():null;
      const calls=renderer.info.render.calls,tri=renderer.info.render.triangles,then=performance.now();
      if(q)gl.beginQuery(ext.TIME_ELAPSED_EXT,q);
      try{return render(...args);}finally{
        if(q){gl.endQuery(ext.TIME_ELAPSED_EXT);pending.set(q,i);}
        rows[i].cpu.push(performance.now()-then);rows[i].calls.push(renderer.info.render.calls-calls);
        rows[i].triangles.push(renderer.info.render.triangles-tri);
      }
    };
  }
  function median(a){const v=a.slice().sort((x,y)=>x-y);return v.length?v[Math.floor(v.length/2)]:null;}
  window.__circuitProfile=()=>({
    timerAvailable:!!ext,resources:{...renderer.info.memory,shaderPrograms:renderer.info.programs?.length},staticSubmissions:window.__view.budget.submissions,
    passes:rows.map((r,i)=>({name:labels[i],samples:r.gpu.length,gpuElapsedMedianMs:median(r.gpu),cpuSubmitMedianMs:median(r.cpu),medianTriangles:median(r.triangles),medianCalls:median(r.calls)}))
  });
}

// PipelineReporter distinguishes a fresh compositor frame from an animation callback.
export async function startPresentation(connection,options={}) {
  // Keep SF_CIRCUIT_PRESENT=1's original full trace unchanged. The opt-in
  // display mode records actual compositor presentation feedback, not rAF or
  // LoAF, without enabling unrelated benchmark/cc/viz categories.
  // Chromium endpoint: components/viz/service/display/display.cc,
  // Display::DidReceivePresentationFeedback emits Display::FrameDisplayed in
  // "benchmark,viz,disabled-by-default-display.framedisplayed".
  // https://chromium.googlesource.com/chromium/src/+/HEAD/components/viz/service/display/display.cc
  const mode=options.mode??process.env.SF_CIRCUIT_PRESENT_MODE??'full';
  if(!['full','display'].includes(mode))throw Error('Presentation mode must be full or display');
  const categories=mode==='display'?['disabled-by-default-display.framedisplayed','blink.user_timing']:['benchmark','blink.user_timing'];
  const traceBufferSizeInKb=mode==='display'?8192:32768;
  const frames=new Map(), marks=new Map(), displayTimes=new Set();
  const collect=({method,params,sessionId})=>{
    if(sessionId||method!=='Tracing.dataCollected')return;
    for(const e of params.value){
      if(e.name==='Display::FrameDisplayed'&&e.ph!=='E'&&e.ph!=='e')displayTimes.add(e.ts);
      if(e.name==='sf-circuit-start'||e.name==='sf-circuit-end')marks.set(e.name,e.ts);
      if(mode==='display')continue;
      const r=e.args?.frame_reporter;
      if(e.name!=='PipelineReporter'||e.ph!=='b'||!r?.has_main_animation)continue;
      const key=[r.layer_tree_host_id,r.frame_source,r.frame_sequence].join('/');
      let s=frames.get(key);if(!s){s={ts:e.ts,states:new Set()};frames.set(key,s);}
      s.states.add(r.state);
    }
  };
  connection.on('protocol-event',collect);
  try{await connection.send('Tracing.start',{traceConfig:{recordMode:'recordContinuously',traceBufferSizeInKb,includedCategories:categories},transferMode:'ReportEvents'});}
  catch(error){connection.off('protocol-event',collect);throw error;}
  return async(report)=>{
    const done=new Promise(resolve=>{
      const listener=e=>{if(!e.sessionId&&e.method==='Tracing.tracingComplete'){connection.off('protocol-event',listener);resolve(e.params);}};
      connection.on('protocol-event',listener);
    });
    try{
      await connection.send('Tracing.end');const {dataLossOccurred}=await done;
      const start=marks.get('sf-circuit-start'),end=marks.get('sf-circuit-end');
      const hasBothMarkers=Number.isFinite(start)&&Number.isFinite(end)&&end>=start;
      const sequence=[...frames.values()].filter(f=>f.ts>=start&&f.ts<=end);
      const window=(a,b)=>{if(mode==='display'||!hasBothMarkers)return {reportedSequences:null,presentedAll:null,incomplete:null,otherStates:[]};
        const v=sequence.filter(f=>f.ts>=a&&f.ts<=b);const missed=v.filter(f=>!f.states.has('STATE_PRESENTED_ALL'));
        return {reportedSequences:v.length,presentedAll:v.length-missed.length,incomplete:missed.length,otherStates:[...new Set(missed.flatMap(f=>[...f.states]))]};};
      const ts=ms=>start+(ms-report.startMs)*1000;
      const display=(a,b)=>{
        const times=[...displayTimes].filter(t=>t>=a&&t<=b).sort((x,y)=>x-y);
        const gaps=[],delays=[];
        for(let i=1;i<times.length;i++){
          const elapsed=(times[i]-times[i-1])/1000;gaps.push(elapsed);
          if(elapsed>25)delays.push({atMs:report.startMs+(times[i]-start)/1000,intervalMs:elapsed});
        }
        gaps.sort((x,y)=>x-y);
        return {updates:times.length,fps:gaps.length?gaps.length*1e6/(times.at(-1)-times[0]):null,
          p99Ms:gaps[Math.ceil(gaps.length*.99)-1]??null,longestMs:gaps.at(-1)??null,over25:delays.length,delays:delays.sort((a,b)=>b.intervalMs-a.intervalMs).slice(0,32)};
      };
      const presented=display(start,end);
      return {mode,traceCategories:categories,traceBufferSizeInKb,pipelineReporter:mode==='full',
        complete:hasBothMarkers&&!dataLossOccurred&&presented.updates>0,dataLossOccurred:!!dataLossOccurred,hasBothMarkers,
        ...window(start,end),traceSpanMs:hasBothMarkers?(end-start)/1000:null,display:presented,
        turns:report.turns.map(t=>({turn:t.turn,...window(ts(t.startMs-500),ts(t.endMs+500)),display:display(ts(t.startMs-500),ts(t.endMs+500))})),
        crossings:report.crossings.map(e=>({id:e.id,...window(ts(e.ms-500),ts(e.ms+500)),display:display(ts(e.ms-500),ts(e.ms+500))}))};
    }finally{connection.off('protocol-event',collect);}
  };
}
