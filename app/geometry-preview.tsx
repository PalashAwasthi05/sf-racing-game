'use client';
import {useEffect,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import {Compass,Layers3,PersonStanding,Sun,CloudSun,ChevronRight} from 'lucide-react';
import type {World,Feature} from './king-pilot';
import type {createWorldView} from './world-view';

export default function GeometryPreview(){
 const mount=useRef<HTMLDivElement>(null);
 const api=useRef<Awaited<ReturnType<typeof createWorldView>>|null>(null);
 const [status,setStatus]=useState('Opening the latest geometry preview…');
 const [mode,setMode]=useState('street'),[overcast,setOvercast]=useState(false);
 const [selected,setSelected]=useState<Feature|null>(null);
 const [world,setWorld]=useState<World|null>(null);
 useEffect(()=>{let cancelled=false;let cleanup=()=>{};
  (async()=>{
   const [response,module]=await Promise.all([fetch('/general-preview/world.json',{cache:'no-cache'}),import('./world-view')]);
   if(!response.ok)throw new Error('Preview metadata unavailable');
   const data:World=await response.json();
   if(cancelled||!mount.current)return;
   setWorld(data);
   const result=await module.createWorldView(mount.current,data,id=>setSelected(data.buildings.find(b=>b.id===id)||null),message=>{if(!cancelled)setStatus(message)},'/general-preview/candidate.glb');
   if(cancelled){result.dispose();return;}
   api.current=result;cleanup=result.dispose;setStatus('');
  })().catch(error=>{if(!cancelled)setStatus('The preview could not open. '+error.message)});
  return()=>{cancelled=true;cleanup();api.current=null;};
 },[]);
 function view(value:string){setMode(value);api.current?.view(value)}
 return <main className="world-app">
  <div className="world-canvas" ref={mount} aria-label="Latest general heuristic geometry preview"/>
  <header className="topbar"><div className="identity"><span className="identity-mark"><Layers3 size={20}/></span><div><span className="eyebrow">GENERAL HEURISTIC · REVISION 0.8</span><h1>Geometry preview</h1></div></div><a className="download" href="/previous">Earlier detailed scene</a></header>
  <nav className="viewbar" aria-label="Camera views">
   <Button variant={mode==='orbit'?'default':'ghost'} onClick={()=>view('orbit')}><Compass/>Orbit</Button>
   <Button variant={mode==='street'?'default':'ghost'} onClick={()=>view('street')}><PersonStanding/>Street</Button>
   <Button variant={mode==='top'?'default':'ghost'} onClick={()=>view('top')}><Layers3/>Plan</Button>
   <Button variant="ghost" size="icon" aria-label={overcast?'Afternoon sun':'Soft daylight'} onClick={()=>{setOvercast(!overcast);api.current?.lighting(!overcast)}}>{overcast?<CloudSun/>:<Sun/>}</Button>
  </nav>
  <aside className="preview-status"><strong>{world?.counts.buildings??65} building shapes · facades unfinished</strong><p>Every building uses the same rules. Gray surfaces show where image-based facade reconstruction is still missing. This is a geometry check, not a finished realism pass.</p>{selected&&<p>{selected.name} · {selected.height.toFixed(1)} m roof height · {Math.round(selected.area).toLocaleString()} m² footprint</p>}<a href="/general-preview/report.json" target="_blank" rel="noreferrer">Generation report</a></aside>
  {mode==='street'&&<Button className="next-street" variant="outline" onClick={()=>api.current?.nextStreet()}>Next street view<ChevronRight/></Button>}
  <div className="bottom-left"><span className="controls-help">{mode==='street'?'Driver height 1.25 m · W A S D to move · drag to look':'Drag to orbit · scroll to zoom · right-drag to pan'}</span></div>
  <footer className="attribution">OpenStreetMap contributors · DataSF · USGS</footer>
  {status&&<div className="load-notice" role="status">{status}</div>}
 </main>;
}
