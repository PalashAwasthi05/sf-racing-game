'use client';
import {useEffect,useRef,useState} from 'react';
import {loadSceneManifest, downloadScene, type SceneManifest} from './scene-package';
import {validateDriveStream, type DriveStream} from './drive-stream';
import {pilotStreamRequest,pilotAreaHref,pilotStreamMatches} from './king-pilot-route';
import {Button} from '@/components/ui/button';
import {Switch} from '@/components/ui/switch';
import {Sheet,SheetContent,SheetTitle,SheetDescription} from '@/components/ui/sheet';
import {DropdownMenu,DropdownMenuTrigger,DropdownMenuContent,DropdownMenuGroup,DropdownMenuLabel,DropdownMenuRadioGroup,DropdownMenuRadioItem} from '@/components/ui/dropdown-menu';
import {ArrowDownToLine,ArrowUpRight,Car,ChevronDown,ChevronRight,Droplets,Moon,Sun,Sunset,CloudSun,X,Compass,Focus,Layers3,MapPin,PersonStanding,RotateCcw,Route} from 'lucide-react';
export type Feature={id:string;name:string;height:number;heightSource:string;footprintSource:string;roofParts:number;center:number[];area:number;facade:string;pointCount:number;sourceUrl:string;facadeEvidence:string;fittedWalls:number;totalWalls:number};
export type World={label?:string;studyRadiusM?:number;contextRadiusM?:number;previewRevision?:string;updateSummary?:string;streetViews?:{position:number[];lookAt:number[];label:string}[];center:{lat:number;lon:number};buildings:Feature[];counts:Record<string,number>;ground:number[];grid:{min:number;step:number;n:number};sources:{name:string;date:string;detail:string;url:string}[]};
const LIGHTS=[['afternoon','Afternoon sun'],['golden','Golden hour'],['overcast','Overcast'],['night','Night drive']] as const;
export default function KingPilot(){
 const mount=useRef<HTMLDivElement>(null);
 const pendingView=useRef<string|null>(null);
 const api=useRef<{view:(v:string)=>void;layer:(l:string,on:boolean)=>void;evidence:(on:boolean)=>void;select:(id:string)=>void;lighting:(preset:string)=>void;damp:(on:boolean)=>void;nextStreet:()=>void;drive:(on:boolean)=>void;car:(on:boolean)=>void}|null>(null);
 const [world,setWorld]=useState<World|null>(null),[selected,setSelected]=useState<Feature|null>(null),[status,setStatus]=useState('Opening the neighborhood…'),[mode,setMode]=useState('street'),[evidence,setEvidence]=useState(false),[sources,setSources]=useState(false);
 const [driving,setDriving]=useState(false),[damp,setDamp]=useState(false);
 const [panel,setPanel]=useState(false),[light,setLight]=useState('afternoon');
 const [download,setDownload]=useState<SceneManifest|null>(null),[saving,setSaving]=useState(false);
 const [ready,setReady]=useState(false),[extent,setExtent]=useState('');
 const downloadAbort=useRef(false);
 async function saveModel(){
  if(!download||saving)return;
  setSaving(true);setStatus('');
  try{
   const blob=await downloadScene(download);if(downloadAbort.current)return;
   const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;
   a.download=(world?.label||'world').toLowerCase().replace(/[^a-z0-9]+/g,'-')+'.glb';a.click();
   window.setTimeout(()=>URL.revokeObjectURL(url),60000);
  }catch{if(!downloadAbort.current)setStatus('The model download failed. Select 3D model to retry.');}
  finally{if(!downloadAbort.current)setSaving(false);}
 }
 const [layers,setLayers]=useState({facades:true,vegetation:true,street_objects:true,markings:true});
 // the areas this viewer can open (public/compare/areas.json, kept by the area preview step) and the build shown now
 const [areas,setAreas]=useState<{id:string;label:string;detail?:string;manifest:string|null}[]>([]),[here,setHere]=useState('');
 useEffect(()=>{setHere(new URL(window.location.href).searchParams.get('manifest')||'');
  fetch('/compare/areas.json',{cache:'no-cache'}).then(r=>r.ok?r.json() as Promise<{areas?:unknown}>:null).then(d=>{if(Array.isArray(d?.areas))setAreas(d.areas)}).catch(()=>{});},[]);
 useEffect(()=>{let cancelled=false,cleanup=()=>{};downloadAbort.current=false;
  const syncPose=()=>{const u=new URL(window.location.href);if(u.searchParams.has('viewFrom')||new URLSearchParams(u.hash.slice(1)).has('viewFrom')){setMode('orbit');setDriving(false);}};syncPose();window.addEventListener('hashchange',syncPose);
  async function openWorld(){
   const [{manifest,data,stream},module]=await Promise.all([(async()=>{
    let stream:DriveStream|undefined,archive:SceneManifest|undefined;
    const request=pilotStreamRequest(new URL(window.location.href));
    if(request==='match-source'||!request)archive=await loadSceneManifest();
    if(request){
      const index=await fetch(request==='corridor'?'/compare/drive-stream.json':'/compare/neighborhood-stream.json',{cache:'no-cache'})
       .catch(error=>{if(request==='match-source')return null;throw error;});
      if(!index?.ok&&request!=='match-source')throw new Error('The neighborhood is unavailable.');
      if(index?.ok){
       const descriptor=await index.json() as DriveStream;
       if(pilotStreamMatches(request,archive?.contentHash,descriptor.sourceWorldHash)){
        const candidate=validateDriveStream(descriptor);
        if(candidate.scope!==(request==='corridor'?'drive-stream':'neighborhood-stream'))throw new Error('Unexpected neighborhood package.');
        stream=candidate;
       }
      }
    }
    const manifest=stream?{...stream.base,world:stream.world,cars:stream.cars??stream.base.cars,overhead:stream.overhead??stream.base.overhead}:archive!;
    const worldPath=manifest.world && /^\/compare\/[\w.-]+\.json$/.test(manifest.world) ? manifest.world : '/world.json';
    const response=await fetch(worldPath,{cache:'no-cache'});
    if(!response.ok)throw new Error('Model data unavailable');
    const data:World=await response.json();
    return {manifest,data,stream};
   })(), import('./world-view')]);
   if(cancelled||!mount.current)return;
   setWorld(data);
   if(stream){
    setHere(stream.scope==='neighborhood-stream'?'@neighborhood':'@corridor');
    const boxes=stream.cells.map(c=>c.tileEN??c.boundsEN);
    const width=Math.max(...boxes.map(b=>b[2]))-Math.min(...boxes.map(b=>b[0]));
    const height=Math.max(...boxes.map(b=>b[3]))-Math.min(...boxes.map(b=>b[1]));
    setExtent(stream.scope==='neighborhood-stream'?`Neighborhood · ${Math.round(width)} × ${Math.round(height)} m tile coverage`:'Four-block driving corridor');
   }else setExtent(`${data.studyRadiusM??200} m study radius`);
   if(data.label)document.title=data.label+' — World Pilot';
   setSelected([...data.buildings].sort((a,b)=>Math.hypot(a.center[0],a.center[1])-Math.hypot(b.center[0],b.center[1]))[0]);
   const query=new URL(window.location.href).searchParams;
   const initialLight=LIGHTS.find(([key])=>key===query.get('light'))?.[0]??'afternoon';
   setLight(initialLight);const initialDamp=query.get('surface')==='damp';setDamp(initialDamp);
   if(!pendingView.current&&query.get('drive')==='1'&&!query.has('viewFrom')&&!window.location.hash){pendingView.current='car';setMode('car');}
   const result=await module.createWorldView(mount.current,data,id=>setSelected(data.buildings.find(b=>b.id===id)||null),setStatus,undefined,manifest,undefined,initialLight,initialDamp,()=>pendingView.current,stream);
   if(cancelled){result.dispose();return;}
   api.current=result;
   setReady(true);
   if(pendingView.current==='car')result.car(true);
   else if(pendingView.current)result.view(pendingView.current);
   pendingView.current=null;
   setDownload(stream?null:manifest);
   cleanup=result.dispose;
   setStatus('');
  }
  openWorld().catch(e=>{if(!cancelled)setStatus('The model could not open. Reload to retry. '+e.message)});
  return()=>{cancelled=true;downloadAbort.current=true;window.removeEventListener('hashchange',syncPose);cleanup();api.current=null;};},[]);
 function view(v:string){setDriving(false);setMode(v);if(api.current)api.current.view(v);else pendingView.current=v}
 function driveCar(){const on=mode!=='car';setDriving(false);setMode(on?'car':'street');if(api.current)api.current.car(on);else pendingView.current=on?'car':'street'}
 return <main className="world-app" data-mode={mode}><div className="world-canvas" ref={mount} aria-label={world?.label ? `Interactive 3D reconstruction of ${world.label}` : 'Interactive 3D reconstruction'}/>
 <header className="topbar"><div className="identity"><span className="identity-mark"><Layers3 size={20}/></span><div><span className="eyebrow">{world?.previewRevision || "SAN FRANCISCO / WORLD PILOT"}</span><h1><DropdownMenu><DropdownMenuTrigger id="world-area-menu" className="area-trigger" title="Open another area">{world?.label || "World pilot"}<ChevronDown size={18}/></DropdownMenuTrigger><DropdownMenuContent className="w-72"><DropdownMenuGroup><DropdownMenuLabel>Open an area</DropdownMenuLabel><DropdownMenuRadioGroup value={here} onValueChange={(v:string)=>{if(v!==here)window.location.assign(pilotAreaHref(v))}}>{([{id:'neighborhood-stream',label:'Pacific Heights',detail:'Full neighborhood · spatial tiles',manifest:'@neighborhood'},{id:'drive-stream',label:'Four-block corridor',detail:'Original driving route',manifest:'@corridor'},...(areas.length?areas:[{id:'king-street',label:'188 King Street',manifest:null}])]).map(a=><DropdownMenuRadioItem key={a.id} value={a.manifest||''} className="area-item"><span><b>{a.label}</b>{a.detail&&<small>{a.detail}</small>}</span></DropdownMenuRadioItem>)}</DropdownMenuRadioGroup></DropdownMenuGroup></DropdownMenuContent></DropdownMenu></h1></div></div><div className="top-actions"><span className="version">STREET STUDY</span><button className="download" onClick={saveModel} disabled={!download||saving} aria-label={saving?'Preparing 3D model download':'Download 3D model'}><ArrowDownToLine size={16}/><span>{saving?'Preparing…':'3D model'}</span></button></div></header>
 <nav className="viewbar" aria-label="Camera views"><Button disabled={!ready} variant={mode==='orbit'?'default':'ghost'} onClick={()=>view('orbit')}><Compass/>Orbit</Button><Button disabled={!ready} variant={mode==='street'?'default':'ghost'} onClick={()=>view('street')}><PersonStanding/>Street</Button><Button disabled={!ready} variant={mode==='top'?'default':'ghost'} onClick={()=>view('top')}><Layers3/>Plan</Button><Button disabled={!ready} variant={mode==='car'?'default':'ghost'} onClick={driveCar}><Car/>{mode==='car'?'Park':'Drive'}</Button><Button disabled={!ready} variant={driving?'default':'ghost'} onClick={()=>{setMode('street');setDriving(!driving);api.current?.drive(!driving)}}><Route/>{driving?'Pause':'Tour'}</Button><span className="view-divider"/><Button disabled={!ready} variant="ghost" size="icon" title={'Lighting: '+LIGHTS.find(l=>l[0]===light)?.[1]+' (click to change)'} aria-label={'Lighting: '+LIGHTS.find(l=>l[0]===light)?.[1]} onClick={()=>{const next=LIGHTS[(LIGHTS.findIndex(l=>l[0]===light)+1)%LIGHTS.length][0];setLight(next);api.current?.lighting(next)}}>{light==='night'?<Moon/>:light==='golden'?<Sunset/>:light==='overcast'?<CloudSun/>:<Sun/>}</Button><Button disabled={!ready} variant={damp?'default':'ghost'} size="icon" title={damp?'Switch to dry roads':'Switch to damp roads'} aria-label="Damp road surface" aria-pressed={damp} onClick={()=>{setDamp(!damp);api.current?.damp(!damp)}}><Droplets/></Button><Button disabled={!ready} variant="ghost" size="icon" aria-label="Reset camera" onClick={()=>view(mode)}><RotateCcw/></Button></nav>
 <Button className="inspect-toggle" variant="outline" onClick={()=>setPanel(!panel)} aria-expanded={panel}>{panel?<X/>:<Layers3/>}{panel?"Close details":"Scene details"}</Button>{mode==='street'&&<Button disabled={!ready} className="next-street" variant="outline" onClick={()=>{setDriving(false);api.current?.nextStreet()}}>Next street view<ChevronRight/></Button>}<aside inert={!ready} className={`inspector ${panel?'is-open':'is-closed'}`}><div className="inspector-title"><span className="eyebrow">SELECTED BUILDING</span><MapPin size={15}/></div><h2>{selected?.name||'Select a building'}</h2><p className="building-note">{selected?.facade||'Click a building to inspect its sources.'}</p>
 {selected&&<><div className="measurements"><div><strong>{selected.height.toFixed(1)}<small> m</small></strong><span>Reconstructed height</span></div><div><strong>{Math.round(selected.area).toLocaleString()}<small> m²</small></strong><span>Footprint area</span></div></div><div className="source-lines"><div><i className="dot observed"/><span>Outline</span><b>{selected.footprintSource}</b></div><div><i className="dot measured"/><span>Height</span><b>{selected.heightSource}</b></div><div><i className="dot inferred"/><span>Facade</span><b>{selected.facadeEvidence}</b></div></div><Button variant="outline" className="focus-button" onClick={()=>{setMode('orbit');api.current?.select(selected.id)}}><Focus/>Focus building</Button></>}
 <div className="layer-section"><div className="eyebrow">SCENE LAYERS</div>{([['facades','Facade details'],['vegetation','Trees'],['street_objects','Street objects'],['markings','Road markings']] as const).map(([key,label])=><label key={key} className="layer-row"><span>{label}</span><Switch id={"world-layer-"+key} nativeButton render={<button type="button"/>} checked={layers[key]} onCheckedChange={on=>{setLayers(s=>({...s,[key]:on}));api.current?.layer(key,on)}} aria-label={label}/></label>)}<label className="layer-row evidence-toggle"><span>Show evidence colors</span><Switch id="world-evidence-colors" nativeButton render={<button type="button"/>} checked={evidence} onCheckedChange={on=>{setEvidence(on);api.current?.evidence(on)}} aria-label="Show evidence colors"/></label></div>
 {evidence&&<div className="legend"><span><i className="dot measured"/>LiDAR height</span><span><i className="dot inferred"/>Inferred detail</span><span><i className="dot observed"/>Visual fit</span></div>}<Button variant="ghost" className="sources-button" onClick={()=>setSources(true)}>Sources & remaining gaps<ArrowUpRight/></Button></aside>
 <div className="bottom-left"><span className="extent">{extent||'Opening neighborhood…'}</span>{mode!=='car'&&<span className="controls-help">{mode==='street'?'Driver height 1.25 m · W A S D to move · drag to look':'Drag to orbit · scroll to zoom · right-drag to pan'}</span>}</div><div className="north"><span>N</span><Compass size={24}/><small>Real-world scale</small></div><footer className="attribution">© OpenStreetMap contributors · DataSF · USGS 2023</footer>{status&&<div className="load-notice" role="status">{status}</div>}
 <Sheet open={sources} onOpenChange={setSources}><SheetContent className="source-sheet"><span className="eyebrow">RECONSTRUCTION IN PROGRESS</span><SheetTitle>What this pass knows</SheetTitle><SheetDescription>{world?.updateSummary || "The detailed scene combines mapped geometry and interpreted facade layouts. Unseen details and metric dimensions still need independent verification."}</SheetDescription><div className="source-counts"><span><b>{world?.counts.buildings}</b> buildings</span><span><b>{world?.counts.windowAssemblies?.toLocaleString()}</b> window assemblies</span><span><b>{world?.counts.imageFittedWalls} / {world?.counts.facadeWalls}</b> inherited visual fits; not independently verified</span><span><b>{world?.counts.envelopeNodes}</b> envelope parts across {world?.counts.envelopeFrontages} frontages</span><span><b>{world?.counts.facadeLetteringPlacements}</b> inherited sign placements</span><span><b>{world?.counts.directBoundaryOpenings}</b> openings fitted from four edges</span><span><b>{world?.counts.lidarPoints?.toLocaleString()}</b> LiDAR points</span></div>{world?.sources.map(s=><a href={s.url} target="_blank" rel="noreferrer" key={s.name} className="source-item"><div><h3>{s.name}<ArrowUpRight size={15}/></h3><p>{s.detail}</p></div><span>{s.date}</span></a>)}<div className="remaining"><h3>Next refinements</h3><p>More independent views, calibrated depths, sculpted ornament, verified curb and paint geometry, and image-verified sign placement. Observed layout and inferred completion are distinct. This scene has not passed a 1:1 street-accuracy test.</p></div></SheetContent></Sheet>
 </main>;
}
