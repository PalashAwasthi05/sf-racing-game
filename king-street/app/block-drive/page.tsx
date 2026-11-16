'use client';
import {useEffect, useRef, useState} from 'react';
import type {createWorldView} from '../world-view';
import type {World} from '../king-pilot';
import {withStartupPackage, type SceneManifest} from '../scene-package';
import {blockVariant, variantLabels, type BlockTrial, type BlockVariant} from '../block-trial';
import {routePackage} from './route-package';
import './trial.css';

const variants=Object.keys(variantLabels) as BlockVariant[];
type Viewer=Awaited<ReturnType<typeof createWorldView>>;
async function readJson<T>(path:string):Promise<T>{
  const response=await fetch(path,{cache:'no-cache'});
  if(!response.ok)throw new Error('Could not load '+path);
  return response.json();
}
export default function BlockDrive(){
  const mount=useRef<HTMLDivElement>(null),viewer=useRef<Viewer|null>(null);
  const [trial,setTrial]=useState<BlockTrial|null>(null),[variant,setVariant]=useState<BlockVariant>('baseline');
  const [notice,setNotice]=useState('Opening the block…'),[ready,setReady]=useState(false);
  const selection=useRef(0);
  const [variantNotice,setVariantNotice]=useState(''),[failedVariant,setFailedVariant]=useState<BlockVariant|null>(null);
  const [position,setPosition]=useState({mode:'manual',distance:0,length:1});
  async function choose(v:BlockVariant){
    const request=++selection.current;setVariant(v);setFailedVariant(null);
    setVariantNotice(v==='baseline'?'':'Loading '+variantLabels[v]+'… You can keep driving.');
    const url=new URL(window.location.href);url.searchParams.set('version',v);window.history.replaceState(null,'',url);
    try{
      const applied=await viewer.current?.block?.variant(v);
      if(request!==selection.current||!applied)return;
      setVariantNotice('');
    }catch{
      if(request!==selection.current)return;
      setVariant('baseline');setFailedVariant(v);setVariantNotice('Could not load '+variantLabels[v]+'. Current mesh is still available.');
    }
  }
  useEffect(()=>{
    let cancelled=false,timer=0,opened:Viewer|null=null;
    const onKey=(e:KeyboardEvent)=>{if(!viewer.current||e.altKey||e.ctrlKey||e.metaKey||e.repeat)return;
      const i=['Digit1','Digit2','Digit3'].indexOf(e.code);if(i>=0){e.preventDefault();choose(variants[i]);}};
    async function open(){
      const v=blockVariant(new URL(window.location.href).searchParams.get('version'));setVariant(v);
      const [{config,manifest,world},module]=await Promise.all([readJson<BlockTrial>('/block-trial/trial.json').then(async config=>{
        const archive=await readJson<SceneManifest>(config.manifest);
        if(!archive.world)throw new Error('The base scene has no street data.');
        try{
          const manifest=routePackage(await readJson<unknown>('/compare/route-block.json'),archive,config);
          if(manifest)return {config,manifest,world:await readJson<World>(manifest.world)};
        }catch{ /* A missing compact package keeps the original comparison available. */ }
        const [manifest,world]=await Promise.all([withStartupPackage(archive),readJson<World>(archive.world)]);
        return {config,manifest,world};
      }),import('../world-view')]);
      if(cancelled||!mount.current)return;
      setTrial(config);document.title=config.label+' — Block comparison';
      opened=await module.createWorldView(mount.current,world,()=>{},s=>{if(!cancelled)setNotice(s);},undefined,manifest,config);
      if(cancelled){opened.dispose();return;}
      viewer.current=opened;void choose(v);
      setPosition(opened.block!.state());setNotice('');setReady(true);
      timer=window.setInterval(()=>{if(viewer.current?.block)setPosition(viewer.current.block.state());},120);
      window.addEventListener('keydown',onKey);
    }
    open().catch(e=>{if(!cancelled)setNotice('The comparison could not open. '+e.message);});
    return ()=>{cancelled=true;selection.current++;window.clearInterval(timer);window.removeEventListener('keydown',onKey);viewer.current=null;opened?.dispose();};
  },[]);
  const block=()=>viewer.current?.block;
  return <main className="world-app block-drive"><div className="world-canvas" ref={mount} aria-label="Drivable street reconstruction"/>
    <header className="trial-header"><div><h1>{trial?.label||'Block comparison'}</h1>{trial&&<small>{Math.round(trial.lengthM)} m driving comparison{trial.comparisonUrl&&<> · <a href={trial.comparisonUrl} target="_blank" rel="noreferrer">Compare street photos</a></>}</small>}</div>
      <nav className="trial-switch" aria-label="Building representation">{variants.map((v,i)=><button key={v} disabled={!ready} aria-pressed={variant===v} onClick={()=>choose(v)} title={'Switch at the current position ('+(i+1)+')'}>{variantLabels[v]}</button>)}</nav></header>
    {notice&&<div className="load-notice" role="status">{notice}</div>}
    <section className="trial-controls" aria-label="Drive and replay" inert={!ready}>
      {variantNotice&&<div role="status">{variantNotice}{failedVariant&&<button onClick={()=>choose(failedVariant)}>Retry {variantLabels[failedVariant]}</button>}</div>}
      <div className="trial-actions"><button onClick={()=>block()?.manual(true)}>Restart drive</button>
        {position.mode!=='manual'&&<button onClick={()=>block()?.manual(false)}>Take the wheel</button>}
        <button onClick={()=>position.mode==='playing'?block()?.pause():block()?.play()}>{position.mode==='playing'?'Pause replay':'Replay route'}</button>
        <span className="trial-distance">{Math.round(position.distance)} / {Math.round(position.length)} m</span></div>
      <input aria-label="Position along the route" type="range" min="0" max="1000" step="1" value={Math.round(1000*position.distance/position.length)}
        onChange={e=>{block()?.seek(Number(e.target.value)/1000);const s=block()?.state();if(s)setPosition(s);}}/>
      <div className="trial-angles"><span>Look</span>{([['Left',Math.PI*.40],['Ahead',0],['Right',-Math.PI*.40]] as const).map(([name,angle])=><button key={name} onClick={()=>block()?.look(angle)}>{name}</button>)}</div>
    </section>
    <footer className="trial-credit"><a href={trial?.sourceUrl} target="_blank" rel="noreferrer">Street imagery © Mapillary contributors</a> · <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noreferrer">CC BY-SA 4.0</a> · <a href={trial?.attribution} target="_blank" rel="noreferrer">Image credits</a> · © OpenStreetMap contributors · DataSF · USGS</footer>
  </main>;
}
