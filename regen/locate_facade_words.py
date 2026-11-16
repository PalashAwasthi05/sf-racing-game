"""Locate repeated readable words in the *same facade frame* across street images.
Writes proposals only. A matching word on a car, street sign or meter scale is
not wall lettering. Preserve the source crop; require wall attachment separately.
"""
import argparse,hashlib,json,re,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source


def candidates(row,v,w,h):
    if row.get('error'):return []
    pad,ppm=v['rectPad'],v['ppm'];out=[]
    for word in row['words']:
        txt=' '.join(word['text'].split());x0,y0,x1,y1=word['rect'];z0=h-(y1-pad)/ppm;z1=h-(y0-pad)/ppm
        if word['confidence']<.8 or not(2<=len(txt)<=55) or len(re.findall('[A-Za-z]',txt))<3:continue
        if x0<pad+2 or x1>pad+w*ppm-2 or not 1.65<z0<z1<h-.1 or z1-z0>1.5:continue
        if x1-x0<14 or y1-y0<5:continue
        out.append({'text':txt,'key':txt.casefold(),'boxM':[(x0-pad)/ppm,z0,(x1-pad)/ppm,z1],'boxPx':[x0,y0,x1,y1],'confidence':word['confidence'],'image':v['image']})
    return out


def same(a,b):
    if a['image']==b['image'] or a['key']!=b['key']:return False
    aa=np.array(a['boxM']);bb=np.array(b['boxM']);sa=aa[2:]-aa[:2];sb=bb[2:]-bb[:2];ca=(aa[2:]+aa[:2])/2;cb=(bb[2:]+bb[:2])/2
    return bool((np.minimum(sa,sb)/np.maximum(sa,sb)>.66).all() and (abs(ca-cb)<np.maximum(.13,np.minimum(sa,sb)*.37)).all())


def main():
    ap=argparse.ArgumentParser();
    for k in ('area','profiles','output','ocr'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();area=a.area.resolve();a.output.mkdir(parents=True,exist_ok=True);sides=load_registered_sides(area/'regen/sides-reg');targets=[];files={}
    for p in json.loads(a.profiles.read_text())['profiles']:
        s=resolve_exact_source(p,area,sides)
        if s['status']!='ok':continue
        sid=p['id'][6:];piece=sides[sid]['piece'];vs=[]
        for v in piece['views']:
            path=str(sides[sid]['path'].parent/'img'/v['rect']);
            if v.get('visibleFraction',0)<.50 or v.get('incidence',0)<.45:continue
            vs.append((v,path));files[path]=None
        if len({v['image'] for v,f in vs})>=2:targets.append((p,s,piece,vs))
    cache=a.output/'ocr.jsonl';known={}
    if cache.exists():
        for l in cache.read_text().splitlines():
            r=json.loads(l);known[r['path']]=r
    missing=[f for f in files if f not in known];print('profiles',len(targets),'unique images',len(files),'missing',len(missing),flush=True)
    if missing:
        with (a.output/'queue.jsonl').open('w') as f:
            for name in missing:f.write(json.dumps(name)+'\n')
        with (a.output/'queue.jsonl').open('r') as q,cache.open('a') as c:
            with subprocess.Popen([str(a.ocr)],stdin=q,stdout=c) as proc:
                if proc.wait()!=0:raise RuntimeError('local OCR did not finish')
        for l in cache.read_text().splitlines():
            r=json.loads(l);known[r['path']]=r
    matches=[];det=0
    for p,s,piece,vs in targets:
        words=[q for v,f in vs for q in candidates(known[f],v,piece['lengthM'],piece['heightM'])];det+=len(words)
        for w in words:
            if w['image']!=s['image']:continue
            ws=[q for q in words if same(q,w)]
            if ws:matches.append({'side':p['id'],'primary':w,'witnesses':ws,'cropFiles':{v['image']:f for v,f in vs}})
    errors=sum(bool(v.get('error')) for v in known.values());out={'profilesWithMultipleViews':len(targets),'uniqueImages':len(known),'readFailures':errors,'possibleWords':det,'matched':matches,'status':'source inspection required for surface attachment; no geometry emitted'};(a.output/'proposals.json').write_text(json.dumps(out,indent=1));print('source errors',errors,'possible words',det,'repeated words in same frame',len(matches))
if __name__=='__main__':main()
