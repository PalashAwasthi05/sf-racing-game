"""Recover a photographed opaque top panel INSIDE a glazed opening. A note or a
dark window alone cannot establish a board. Require source-recognized opaque
infill, one sustained color/texture boundary in the opening, and agreement
across two independent views. Sky, glazing, frames and shutters stay separate.
"""
import argparse,hashlib,json,re
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source


def samples(im,rect,frame):
    H,W=frame['heightM'],frame['lengthM'];ppm,pad=frame['ppm'],frame['rectPad'];x,z,X,Z=np.array(rect)*[W,H,W,H];width,height=X-x,Z-z
    # Use the middle of the aperture, clear of jambs and the narrowing arch.
    lo=round(pad+(x+.13*width)*ppm);hi=round(pad+(X-.13*width)*ppm);t=round(pad+(H-(z+.86*height))*ppm);b=round(pad+(H-(z+.28*height))*ppm)
    if not(pad<=lo<hi<im.shape[1]-pad+1 and pad<=t<b<im.shape[0]-pad+1 and (hi-lo)>=32 and (b-t)>=28):return []
    ar=im[t:b,lo:hi,::-1].astype(float);grey=ar@[.2126,.7152,.0722];strip=np.array([np.median(q,axis=1) for q in np.array_split(grey,7,axis=1)]);sm=gaussian_filter1d(strip,.9,axis=1)
    # An arch head can hold a very dark valance, a pale blank infill, or dark
    # glazing. Darkness by itself is not proof. The upper block must be a
    # single low-texture surface interrupted by at most thin printed letters.
    g=round(max(2,.055*ppm));margin=round(.115*ppm);opts=[]
    for k in range(max(2*g+4,int(.12*len(grey))),min(len(grey)-2*g-3,int(.83*len(grey)))):
        upper=ar[:max(1,k-margin)];lower=ar[min(len(ar)-1,k+margin):]
        if min(len(upper),len(lower))<.35*ppm:continue
        col=np.median(upper.reshape(-1,3),axis=0);solid=np.mean(np.linalg.norm(upper-col,axis=2)<23)
        if solid<.71:continue
        # A color step across most of the width must persist below the edge;
        # lettering, a thin transom bar, or a reflection stripe isn't an infill.
        top=sm[:,max(0,k-margin*2):max(1,k-margin)].mean(1);bot=sm[:,min(len(ar)-2,k+margin):min(len(ar),k+2*margin+1)].mean(1);diff=bot-top
        positive=np.sum(diff>13);negative=np.sum(diff<-13)
        if max(positive,negative)<5 or np.linalg.norm(np.median(lower.reshape(-1,3),axis=0)-col)<29:continue
        edge=max(abs(np.median(sm[:,k+g]-sm[:,k-g])),abs(np.median(sm[:,k+1]-sm[:,k-1])))
        score=edge*np.median(abs(diff))*solid
        if edge<5:continue
        opts.append((score,k,col,solid))
    if not opts:return []
    best=max(opts,key=lambda s:s[0]);score,k,col,solid=best
    if any(abs(q[1]-k)>.25*ppm and q[0]>score*.8 for q in opts):return []
    return {'splitZM':round(H-(t+k-pad)/ppm,3),'color':np.rint(col).astype(int).tolist(),'solidShare':round(float(solid),3),'score':round(float(score),2)}


def has_panel_semantics(reading):
    notes=str(reading.get('notes','')).lower()
    return bool(re.search(r'\b(?:sign\s?(?:board|band)|opaque\s+(?:upper|transom|panel)|solid\s+(?:lunette|transom|tympanum)|boarded\s+(?:lunette|upper|transom)|(?:painted|opaque)\s+infill)\b',notes))


def main():
    ap=argparse.ArgumentParser()
    for k in ('area','profiles','output'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();area=a.area.resolve();pp=json.loads(a.profiles.read_text());sides=load_registered_sides(area/'regen/sides-reg');audit=[];n=0
    for p in pp['profiles']:
        if not has_panel_semantics(p.get('reading',{})):continue
        ops=[(i,o) for i,o in enumerate(p['openings']) if o['style'].get('kind')=='shopfront' and not o['style'].get('upperInfill')]
        if not ops:continue
        ss=resolve_exact_source(p,area,sides)
        if ss['status']!='ok':continue
        sp=sides[p['id'][6:]];piece=sp['piece'];vs=[]
        for v in piece['views']:
            if v.get('incidence',0)<.50 or v.get('visibleFraction',0)<.75:continue
            path=sp['path'].parent/'img'/v['rect'];vs.append((v,path,cv2.imread(str(path))))
        for i,o in ops:
            n+=1;views=[]
            for v,path,im in vs:
                res=samples(im,o['rect'],{**piece,**v})
                if res:views.append({**res,'image':v['image'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            base=next((v for v in views if v['image']==ss['image']),None)
            if not base:continue
            h=p['heightRefM'];z,Z=np.array(o['rect'])[1::2]*h
            agree=[v for v in views if v['image']!=base['image'] and abs(v['splitZM']-base['splitZM'])<min(.24,.075*(Z-z)) and np.linalg.norm(np.array(v['color'])-base['color'])<30]
            if not agree:continue
            group=[base,*agree];col=np.median([q['color'] for q in group],axis=0).round().astype(int);cut=float(np.median([v['splitZM'] for v in group]))
            o['style']['upperInfill']={'fromLeafZ':round((cut-z)/(Z-z),5),'material':'#%02x%02x%02x'%tuple(col),'sourceHashes':[v['sha256'] for v in group],'heightDisagreementM':round(max(abs(v['splitZM']-cut) for v in group),3),'kind':'source-described-opaque-panel'}
            audit.append({'profile':p['id'],'feature':p['match']['featureId'],'opening':i,'views':group})
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'profiles.json').write_text(json.dumps(pp,separators=(',',':')));(a.output/'audit.json').write_text(json.dumps({'tested':n,'admitted':audit},indent=1));print('tested',n,'admitted',len(audit))
if __name__=='__main__':main()
