"""Identify real horizontal leaf joints in two independent photographs of a closed door.
No location/style defaults. Annotated absolute wall-opening frame is only a search
region; a parked vehicle, cropping, or conflicting repeated pattern causes abstention.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source


def detect(im,rect,frame):
    H,W,ppm,pad=frame['heightM'],frame['lengthM'],frame['ppm'],frame['rectPad'];x0,z0,x1,z1=rect
    if min(z0,x0)<-.015 or max(x1,z1)>1.015:return None
    lo=round(pad+(x0+.10*(x1-x0))*W*ppm);hi=round(pad+(x1-.10*(x1-x0))*W*ppm)
    ya=round(pad+(1-z1+.07*(z1-z0))*H*ppm);yb=round(pad+(1-z0-.07*(z1-z0))*H*ppm)
    if min(ya,lo)<pad or hi>im.width-pad+1 or yb>im.height-pad+1 or hi-lo<40 or yb-ya<32:return None
    a=np.asarray(im.crop((lo,ya,hi,yb)).convert('L'),float);smooth=gaussian_filter1d(a,.75,axis=0)
    # Score a narrow dark seam, not the silhouette of two different objects.
    dd=(smooth[4:]+smooth[:-4])/2-smooth[2:-2];strips=np.stack([np.median(s,axis=1) for s in np.array_split(dd,7,axis=1)]);val=np.median(strips,axis=0)
    peaks,_=find_peaks(val,height=max(2.6,float(np.median(abs(val-np.median(val)))*3)),distance=3,prominence=2.5)
    supported=[int(i) for i in peaks if np.sum(strips[:,max(0,i-1):i+2].max(1)>max(2.,val[i]*.40))>=6]
    if len(supported)<3:return None
    gaps=np.diff(supported);pitch=float(np.median(gaps));spacing=pitch/ppm;u=np.array(supported)+ya+2
    if not .052<=spacing<=.65 or np.max(abs(gaps-pitch))>max(1.5,pitch*.17) or np.ptp(u)<.50*(yb-ya):return None
    if spacing<.18 and len(supported)<7:return None
    # Wider joints have to be measured individually. Large recessed cassettes,
    # windows and a repeated pane grid are not the same as horizontal slats.
    dx=np.nanmedian(abs(np.diff(a,axis=1)),axis=0);has_stile=np.sum(dx>max(10.,np.median(dx)*5))>2
    if has_stile:return None
    return {'pitchM':round(spacing,4),'lineZM':np.round(H-(u-pad)/ppm,3).tolist(),'metricROI':[lo,ya,hi,yb],'kind':'close-horizontal-courses' if spacing<.18 else 'wide-horizontal-courses','contrast':round(float(np.median(val[supported])),2)}


def agreement(a,b,height):
    if not a or not b or a['kind']!=b['kind'] or abs(a['pitchM']-b['pitchM'])>.1*min(a['pitchM'],b['pitchM']):return False
    aa=np.asarray(a['lineZM']);bb=np.asarray(b['lineZM']);d=abs(aa[:,None]-bb)
    tol=max(.035,min(.09,.22*a['pitchM']));return bool((d.min(axis=0)<tol).mean()>=.74 and (d.min(axis=1)<tol).mean()>=.74 and height>1.)


def main():
    ap=argparse.ArgumentParser();
    for k in ('area','profiles','output'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();area=a.area.resolve();pp=json.loads(a.profiles.read_text());sides=load_registered_sides(area/'regen/sides-reg');out=[];tested=0
    for p in pp['profiles']:
        candidates=[(i,o) for i,o in enumerate(p.get('openings',[])) if o['style'].get('kind') in ('rollup','garage') and o['style'].get('doorKind','panel')=='panel' and not o['style'].get('doorCells') and not o['style'].get('doorConstruction')]
        if not candidates:continue
        s=resolve_exact_source(p,area,sides)
        if s['status']!='ok':continue
        prov=p['readingProvenance'];sp=sides[p['id'][6:]];piece=sp['piece'];rp=Path(prov['sourceReading']['path']);r=json.loads((area/rp if prov.get('pathBase')=='area' else rp).read_text());views=[]
        for v in piece['views']:
            if v.get('incidence',0)<.65 or v.get('visibleFraction',0)<.73 or v.get('registered') not in ('bias','skyline'):continue
            f=sp['path'].parent/'img'/v['rect'];views.append((v,f,Image.open(f).convert('L')))
        if len({v['image'] for v,_,im in views})<2:continue
        for i,o in candidates:
            ix=o.get('sourceIndex',-1)
            if not isinstance(ix,int) or not 0<=ix<len(r.get('openings',[])):continue
            observed=r['openings'][ix]
            if observed.get('kind') not in ('garage','rollup') or observed.get('door',{}).get('kind')!='panel':continue
            tested+=1;qs=[]
            for v,f,im in views:
                q=detect(im,o['rect'],{**piece,**v});
                if q:qs.append({**q,'image':v['image'],'rect':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
            primary=next((q for q in qs if q['image']==s['image']),None)
            witnesses=[q for q in qs if q is not primary and agreement(primary,q,p['heightRefM']*(o['rect'][3]-o['rect'][1]))]
            if primary and witnesses:out.append({'profile':p['id'],'feature':p['match']['featureId'],'opening':i,'rect':o['rect'],'primary':primary,'witnesses':witnesses})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'profiles':len(pp['profiles']),'testedClosedDoors':tested,'candidateCount':len(out),'candidates':out,'productionApplied':False},indent=1));print(tested,len(out))
if __name__=='__main__':main()
