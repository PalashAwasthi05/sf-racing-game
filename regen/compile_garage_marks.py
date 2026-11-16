"""Transfer only repeatable ink/shadow runs of a measured door joint, not a full row."""
import argparse,hashlib,json,re
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import binary_closing,binary_opening,gaussian_filter1d,label
from regen.reading_provenance import load_registered_sides


def samples(v,piece):
    fname=Path(v['rect'])
    if hashlib.sha256(fname.read_bytes()).hexdigest()!=v['sha256']:raise ValueError('door source changed')
    view=next(w for w in piece['views'] if w['image']==v['image']);ppm,pad=view['ppm'],view['rectPad'];l,t,r,b=v['metricROI']
    im=gaussian_filter1d(cv2.imread(str(fname),0).astype(float),.65,axis=0);rows=[]
    for z in v['lineZM']:
        y=round(pad+(piece['heightM']-z)*ppm)
        if not 2<=y<im.shape[0]-2:raise ValueError('observed row outside crop')
        rows.append(((im[y-2]+im[y+2])/2-im[y])[l:r])
    return np.array(rows),ppm,pad


def agree_runs(a,b,ppm):
    if a.shape!=b.shape or len(a)<3:return []
    n=a.shape[1];window=round(min(.32*n,.4*ppm));scores=[]
    for shift in range(-window,window+1):
        l=max(0,-shift);r=min(n,n-shift);aa=a[:,l:r].ravel();bb=b[:,l+shift:r+shift].ravel()
        if min(np.std(aa),np.std(bb))<2:continue
        scores.append((float(np.corrcoef(aa,bb)[0,1]),shift))
    if not scores:return []
    score,shift=max(scores)
    if score<.7 or any(s>score-.09 and abs(k-shift)>max(3,.10*ppm) for s,k in scores):return []
    l=max(0,-shift);r=min(n,n-shift);out=[]
    for i,(ar,br) in enumerate(zip(a[:,l:r],b[:,l+shift:r+shift])):
        joint=(ar>max(3,np.percentile(ar,80)*.30))&(br>max(3,np.percentile(br,80)*.30))
        tags,count=label(binary_opening(binary_closing(joint,np.ones(3)),np.ones(4)))
        for j in range(1,count+1):
            ix=np.flatnonzero(tags==j)
            if len(ix)<.18*ppm:continue
            out.append((i,int(l+ix[0]),int(l+ix[-1]+1),float(np.median(np.minimum(ar[ix],br[ix])))))
    return out


def main():
    ap=argparse.ArgumentParser()
    for k in ('area','profiles','courses','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();out=json.loads(a.profiles.read_text());pmap={p['id']:p for p in out['profiles']};sides=load_registered_sides(a.area/'regen/sides-reg');audit=[]
    for c in json.loads(a.courses.read_text())['candidates']:
        p=pmap[c['profile']];o=p['openings'][c['opening']];s=o['style']
        if o['rect']!=c['rect'] or s.get('doorKind')!='panel' or s.get('doorCells') or s.get('doorConstruction') or not re.fullmatch('#[0-9a-fA-F]{6}',s.get('material','')):continue
        pri=c['primary'];sec=c['witnesses'][0];piece=sides[c['profile'][6:]]['piece'];aa,ppm,pad=samples(pri,piece);bb,ppm2,_=samples(sec,piece)
        if ppm!=ppm2:continue
        rows=agree_runs(aa,bb,ppm);x,z,X,Z=o['rect'];w=piece['lengthM'];h=piece['heightM'];l=pri['metricROI'][0];marks=[]
        for i,L,R,contrast in rows:
            u0=((l+L-pad)/ppm/w-x)/(X-x);u1=((l+R-pad)/ppm/w-x)/(X-x);zz=(pri['lineZM'][i]/h-z)/(Z-z)
            if not(0<u0<u1<1 and 0<zz<1):continue
            # One texel is the finite source resolution; the strip is a tonal
            # mark, not a physical claim about the profile of the joinery.
            marks.append({'u':[round(u0,5),round(u1,5)],'z':round(zz,5),'widthM':round(1/ppm,4),'darkening':round(min(.3,contrast/128),3)})
        if len(set(m['z'] for m in marks))<3:continue
        s['observedLeafMarks']={'marks':marks,'sourceHashes':[pri['sha256'],sec['sha256']],'wholePanelLayoutKnown':False}
        audit.append({'profile':p['id'],'feature':p['match']['featureId'],'opening':c['opening'],'marks':len(marks)})
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'profiles.json').write_text(json.dumps(out,separators=(',',':')));(a.output/'audit.json').write_text(json.dumps(audit,indent=1));print(audit)
if __name__=='__main__':main()
