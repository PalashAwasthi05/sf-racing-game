"""Recover repeated opaque garage door panels without guessing a grid.
Only two-view, boxed, repeated contours inside an already observed closed leaf
qualify. Never infer glass, occluded panels, missing rows, or the door state.
Run from work/ with PYTHONPATH=. Each output layout retains source crop hashes.
"""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source


def panels(im,rect,frame):
    w,h=frame['lengthM'],frame['heightM'];ppm,pad=frame['ppm'],frame['rectPad']
    x,z,X,Z=rect; box=np.rint([pad+x*w*ppm,pad+(1-Z)*h*ppm,pad+X*w*ppm,pad+(1-z)*h*ppm]).astype(int)
    a,b,c,d=box
    if not (0<=a<c<=im.shape[1] and 0<=b<d<=im.shape[0] and c-a>=40 and d-b>=30):return []
    crop=im[b:d,a:c];gr=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);found=[]
    for low,high in [(12,34),(25,60),(45,110)]:
        edges=cv2.Canny(cv2.GaussianBlur(gr,(3,3),.65),low,high)
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for ct in contours:
            p=cv2.approxPolyDP(ct,max(1.,.02*cv2.arcLength(ct,True)),True)
            if len(p)!=4 or not cv2.isContourConvex(p):continue
            l,t,dw,dh=cv2.boundingRect(p);p=p[:,0,:].astype(float)
            if min(l,t)<2 or l+dw>c-a-2 or t+dh>d-b-2 or not (max(7,.17*ppm)<dw<.58*(c-a) and max(5,.1*ppm)<dh<.38*(d-b)):continue
            if not .65<dw/dh<7.5 or cv2.contourArea(ct)<.80*(dw-1)*(dh-1):continue
            delta=np.abs(p-np.roll(p,1,axis=0))
            if np.any(delta.min(1)>np.maximum(1.8,.13*delta.max(1))):continue
            # No pane semantics: the annotated leaf must already be opaque. Reject
            # a high-variance interior (glass, cars, deep shadow/objects).
            yy=gr[t+2:t+dh-2,l+2:l+dw-2]
            if not yy.size or np.percentile(yy,90)-np.percentile(yy,10)>47:continue
            r=np.array([l,t,l+dw,t+dh],float)
            if any(np.max(abs(r-q))<=max(2.5,.11*dh) for q in found):continue
            found.append(r)
    return found,box


def match_panels(a,b,ppm):
    if len(a)<4 or len(b)<4:return []
    aa,bb=np.array(a),np.array(b)
    ac=(aa[:,:2]+aa[:,2:])/2;bc=(bb[:,:2]+bb[:,2:])/2
    aw=aa[:,2:]-aa[:,:2];bw=bb[:,2:]-bb[:,:2]
    # Objects at a recess may shift consistently between rectified images.
    # Limit motion, require matching extents and an unambiguous shared shift.
    dif=bc[None]-ac[:,None];ratio=np.maximum(aw[:,None]/bw[None],bw[None]/aw[:,None]);ok=(ratio<1.23).all(2)&(np.linalg.norm(dif,axis=2)<.38*ppm)
    opts=[]
    for shift in np.unique(np.round(dif[ok]/2)*2,axis=0):
        dist=np.linalg.norm(dif-shift,axis=2);ij=linear_sum_assignment(np.where(ok,dist,1e5))
        pairs=[(i,j) for i,j in zip(*ij) if ok[i,j] and dist[i,j]<max(2.,.055*ppm)]
        if len(pairs)>=4:opts.append((len(pairs),-np.median([dist[i,j] for i,j in pairs]),shift,pairs))
    if not opts:return []
    opts.sort(key=lambda x:(-x[0],-x[1]));n,err,shift,pairs=opts[0]
    if any(k>=n-1 and np.linalg.norm(s-shift)>.13*ppm for k,e,s,pp in opts):return []
    # Repetition must occur in two dimensions; don't assume all panels exist.
    keep=[]
    for i,j in pairs:
        siblings=[k for k,_ in pairs if np.all(np.maximum(aw[i]/aw[k],aw[k]/aw[i])<1.23)]
        row=any(abs(ac[k,1]-ac[i,1])<.18*aw[i,1] and abs(ac[k,0]-ac[i,0])>.85*aw[i,0] for k in siblings)
        col=any(abs(ac[k,0]-ac[i,0])<.18*aw[i,0] and abs(ac[k,1]-ac[i,1])>.85*aw[i,1] for k in siblings)
        if row and col:keep.append((i,j))
    # Nested duplicate contours would make a raised panel sit on itself.
    final=[]
    for i,j in sorted(keep,key=lambda q:-np.prod(aw[q[0]])):
        if all(np.prod(np.maximum(0,np.minimum(aa[i,2:],aa[k,2:])-np.maximum(aa[i,:2],aa[k,:2])))<.15*np.prod(aw[i]) for k,h in final):final.append((i,j))
    return final if len(final)>=4 else []


def main():
    ap=argparse.ArgumentParser()
    for k in ('area','profiles','output'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();area=a.area.resolve();src=json.loads(a.profiles.read_text());sides=load_registered_sides(area/'regen/sides-reg');out=[];tested=0
    for p in src['profiles']:
        doors=[(i,o) for i,o in enumerate(p.get('openings',[])) if o['style'].get('kind') in ('rollup','garage') and o['style'].get('doorKind')=='panel' and not o['style'].get('doorCells') and not o['style'].get('doorConstruction')]
        if not doors:continue
        s=resolve_exact_source(p,area,sides)
        if s['status']!='ok':continue
        prov=p['readingProvenance'];sp=sides[p['id'][6:]];piece=sp['piece'];rp=Path(prov['sourceReading']['path']);r=json.loads((area/rp if prov.get('pathBase')=='area' else rp).read_text());views=[]
        for v in piece['views']:
            if v.get('incidence',0)<.60 or v.get('visibleFraction',0)<.72 or v.get('registered') not in ('bias','skyline'):continue
            f=sp['path'].parent/'img'/v['rect'];views.append((v,f,cv2.imread(str(f))))
        if len({v['image'] for v,f,im in views})<2:continue
        for i,o in doors:
            ix=o.get('sourceIndex',-1)
            if not isinstance(ix,int) or not 0<=ix<len(r.get('openings',[])) or r['openings'][ix].get('door',{}).get('kind')!='panel':continue
            tested+=1;qs=[]
            for v,f,im in views:
                q=panels(im,o['rect'],{**piece,**v})
                if q and len(q[0])>=4:qs.append((v,f,*q))
            primary=next((q for q in qs if q[0]['image']==s['image']),None)
            if not primary:continue
            v,f,aa,box=primary
            for v2,f2,bb,box2 in qs:
                if v2['image']==v['image'] or not np.array_equal(box,box2):continue
                pairs=match_panels(aa,bb,v['ppm'])
                if not pairs:continue
                w,h=box[2:]-box[:2]; cells=[]
                for k,_ in pairs:
                    x,y,X,Y=aa[k];cells.append({'kind':'panel','rect':np.round([x/w,1-Y/h,X/w,1-y/h],5).tolist()})
                out.append({'profile':p['id'],'feature':p['match']['featureId'],'opening':i,'cells':cells,'boxPx':box.tolist(),'primary':str(f),'secondary':str(f2),'sourceSha256':[hashlib.sha256(ff.read_bytes()).hexdigest() for ff in (f,f2)],'panelBoxesPx':[[aa[k].tolist(),bb[j].tolist()] for k,j in pairs]});break
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'audit.json').write_text(json.dumps({'tested':tested,'accepted':out,'profilesSha256':hashlib.sha256(a.profiles.read_bytes()).hexdigest()},indent=1))
    print('Two-view panel layouts:',len(out),'/',tested,'doors;',sum(len(r['cells']) for r in out),'cells')
if __name__=='__main__':main()
