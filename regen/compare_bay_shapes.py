"""Test the cross-section of a projecting bay against curved/straight trim in
its actual photographs. Use the calibrated perspective on the rectified wall:
a horizontal course on an outcrop bows in that image. No building-specific
dimensions or shapes. Agreement across a second camera AND another height is
required before a new shape can replace the original reading.
"""
import argparse,hashlib,json,os,re
from pathlib import Path
import cv2
import numpy as np
from pipeline.bays import corners
from regen.reading_provenance import load_registered_sides
from regen.audit_aperture_repair_targets import resolve_exact_source

def candidates(img,piece,view,camera,bay):
    w,h=piece['lengthM'],piece['heightM'];x,z,X,Z=np.array(bay['rect'])*[w,h,w,h];p=bay.get('projectionM',0);ppm,pad=view['ppm'],view['rectPad']
    if min(X-x,p,h)<=0 or (X-x)*ppm<70:return {}
    uc,zc,dist=camera
    gray=cv2.GaussianBlur(img,(3,3),.7).astype('float32')
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    levels=[('top',Z)]+[('course'+str(i),s*h) for i,s in enumerate(bay.get('storeys',[])) if z+.4<s*h<Z-.4];out={}
    for level,zz in levels:
        if not 0<zz<h-.25:continue
        score={}
        for shape in ('square','angled','curved'):
            _,pts=corners(x,X,p,shape,bay.get('sideAngleDeg',45.));pts=np.asarray(pts,float)
            # Sample the width, not its polygon vertex count, so a curved shape
            # doesn't get more votes just because it has more triangles.
            if shape=='square':uu=np.linspace(x+.04,X-.04,66);depth=np.full_like(uu,p)
            else:uu=np.linspace(x+.04,X-.04,66);depth=np.interp(uu,pts[:,0],pts[:,1])
            if np.max(depth)>=dist-1:continue
            k=(dist-.05)/(dist-depth)
            xs=pad+ppm*(uc+(uu-uc)*k)
            if not np.all((xs>pad+max(3,.06*ppm))&(xs<img.shape[1]-pad-max(3,.06*ppm))):continue
            # Offset each candidate by the identical allowed elevation error;
            # no independent per-pixel alignment or manual tracing.
            offs=np.arange(-.55,.551,.025);ys=pad+ppm*(h-(zc+(zz+offs[:,None]-zc)*k[None,:]))
            xs=np.broadcast_to(xs,ys.shape);valid=(ys>pad+2)&(ys<img.shape[0]-pad-2)
            dx=np.gradient(xs,axis=1);dy=np.gradient(ys,axis=1);length=np.maximum(.1,np.hypot(dx,dy))
            mx=xs.astype('float32');my=ys.astype('float32');gxv=cv2.remap(gx,mx,my,cv2.INTER_LINEAR);gyv=cv2.remap(gy,mx,my,cv2.INTER_LINEAR);grad=(-dy*gxv+dx*gyv)/length
            # Both light and dark course edges are valid; polarity, however,
            # must be consistent across the frontage rather than repeated noise.
            opts=[]
            for pol in (-1,1):
                t=np.where(valid,pol*grad,0);cov=(t>4).mean(axis=1)
                shoulder=np.minimum((t[:,:14]>4).mean(axis=1),(t[:,-14:]>4).mean(axis=1))
                val=np.quantile(np.clip(t,0,50),.35,axis=1)*cov
                m=int(np.argmax(val));opts.append((round(float(val[m]),3),float(cov[m]),round(float(offs[m]),3),pol,round(float(shoulder[m]),3)))
            best=max(opts);score[shape]=dict(score=best[0],coverage=round(best[1],3),offset=best[2],polarity=best[3],shoulders=best[4])
        if len(score)==3:out[level]=score
    return out

def choose(views):
    votes={};qualified=[]
    for view,data in views.items():
        for lv,s in data.items():
            ranks=sorted(s,key=lambda k:s[k]['score'],reverse=True);a,b=ranks[:2]
            if s[a]['shoulders']<.60 or s[a]['coverage']<.73 or s[a]['score']<5 or s[a]['score']<1.35*s[b]['score']+1.4:continue
            votes.setdefault((a,lv),[]).append(view)
    for shape in ('square','angled','curved'):
        levels=[k[1] for k,v in votes.items() if k[0]==shape and len(set(v))>=2]
        if len(levels)>=2 and not any(k[0]!=shape and len(set(v))>=2 for k,v in votes.items()):qualified.append(shape)
    return qualified[0] if len(qualified)==1 else None

def main():
    ap=argparse.ArgumentParser()
    for k in ('area','profiles','output'):ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();area=a.area.resolve();os.environ['SFW_AREA']=str(area)
    import frontage_crops_v15 as fc
    from regen.register_cameras import apply_registration
    _,fr=fc.load_world();reg=area/'imagery/registered-cameras.json';cams=apply_registration(fc.load_cameras(fr),reg);sides=load_registered_sides(area/'regen/sides-reg')
    out=[];tested=0;pp=json.loads(a.profiles.read_text())
    for p in pp['profiles']:
        if not p.get('bays'):continue
        src=resolve_exact_source(p,area,sides)
        if src['status']!='ok':continue
        sid=sides[p['id'][6:]];piece=sid['piece'];n=np.array(p['normal']);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]]);views=[]
        for v in piece['views']:
            if v.get('incidence',0)<.55 or v.get('visibleFraction',0)<.7 or v.get('registered') not in ('skyline','bias'):continue
            c=cams[v['image']];xy=np.array(c['xy']);eye=fc.ground_at(xy)+fc.EYE_HEIGHT_M+c.get('heightOffsetM',0)
            cam=[float(xy@u-p['frameU'][0]),float(eye-piece['groundZ']),float(xy@n-p['planeV'])]
            f=sid['path'].parent/'img'/v['rect'];views.append((v,cam,cv2.imread(str(f),0),str(f)))
        if len(views)<2:continue
        for i,b in enumerate(p['bays']):
            results={v['image']:candidates(im,piece,v,c,b) for v,c,im,f in views};tested+=1;shape=choose(results)
            out.append({'side':p['id'],'bay':i,'previous':b.get('shape'),'proposed':shape,'feature':p['match']['featureId'],'imageReferences':[f for v,c,im,f in views],'scores':results})
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'audit.json').write_text(json.dumps({'tested':tested,'results':out,'applied':False},separators=(',',':')))
    from collections import Counter
    print(tested,Counter((r['previous'],r['proposed']) for r in out if r['proposed']))
if __name__=='__main__':main()
