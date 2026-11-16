"""Keep a changed bay contour only when BOTH photographs also support the
positions of its existing apertures on the new surface. Never invent a window
on the now-perpendicular side, relocate apertures or copy another bay."""
import argparse,json,os,hashlib
from pathlib import Path
import cv2,numpy as np
from pipeline.bays import corners
from regen.reading_provenance import load_registered_sides

def edge_score(img,ops,piece,v,bay,cam,shape):
    ppm,pad=v['ppm'],v['rectPad'];w,h=piece['lengthM'],piece['heightM']
    x,z,X,Z=np.array(bay['rect'])*[w,h,w,h];p=bay['projectionM'];uc,zc,dist=cam
    if p<=0 or dist<p+1:return {}
    _,pts=corners(x,X,p,shape,bay.get('sideAngleDeg',45));pts=np.asarray(pts)
    # At a square bay's front the outermost apertures may approach its corner.
    # Every tested sample is inside the frontage, avoiding depth ambiguities.
    rows={};sob=np.abs(cv2.Sobel(cv2.GaussianBlur(img,(3,3),.8),cv2.CV_32F,1,0,ksize=3))*.125
    for op in ops:
      xx,zz,XX,ZZ=np.array(op['rect'])*[w,h,w,h]
      if op['style']['kind']!='window' or not(x-.01<=xx<XX<=X+.01 and z-.01<=zz<ZZ<=Z+.01):continue
      layer=round((zz+ZZ)/2,1);edges=[]
      for u in (xx,XX):
        dep=p if shape=='square' else np.interp(u,pts[:,0],pts[:,1]);k=(dist-.05)/(dist-dep)
        xxp=pad+ppm*(uc+(u-uc)*k)
        ys=pad+ppm*(h-(zc+(np.linspace(zz+.19*(ZZ-zz),ZZ-.19*(ZZ-zz),19)-zc)*k))
        # Pick one global image shift per storey, shared by every jamb.
        shifts=np.arange(-.30,.301,.025);xs=xxp+ppm*shifts;xxp2=np.broadcast_to(xs[:,None],(len(xs),len(ys)));yy=np.broadcast_to(ys[None,:],xxp2.shape)
        if xxp2.min()<pad+2 or xxp2.max()>img.shape[1]-pad-2 or yy.min()<pad+2 or yy.max()>img.shape[0]-pad-2:return {}
        ss=cv2.remap(sob,xxp2.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR)
        edges.append(np.median(np.clip(ss,0,70),axis=1))
      rows.setdefault(layer,[]).extend(edges)
    out={}
    for layer,e in rows.items():
      if len(e)<4:continue
      vals=np.array(e);quality=np.quantile(vals,.35,axis=0)*(vals>7).mean(axis=0);idx=np.argmax(quality)
      out[layer]={'score':round(float(quality[idx]),3),'coverage':round(float((vals[:,idx]>7).mean()),3),'edges':len(vals),'shiftM':round(float(shifts[idx]),3)}
    return out
def main():
 ap=argparse.ArgumentParser()
 for k in ('area','profiles','scores','output'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();area=a.area.resolve();os.environ['SFW_AREA']=str(area)
 import frontage_crops_v15 as fc
 from regen.register_cameras import apply_registration
 _,fr=fc.load_world();cams=apply_registration(fc.load_cameras(fr),area/'imagery/registered-cameras.json');sides=load_registered_sides(area/'regen/sides-reg')
 d=json.loads(a.profiles.read_text());pp={p['id']:p for p in d['profiles']};reports=[]
 for r in json.loads(a.scores.read_text())['results']:
    if not r['proposed'] or r['proposed']==r['previous']:continue
    p=pp[r['side']];b=p['bays'][r['bay']];row=sides[r['side'][6:]];piece=row['piece'];n=np.array(p['normal']);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]]);vs=[]
    for v in piece['views']:
      if v['image'] not in r['scores'] or not r['scores'][v['image']]:continue
      f=row['path'].parent/'img'/v['rect'];im=cv2.imread(str(f),0);c=cams[v['image']];xy=np.array(c['xy']);cam=[float(xy@u-p['frameU'][0]),fc.ground_at(xy)+fc.EYE_HEIGHT_M+c.get('heightOffsetM',0)-piece['groundZ'],float(xy@n-p['planeV'])]
      before=edge_score(im,p['openings'],piece,v,b,cam,r['previous']);after=edge_score(im,p['openings'],piece,v,b,cam,r['proposed'])
      vs.append({'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'before':before,'after':after,'matches':[k for k,vv in after.items() if k in before and vv['score']>=5 and vv['score']>before[k]['score']*1.22+1 and vv['coverage']>=.70]})
    accepted=len(vs)>=2 and len(set.intersection(*(set(v['matches']) for v in vs)))>=2
    if accepted:b.update(shape=r['proposed'],shapeEvidence={'sourceHashes':[v['sha256'] for v in vs],'basis':'multi-view contour and aperture edges; preserved aperture x/z bounds'})
    reports.append({k:r[k] for k in ('side','feature','bay','previous','proposed')}|{'accepted':accepted,'views':vs})
 a.output.mkdir(exist_ok=True,parents=True);(a.output/'profiles.json').write_text(json.dumps(d,separators=(',',':')));(a.output/'audit.json').write_text(json.dumps(reports,indent=1));print([(r['side'],r['accepted'],[q['matches'] for q in r['views']]) for r in reports])
if __name__=='__main__':main()
