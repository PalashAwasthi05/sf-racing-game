"""A clear opaque panel provides a search height/color for its siblings, never
a license to fill them. Require an actual persistent boundary in each receiver
in two registered source views. Narrow independent strips tolerate occlusion."""
import argparse, hashlib,json
from pathlib import Path
import numpy as np
import cv2
from regen.reading_provenance import load_registered_sides

def seam(im,frame,rect,zref,color):
    x,z,X,Z=np.array(rect)*np.array([frame['lengthM'],frame['heightM']]*2)
    ppm,pad,H=frame['ppm'],frame['rectPad'],frame['heightM']
    if not z+.35<zref<Z-.65 or (X-x)*ppm<40:return
    L=round(pad+(x+.08*(X-x))*ppm);R=round(pad+(X-.08*(X-x))*ppm)
    if L<=0 or R>=im.shape[1]:return
    ar=im[:,L:R].astype(float);groups=np.array_split(ar,8,axis=1);opts=[]
    for yy in range(round(pad+(H-zref-.32)*ppm),round(pad+(H-zref+.32)*ppm)+1):
        a,b=round(.035*ppm)+1,round(.23*ppm)+1
        if yy-2*b<0 or yy+2*b>ar.shape[0]:continue
        yes=[];strength=[];shade=[]
        for j,g in enumerate(groups):
            top=g[yy-2*b:yy-a];l=np.median(top.reshape(-1,3),axis=0)
            bottom=np.median(g[yy+a:yy+b].reshape(-1,3),axis=0);dist=float(np.linalg.norm(bottom-l))
            gap=float(np.linalg.norm(np.median(g[yy+a:yy+a+2],axis=(0,1))-np.median(g[yy-a-2:yy-a],axis=(0,1))))
            if (np.linalg.norm(l-color)<34 and np.mean(np.linalg.norm(top-l,axis=2)<24)>.76
                and dist>=20 and gap>=.50*dist and np.linalg.norm(np.median(g[yy+b:yy+2*b],axis=(0,1))-l)>.60*dist):
                yes.append(j);strength.append(min(dist,100)*min(1,gap/dist));shade.append(l)
        if len(yes)>=3 and max(yes)-min(yes)>=3:
            opts.append((sum(strength),yy,yes,np.median(shade,axis=0)))
    if not opts:return
    best=max(opts,key=lambda q:q[0]);score,yy,yes,shade=best
    if any(abs(q[1]-yy)>.17*ppm and q[0]>.85*score for q in opts):return
    return {'splitZM':H-(yy-pad)/ppm,'strips':yes,'color':np.rint(shade).astype(int).tolist()}
def extend(pp,area):
    sides=load_registered_sides(area/'regen/sides-reg');audit=[]
    for p in pp['profiles']:
        ops=p.get('openings',[]);seeds=[o for o in ops if o['style'].get('upperInfill',{}).get('kind')=='source-described-opaque-panel']
        if not seeds:continue
        row=sides[p['id'][6:]];piece=row['piece'];views=[]
        for v in piece['views']:
            if v.get('incidence',0)<.50 or v.get('visibleFraction',0)<.75:continue
            path=row['path'].parent/'img'/v['rect'];views.append((v,cv2.imread(str(path))[:,:,::-1],hashlib.sha256(path.read_bytes()).hexdigest()))
        for i,o in enumerate(ops):
            if (o in seeds or o['style'].get('kind') not in ('rollup','shopfront')
                or o['style'].get('doorKind')=='open'):continue
            for s in seeds:
                if s['style'].get('head')!=o['style'].get('head'):continue
                x,z,X,Z=o['rect'];sx,sz,sX,sZ=s['rect'];H=p['heightRefM'];W=piece['lengthM']
                if max(abs(z-sz),abs(Z-sZ))*H>.28 or abs((X-x)/(sX-sx)-1)>.33:continue
                exemplar=s['style']['upperInfill'];cut=(sz+exemplar['fromLeafZ']*(sZ-sz))*H
                col=[int(exemplar['material'][k:k+2],16) for k in (1,3,5)]
                fits=[dict(r,image=v['image'],sha256=sha) for v,im,sha in views if (r:=seam(im,{**piece,**v},o['rect'],cut,col))]
                if len({r['image'] for r in fits})<2 or max(len(r['strips']) for r in fits)<5:continue
                zz=[r['splitZM'] for r in fits];colors=np.array([r['color'] for r in fits])
                if np.ptp(zz)>.19 or np.max(np.linalg.norm(colors-colors.mean(0),axis=1))>17:continue
                c=np.median(colors,axis=0).round().astype(int)
                o['style']['upperInfill']={'kind':'source-described-opaque-panel','method':'two-camera-independent-strips','material':'#%02x%02x%02x'%tuple(c),'fromLeafZ':round((np.median(zz)/H-z)/(Z-z),5),'sourceHashes':[r['sha256'] for r in fits]}
                audit.append(dict(side=p['id'],opening=i,views=fits));break
    return audit
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('area','profiles','output'):a.add_argument('--'+k,type=Path,required=True)
    args=a.parse_args();pp=json.loads(args.profiles.read_text());d=extend(pp,args.area);args.output.mkdir(exist_ok=True,parents=True);(args.output/'profiles.json').write_text(json.dumps(pp,separators=(',',':')));(args.output/'audit.json').write_text(json.dumps(d,indent=1));print('additional verified panels:',[(q['side'],q['opening']) for q in d])
