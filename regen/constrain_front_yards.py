"""Keep facade-attached private planting and garden barriers on the building side of a carriageway.

Their source distance from the wall is a photo estimate. Clip inferred extents
in the original facade frame; never move a hedge across the road or fabricate a
replacement. Public mapped street trees, islands and elevated foliage are separate.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,box,shape
from shapely.ops import unary_union
from shapely.affinity import affine_transform
from shapely.strtree import STRtree


def guarded_local_extent(x0,x1,off,depth,obstacles,*,pad=.15,fixed_depth=False):
    if x1<=x0 or depth<=0 or off<0:raise ValueError('invalid local extent')
    if obstacles.is_empty:return (x0,x1,off,depth)
    # Project every obstacle column back to the first curb encountered outward
    # from the wall. Thus even a planting estimate entirely across a narrow
    # street is blocked, without changing a valid garden on this side of it.
    corridor=box(x0-pad,0.,x1+pad,off+depth+pad)
    veto=obstacles.buffer(pad).intersection(corridor)
    if veto.is_empty:return (x0,x1,off,depth)
    breaks={x0,x1}
    for poly in getattr(veto,'geoms',[veto]):
        if poly.geom_type=='Polygon':
            for x,_ in poly.exterior.coords:
                if x0<x<x1:breaks.add(float(x))
    # Coalesce nearly coincident vertices from buffered curves; solve for one
    # retained rectangle instead of emitting many little planters or end piers.
    xs=[x0]+[x for x in sorted(breaks) if x0+.08<x<x1-.08]+[x1];xs=np.unique(np.round(xs,6));caps=[]
    for l,r in zip(xs,xs[1:]):
        meet=veto.intersection(box(l+.000001,0.,r-.000001,off+depth+pad)) if r-l>.000002 else veto
        caps.append(min(depth,meet.bounds[1]-off) if not meet.is_empty else depth)
    best=None;score=0.
    for i,l in enumerate(xs[:-1]):
        h=depth
        for j in range(i,len(caps)):
            h=min(h,caps[j]);w=xs[j+1]-l
            if h<(depth-1e-6 if fixed_depth else min(depth,.18)):break
            if w<.30:continue
            area=w*h
            if area>score:score=area;best=(float(l),float(xs[j+1]),off,float(h))
    if best is None or score<(x1-x0)*depth*.18:return None
    return best


def constrain(profile,tree,polys):
    p=copy.deepcopy(profile);events=[]
    if not p.get('frameU') or p.get('planeV') is None:return p,events
    n=np.array(p['normal'],float);n/=np.linalg.norm(n);u=np.array([-n[1],n[0]]);left,right=p['frameU'];W=right-left
    if W<=0:return p,events
    o=n*float(p['planeV'])+u*left
    for coll in ('hedges','planting','streetEdge'):
        kept=[]
        for i,obj in enumerate(p.get(coll,[])):
            kind=obj.get('kind',{'hedges':'hedge','planting':'shrub','streetEdge':'wall'}[coll]);off=float(obj.get('offsetM',.5 if coll=='hedges' else .3 if coll=='planting' else 0.));d=float(obj.get('depthM',{'hedges':1.2,'planting':.8,'streetEdge':1.}[coll]))
            # Walls and fences have fixed thickness in the renderer: they may
            # shorten along their span, but cannot be thinned by editing depthM.
            fixed=kind in ('wall','fence','gate')
            if fixed:d=(.40 if obj.get('pierWidthM',0.)>.1 else .33) if kind=='wall' else .035
            if kind not in ('wall','fence','gate','shrub','planter','hedge') or float(obj.get('baseM',0.))>2.4 or not obj.get('x') or off<0 or d<=0:
                kept.append(obj);continue
            x0,x1=np.array(obj['x'])*W
            if x1<=x0:kept.append(obj);continue
            q=Polygon([o+a*u+b*n for a,b in [(x0-.2,-.2),(x1+.2,-.2),(x1+.2,off+d+.2),(x0-.2,off+d+.2)]])
            ids=tree.query(q,predicate='intersects');
            if len(ids)==0:kept.append(obj);continue
            obstacle=unary_union([polys[j].intersection(q) for j in ids]);local=affine_transform(obstacle,[*u,*n,-float(o@u),-float(o@n)])
            ret=guarded_local_extent(x0,x1,off,d,local,pad=.20,fixed_depth=fixed)
            if ret is not None and np.allclose(ret,(x0,x1,off,d),atol=1e-5):kept.append(obj);continue
            events.append({'collection':coll,'index':i,'kind':kind,'old':[x0,x1,off,d],'new':ret,'reason':'mapped-road-cycle-or-crossing-between-frontage-and-inferred-span'})
            if ret is None:continue
            obj['x']=[ret[0]/W,ret[1]/W];
            if not fixed:obj['depthM']=ret[3]
            kept.append(obj)
        if coll in p:p[coll]=kept
    return p,events


def main():
    ap=argparse.ArgumentParser();
    for k in ('profiles','street-plan','output','audit'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();src=json.loads(a.profiles.read_text());plan=json.loads(a.street_plan.read_text());polys=[]
    for k in ('road','cycle','crossings'):
        g=shape(plan[k]);polys.extend(p for p in getattr(g,'geoms',[g]) if p.geom_type=='Polygon' and not p.is_empty)
    tree=STRtree(polys);out=[];audit=[]
    for p in src['profiles']:
        pp,ev=constrain(p,tree,polys);out.append(pp)
        if ev:audit.append({'side':p['id'],'building':p['match']['featureId'],'changes':ev})
    for f in (a.output,a.audit):f.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({**src,'profiles':out},separators=(',',':'))+'\n');a.audit.write_text(json.dumps({'profilesChecked':len(out),'streetPlanSource':str(a.street_plan),'streetPlanSha256':hashlib.sha256(a.street_plan.read_bytes()).hexdigest(),'inputProfilesSha256':hashlib.sha256(a.profiles.read_bytes()).hexdigest(),'changed':audit},indent=1)+'\n');print(len(out),'profiles;',len(audit),'changed;',sum(len(q['changes']) for q in audit),'planting/barrier extents adjusted or rejected')
if __name__=='__main__':main()
