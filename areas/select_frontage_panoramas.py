"""Pick additional street panoramas by visible wall coverage and separation between viewpoints."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Polygon,shape
from shapely.geometry.polygon import orient

def select(area):
    os.environ["SFW_AREA"]=str(area.resolve())
    import frontage_crops_v15 as fc
    data=area/"imagery"
    region=json.loads((area/"config.json").read_text())["region"]
    center=region["center"];frame=fc.LocalFrame(center["lon"],center["lat"])
    fp=fc.load_footprints(frame)
    occ=fc.Occluders(fp)
    all_photos=[p for p in json.loads((data/"coverage.json").read_text())["images"] if p.get('is_pano') and p.get('captured_at',0)>=1672531200000]
    ll=np.asarray([p["tile_location"] for p in all_photos])
    in_area=contains_xy(shape(region['contextGeometry']),ll[:,0],ll[:,1])
    pics=[p for p,k in zip(all_photos,in_area) if k]
    xy=np.column_stack(frame.local(*ll[in_area].T));tree=cKDTree(xy)
    index=json.loads((data/'source-index-registered.json').read_text()) if (data/'source-index-registered.json').exists() else {}
    cached={k.partition(':')[2] for k in index};chosen=set()
    for bid, footprint in fp.items():
        if not bid.startswith("sf_"):continue
        for poly in getattr(footprint,'geoms',[footprint]):
            if not isinstance(poly,Polygon):continue
            coords=np.asarray(orient(poly.simplify(.4),sign=1).exterior.coords)
            for a,b in zip(coords,coords[1:]):
                delta=b-a;length=np.linalg.norm(delta)
                if length<9:continue
                mid=(a+b)*.5;normal=np.array([delta[1],-delta[0]])/length
                ids=tree.query_ball_point(mid,52)
                wall=dict(midpoint=mid,normal=normal,lengthM=min(length*.8,26))
                ranked=[]
                for i in ids:
                    dxy=xy[i]-mid;d=np.linalg.norm(dxy)
                    if d<8 or np.dot(dxy,normal)/d<.69:continue
                    p=pics[i];score=fc.distance_preference(d)*float(np.dot(dxy,normal)/d)
                    key=(int(p['captured_at'])//(180*86400000), int(str(p["id"]) in cached or str(p["id"]) in chosen),score)
                    ranked.append((key,i))
                seen=[]
                for _,i in sorted(ranked,reverse=True):
                    if any(np.linalg.norm(xy[i]-xy[j])<8 for j in seen):continue
                    visible,mid_visible=fc.visible_fraction(xy[i],wall,occ)
                    if not mid_visible or visible<.88:continue
                    seen.append(i)
                    if len(seen)==3:break
                if len(seen)>=2:chosen.update(str(pics[i]["id"]) for i in seen)
    ids=sorted(chosen-cached)
    (data/'frontage-ids.json').write_text(json.dumps(ids))
    print('Additional panoramic viewpoints:',len(ids),'already downloaded candidates:',len(chosen & cached))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('area',type=Path);select(p.parse_args().area)
