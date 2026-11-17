"""Read repeatable colors and, where both views resolve them, opening rhythms on mapped walls."""
import argparse
import json
import math
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

def generate(area):
    os.environ["SFW_AREA"] = str(area.resolve())
    import frontage_crops_v15 as fc
    import auto_read_frontages_v24 as reader

    center = json.loads((area/"config.json").read_text())["region"]["center"]
    frame = fc.LocalFrame(center["lon"], center["lat"])
    footprints = fc.load_footprints(frame)
    occluders = fc.Occluders(footprints)
    cameras = fc.load_cameras(frame)
    trees = fc.load_trees(frame)
    inventory = json.loads((area/"data/buildings.json").read_text())
    rows = {"sf_"+r["sf16_bldgid"].replace(".", "_"):r for r in inventory}
    counters = Counter()
    profiles = []

    @lru_cache(maxsize=8)
    def image(camera_id):
        with Image.open(cameras[camera_id]["path"]) as original:
            original.draft("RGB", (2200,1100))
            original.thumbnail((2400,1200))
            return np.asarray(original.convert("RGB"))

    for bid, footprint in footprints.items():
        if bid not in rows: continue
        height = float(rows[bid].get("hgt_median_m") or 0)
        if height < 5: continue
        sides = {}
        for poly in getattr(footprint, "geoms", [footprint]):
            if not isinstance(poly, Polygon): continue
            coords = np.asarray(orient(poly.simplify(.4),sign=1).exterior.coords)
            for a,b in zip(coords,coords[1:]):
                segment = b-a
                length = float(np.linalg.norm(segment))
                if length < 9: continue
                normal = np.array([segment[1],-segment[0]]) / length
                key = round(math.atan2(normal[1],normal[0])/(math.pi/18))%36
                if key not in sides or length>sides[key]["lengthM"]:
                    sides[key] = dict(midpoint=(a+b)/2, normal=normal, lengthM=length,heightM=height)
        for wall in sides.values():
            counters["wallCandidates"] += 1
            # Read a bounded, interior patch of the face. The street camera cannot
            # reliably resolve all stories or distant ends of a wide tower.
            patch = dict(wall, lengthM=min(wall["lengthM"]*.8,26),heightM=min(height*.88,23))
            views = [v for v in fc.wall_candidates(patch,cameras,occluders,trees,52)
                     if v["incidence"]>=.66 and v["visibleFraction"]>=.88 and v["treesNearSightLine"]<=1]
            readings=[]
            for v in sorted(views, key=lambda v:-v["score"])[:6]:
                cam = cameras[v["image"]]
                if any(np.linalg.norm(cam["xy"]-cameras[old[0]]["xy"]) < 7 for old in readings):continue
                rgb = image(v["image"])
                img,h,l = reader.rectify(patch,cam,0,rgb,base=1.1)
                r = reader.read(img,h,l)
                if r.get("wallShare",0)<.23 or r["skyShare"]>.16 or r["foliageShare"]>.18:continue
                readings.append((v["image"],r))
                if len(readings)>=3:break
            for idx,(iid,r) in enumerate(readings):
                other = next(((i,q) for i,q in readings[idx+1:] if r["family"][0]==q["family"][0] and
                              np.linalg.norm(np.subtract(r["wallRgb"],q["wallRgb"]))<38),None)
                if not other:continue
                oid,q=other
                cons=dict(r,wallRgb=np.rint(np.mean([r["wallRgb"],q["wallRgb"]],axis=0)).astype(int).tolist())
                same_grid=(height<=26 and wall["lengthM"]<=40 and
                           min(r.get("gridConfidence",0),q.get("gridConfidence",0))>=.36 and
                           not r.get("baysUnclear") and not q.get("baysUnclear") and
                           abs(r.get("floorPitchM",0)-q.get("floorPitchM",0))<.45 and
                           abs(r.get("bayPitchM",0)-q.get("bayPitchM",0))<.45)
                if not same_grid:cons.pop("floorPitchM",None)
                cons["shopfront"]=r["shopfront"] and q["shopfront"]
                item=reader.profile_for(bid,{},wall,cons,iid)
                item["sourceIds"].append("mapillary:"+oid)
                item["materialEvidence"]["sourceIds"]=item["sourceIds"]
                item["sourceWallMidpointEN"]=np.round(wall["midpoint"],2).tolist()
                item["sourceWallLengthM"]=round(wall["lengthM"],2)
                profiles.append(item)
                counters["pairedFacings"]+=1
                counters["pairedGrids"]+=int(same_grid)
                break
    packet={"schemaVersion":1,"method":"Mapped wall projections from distinct panorama positions with unobstructed sightlines and agreeing samples; camera poses are provider estimates.", "stats":dict(counters),"profiles":profiles}
    index=json.loads((area/"imagery/source-index-registered.json").read_text())
    packet["sources"]={key:{"provider":"Mapillary","url":index[key]["sourceUrl"],
                            "captureDate":index[key]["captureDate"],"registration":"provider pose; multi-view color agreement"}
                       for key in sorted({key for p in profiles for key in p["sourceIds"]})}
    (area/"imagery/frontage-profiles.json").write_text(json.dumps(packet,indent=1))
    print(dict(counters),len({p["match"]["featureId"] for p in profiles}),"distinct buildings",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("area",type=Path);generate(p.parse_args().area)
