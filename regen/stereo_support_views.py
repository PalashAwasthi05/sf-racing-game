"""Select complementary local panoramas for depth instead of taking two near-identical front views.

Keeps the review camera fixed. Two witnesses should see the same facade from
spatially different bearings; azimuth diversity is balanced with visibility,
source date and the quality of their camera registration. No image uploads.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image


def bearing_offset(center,primary,candidate):
    a,b=np.asarray(primary)-center,np.asarray(candidate)-center
    return float(np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)))


def select(rows, center, primary, capture, count=3):
    options=[]; center=np.asarray(center,float)
    for row in rows:
        ang=bearing_offset(center,primary,row['cameraLocal'])
        if abs(ang)<6 or abs(ang)>52 or row['visibleFraction']<.55:continue
        temporal=1. if row['captureDate']==capture else .78
        score=row['score']*temporal*(1. if row['registered']=='skyline' else .82)/(1+.13*row['treesNearSightLine'])*min(abs(ang)/16,1.)
        options.append((score,ang,row))
    chosen=[]
    while options and len(chosen)<count:
        def spread(item):return min([1.]+[abs(item[1]-q['parallaxDeg'])/14. for q in chosen])
        best=max(options,key=lambda q:q[0]*spread(q)); options.remove(best)
        if spread(best)<.4:continue
        chosen.append({**best[2],'parallaxDeg':round(best[1],2)})
    return chosen


def main():
    ap=argparse.ArgumentParser()
    for k in ('area','sides','jobs','calibration','out'):ap.add_argument('--'+k,required=True,type=Path)
    a=ap.parse_args();os.environ['SFW_AREA']=str(a.area.resolve())
    import frontage_crops_v15 as fc
    from areas.ring.poses import side_camera_geometry,vfov_for_image
    from regen.register_cameras import apply_registration
    _,frame=fc.load_world();cams=apply_registration(fc.load_cameras(frame),a.area/'imagery/registered-cameras.json');occ=fc.Occluders(fc.load_footprints(frame));trees=fc.load_trees(frame)
    primary=json.loads(a.jobs.read_text());sides={p['id']:p for p in json.loads(a.sides.read_text())['pieces']};cal=json.loads(a.calibration.read_text())['images'];out=list(primary); report=[]
    a.out.mkdir(parents=True,exist_ok=False);imgs=a.out/'img';imgs.mkdir();hashes={}
    for j in primary:
        side=sides[j['side']];wall=side_camera_geometry(side);cands=fc.wall_candidates(wall,cams,occ,trees,fc.MAX_DISTANCE_M*1.25)
        for c in cands:c.update(captureDate=cams[c['image']]['captureDate'],registered=cams[c['image']]['registered'])
        choice=select([c for c in cands if c['image']!=j['image']], wall['midpoint'],j['from'][:2],j['captureDate'])
        rr={'side':j['side'],'eligibleViews':len(cands),'selected':[]};report.append(rr)
        for c in choice:
            key=c['image'];cam=cams[key];source=cam['path'];row=cal.get(key);digest=hashes.get(key)
            if digest is None:digest=hashes[key]=hashlib.sha256(source.read_bytes()).hexdigest()
            calibrated=bool(row and digest==row['sourceSha256'] and np.allclose(cam['R'],row['referenceRotation']));rot=np.array(row['rotation']) if calibrated else cam['R']
            eye=np.r_[cam['xy'],fc.ground_at(cam['xy'])+fc.EYE_HEIGHT_M+cam.get('heightOffsetM',0.)]; target=np.r_[wall['midpoint'],side['groundZ']+.42*side['heightM']];forward=target-eye;forward/=np.linalg.norm(forward)
            vp=fc.view_parameters(cam['xy'],wall,side['groundZ']);hf=min(105.,max(45.,vp['hfovDeg']*1.4));vf=min(105.,max(55.,vp['vfovDeg']*1.6));img,_=fc.render_perspective(fc.load_pano(source),rot,forward,hf,vf,width=1000,max_height=1100)
            path=imgs/(j['side']+'__'+key+'.jpg');Image.fromarray(img).save(path,quality=90);h,w=img.shape[:2];out.append({'side':j['side'],'building':j['building'],'image':key,'captureDate':cam['captureDate'],'registered':cam['registered'],'persp':str(path),'width':w,'height':h,'from':eye.tolist(),'at':target.tolist(),'vfovDeg':vfov_for_image(hf,w,h),'poseBasis':'registered, gravity-corrected when available','priorGravityCalibrationAvailable':calibrated,'sourcePanoSha256':digest});rr['selected'].append({'image':key,'parallaxDeg':c['parallaxDeg'],'registered':cam['registered'],'sourcePanoSha256':digest})
    (a.out/'poses.json').write_text(json.dumps({'poses':out,'selection':report},indent=1)+'\n');print(len(report),'frontages,',len(out)-len(primary),'additional local source views')
if __name__=='__main__':main()
