"""Recover candidate facade depths from matched perspective pixels, entirely locally.

Observations are in the source frame, not the render. A semantic image ROI selects
matches, never defines a metric endpoint. Two images are not a third-view check.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from general_world.stereo_rays import closest_positive_rays


def camera(job):
    origin=np.asarray(job['from'],float); forward=np.asarray(job['at'],float)-origin
    forward/=np.linalg.norm(forward); right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right); f=.5*job['height']/math.tan(math.radians(job['vfovDeg'])*.5)
    return origin,np.asarray([right,down,forward]),f,np.asarray([job['width'],job['height']],float)*.5-.5


def ray(cam,px):
    c,axes,f,center=cam;return np.r_[(np.asarray(px)-center)/f,1.]@axes


def pixel(cam,p):
    c,axes,f,center=cam;v=axes@(np.asarray(p)-c)
    return center+f*v[:2]/v[2] if v[2]>0 else np.full(2,np.inf)


def triangulate(cams,xy):
    r=closest_positive_rays(cams[0][0],ray(cams[0],xy[0]),cams[1][0],ray(cams[1],xy[1]),min_angle_deg=5.,max_residual_m=.18)
    if not r['valid']:return None
    center=r['center'];errors=[np.linalg.norm(pixel(c,center)-p) for c,p in zip(cams,xy)]
    if max(errors)>2.5:return None
    return {'xyz':center.tolist(),'reprojectionPx':round(float(max(errors)),3),'rayGapM':round(r['rayResidualM'],3),'angleDeg':round(r['angleDeg'],2)}


def mutual_matches(a,b):
    sift=cv2.SIFT_create(nfeatures=2200); feats=[sift.detectAndCompute(im,None) for im in (a,b)]
    if any(d is None or len(d)<2 for k,d in feats):return []
    m=cv2.BFMatcher(cv2.NORM_L2); pairs=[]
    for i in (0,1):
        pairs.append({x.queryIdx:x.trainIdx for candidates in m.knnMatch(feats[i][1],feats[1-i][1],k=2) if len(candidates)==2 for x,y in [candidates] if x.distance<.73*y.distance})
    # A keypoint may have several SIFT orientations; do not count the same location twice.
    seen=set();ret=[]
    for i,j in pairs[0].items():
        if pairs[1].get(j)!=i:continue
        a,b=feats[0][0][i].pt,feats[1][0][j].pt;key=tuple(round(z) for z in (*a,*b))
        if key not in seen:ret.append((a,b));seen.add(key)
    return ret


def dominant_plane(samples,region):
    """Require coverage across the ROI, coherent offsets and independent locations."""
    x0,y0,x1,y1=region
    rows=[s for s in samples if x0<=s['pixels'][0][0]<=x1 and y0<=s['pixels'][0][1]<=y1]
    out={'featureTracks':len(rows),'status':'unresolved'}
    if len(rows)<8:return {**out,'reason':'insufficient-local-tracks'}
    depths=np.asarray([s['depthM'] for s in rows]); support=np.abs(depths[:,None]-depths[None,:])<=.22
    mask=support[np.argmax(support.sum(axis=1))];pos=np.asarray([s['pixels'][0] for s in rows])[mask]
    span=np.ptp(pos,axis=0)/np.maximum(1.,[x1-x0,y1-y0]); med=np.median(depths[mask]);mad=np.median(abs(depths[mask]-med));out.update(support=int(mask.sum()),supportFraction=round(float(mask.mean()),3),roiSpan=span.round(3).tolist())
    if mask.sum()<8 or mask.mean()<.65 or min(span)<.38 or mad>.12:return {**out,'reason':'weak-or-localized-plane-consensus'}
    return {**out,'status':'candidate-depth-not-verified','offsetFromFrameM':round(float(med),3),'medianAbsDeviationM':round(float(mad),3)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True,type=Path);p.add_argument('--inventory',required=True,type=Path);p.add_argument('--profiles',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--refine-orientation',action='store_true');a=p.parse_args()
    jobs=json.loads(a.jobs.read_text());jobs=jobs['poses'] if isinstance(jobs,dict) else jobs;by_side=defaultdict(list)
    for j in jobs:by_side[j['side']].append(j)
    inventory=json.loads(a.inventory.read_text());prof={x['id']:x for x in json.loads(a.profiles.read_text())['profiles']};records=[]
    for ob in inventory['records']:
        sid=ob['side']; profile=prof[ob['profileId']];src=ob['source'];ref=[j for j in by_side[sid] if j['image']==src['imageId']];alt=[j for j in by_side[sid] if j['image']!=src['imageId']];out={'side':sid,'status':'unresolved'};records.append(out)
        if len(ref)!=1 or not alt:out['reason']='ambiguous-or-missing-source-pair';continue
        js=ref+alt;paths=[Path(j['persp']) for j in js];hashes=[hashlib.sha256(f.read_bytes()).hexdigest() for f in paths];out['sources']=[{'image':j['image'],'photoSha256':h,'captureDate':j['captureDate'],'registration':j['registered']} for j,h in zip(js,hashes)]
        if hashes[0]!=src['originalPerspectiveSha256']:out['reason']='source-photo-changed';continue
        ims=[cv2.imread(str(f),cv2.IMREAD_GRAYSCALE) for f in paths]
        if any(im is None or im.shape!=(j['height'],j['width']) for im,j in zip(ims,js)):out['reason']='frame-dimensions-mismatch';continue
        cams=[camera(j) for j in js]; tracks=defaultdict(list); per_view=[];n=np.array(profile['normal'],float);n/=np.linalg.norm(n)
        for v in range(1,len(ims)):
            pairs=mutual_matches(ims[0],ims[v]);count=0;witness_cam=cams[v];fit={'applied':False}
            if a.refine_orientation:
                from regen.stereo_orientation import refit
                witness_cam,fit=refit(cams[0],witness_cam,pairs)
            for xy in pairs:
                fr=triangulate([cams[0],witness_cam],xy)
                if not fr:continue
                count+=1;tracks[tuple(np.round(xy[0],2))].append({**fr,'pixels':np.round(xy,2).tolist(),'depthM':round(float(n@fr['xyz'][:2]-profile['planeV']),3),'witness':js[v]['image']})
            per_view.append({'witness':js[v]['image'],'baselineM':round(float(np.linalg.norm(cams[0][0]-cams[v][0])),3),'matched':len(pairs),'triangulated':count,'relativeOrientation':fit})
        features=[]; disagreements=0
        for rows in tracks.values():
            # Three-view support means the same primary image keypoint is seen by both external
            # cameras and each independently puts it at the same depth, not merely in the same ROI.
            values=np.array([x['xyz'] for x in rows]);dist=np.linalg.norm(values[:,None,:]-values[None,:,:],axis=2)
            groups=dist<.30;mask=groups[np.argmax(groups.sum(axis=1))];group=[r for r,use in zip(rows,mask) if use]
            if len(rows)>1 and len(group)<2:disagreements+=1
            center=np.median([r['xyz'] for r in group],axis=0);best=min(group,key=lambda r:r['reprojectionPx']);features.append({**best,'xyz':center.round(4).tolist(),'depthM':round(float(n@center[:2]-profile['planeV']),3),'agreeingWitnesses':[r['witness'] for r in group]})
        confirmed=[f for f in features if len(f['agreeingWitnesses'])>=2]
        out.update(status='evaluated',pairs=per_view,matched=sum(t['matched'] for t in per_view),triangulated=len(features),multiViewTracks=len(confirmed),conflictingTracks=disagreements,features=[{'family':f['family'],'pixelBounds':f['pixelBounds'],**dominant_plane(features,f['pixelBounds']),'independentViewCheck':dominant_plane(confirmed,f['pixelBounds'])} for f in ob['visualFeatures'] if f.get('pixelBounds')],tracks=features)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'schema':'stereo-context-feature-depth-v1','sourceInputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.jobs,a.inventory,a.profiles)},'records':records,'geometryChanged':False},indent=1)+'\n');print([(r['side'],r.get('matched'),r.get('triangulated'),sum(f['status'].startswith('candidate') for f in r.get('features',[]))) for r in records])

if __name__=='__main__':main()
