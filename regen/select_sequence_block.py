"""Rank topology-defined road blocks using cached original sequences and actual SIFT overlap.
Run: UV_CACHE_DIR=/tmp/block-trial-uv uv run --with numpy --with scipy --with pyproj --with opencv-python-headless --with pillow python work/regen/select_sequence_block.py --area <area-directory>
No street/building identity enters the selection. No network or credential is used.
"""
import argparse, collections, hashlib, json, math, pathlib, re, xml.etree.ElementTree as ET
import numpy as np
import cv2
from pyproj import Transformer
from scipy.spatial.transform import Rotation

ap=argparse.ArgumentParser(); ap.add_argument('--area',type=pathlib.Path,required=True); a=ap.parse_args()
area=a.area.resolve(); out=area/'block-splat-trial'; out.mkdir(exist_ok=True)
world=json.loads((area/'world.json').read_text()); src=json.loads((area/'imagery/source-index-registered.json').read_text()); reg=json.loads((area/'imagery/registered-cameras.json').read_text()); cov=json.loads((area/'imagery/coverage.json').read_text())
config=json.loads((area/'config.json').read_text());region=config.get('region',{})
context_radius=float(world.get('contextRadiusM',region.get('contextRadiusM')));study_radius=float(world.get('studyRadiusM',region.get('studyRadiusM',context_radius)))
zone_match=re.search(r'UTM\s+zone\s+(\d+)([NS])',world.get('coordinateSystem',''),re.I)
zone=int(zone_match[1]) if zone_match else int((world['center']['lon']+180)//6)+1
northern=(zone_match[2].upper()=='N') if zone_match else world['center']['lat']>=0
epsg=world.get('projectedCrs') or (32600 if northern else 32700)+zone
proj=Transformer.from_crs(4326,epsg,always_xy=True); origin=np.array(proj.transform(world['center']['lon'],world['center']['lat']))
def xy(lon,lat): return np.array(proj.transform(lon,lat))-origin
root=ET.parse(area/'data/osm.xml').getroot(); nodes={n.get('id'):xy(float(n.get('lon')),float(n.get('lat'))) for n in root.findall('node')}
ways=[]; neighbors=collections.defaultdict(set);edge_meta={}
for way in root.findall('way'):
    tags={t.get('k'):t.get('v') for t in way.findall('tag')}
    if tags.get('highway') not in ('residential','tertiary','secondary','primary','unclassified'):continue
    ns=[n.get('ref') for n in way.findall('nd') if n.get('ref') in nodes]
    if not any(np.linalg.norm(nodes[n])<context_radius for n in ns):continue
    ways.append((way.get('id'),ns,tags))
    for l,r in zip(ns,ns[1:]): neighbors[l].add(r);neighbors[r].add(l);edge_meta[frozenset((l,r))]=(way.get('id'),tags)
cuts={n for n,v in neighbors.items() if len(v)!=2}; blocks=[];seen=set()
for wid,ns,tags in ways:
    start=0
    for i in range(1,len(ns)):
        if ns[i] not in cuts and i<len(ns)-1:continue
        ids=ns[start:i+1];start=i
        if ids[0] not in cuts or ids[-1] not in cuts:continue
        key=tuple(sorted((ids[0],ids[-1])))
        if key in seen:continue
        seen.add(key);p=np.array([nodes[n] for n in ids]);length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        if 65<=length<=210 and np.max(np.linalg.norm(p,axis=1))<study_radius:blocks.append(dict(osmWay=wid,name=tags.get('name'),nodes=ids,polyline=p.tolist(),lengthM=float(length)))
# Continue degree-two chains across OSM way boundaries: way splitting is metadata,
# not a physical intersection. Existing single-way traversal order stays stable.
joined=0
for start in sorted(cuts):
    for nxt in sorted(neighbors[start]):
        ids=[start,nxt];prev=start;cur=nxt
        while cur not in cuts:
            following=next(n for n in neighbors[cur] if n!=prev);prev,cur=cur,following;ids.append(cur)
            if len(ids)>len(neighbors):raise ValueError('invalid road chain')
        key=tuple(sorted((start,cur)))
        if key in seen:continue
        seen.add(key);p=np.array([nodes[n] for n in ids]);length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
        if not (65<=length<=210 and np.max(np.linalg.norm(p,axis=1))<study_radius):continue
        metas=[edge_meta[frozenset((l,r))] for l,r in zip(ids,ids[1:])];wid,tags=metas[0];wayids=list(dict.fromkeys(m[0] for m in metas));joined+=len(wayids)>1
        blocks.append(dict(osmWay=wid,osmWays=wayids,name=tags.get('name'),nodes=ids,polyline=p.tolist(),lengthM=float(length)))
cams=[]
for key,r in src.items():
    iid=key.split(':')[-1]; rr=reg.get(iid,{}); path=pathlib.Path(r['imagePath'])
    if not path.exists():path=area/'imagery/images'/f'{iid}.jpg'
    if not path.exists() or not rr.get('accepted'):continue
    cams.append(dict(id=iid,driveId=r.get('viewGroup'),date=r.get('captureDate'),xy=rr['xy'],path=str(path),rec=r,reg=rr))
P=np.array([c['xy'] for c in cams])
def project_route(points,line):
    best=np.full(len(points),1e9);along=np.zeros(len(points));offset=0.
    for v,w in zip(line,line[1:]):
        delta=w-v;L=np.linalg.norm(delta); t=np.clip((points-v)@delta/L**2,0,1);d=np.linalg.norm(points-v-t[:,None]*delta,axis=1);m=d<best;best[m]=d[m];along[m]=offset+t[m]*L;offset+=L
    return best,along
rank=[]
for block in blocks:
    d,t=project_route(P,np.array(block['polyline']));groups=collections.defaultdict(list)
    for k in np.where((d<12)&(t>3)&(t<block['lengthM']-3))[0]:groups[cams[k]['driveId']].append((float(t[k]),int(k)))
    good=[]
    for drive,items in groups.items():
        items.sort(); span=items[-1][0]-items[0][0];gap=max(np.diff([x[0] for x in items]),default=999)
        if len(items)>=7 and span/block['lengthM']>.65 and gap<20: good.append(dict(drive=drive,items=items,span=span,gap=float(gap)))
    if len(good)<2:continue
    good.sort(key=lambda g:(-g['span'],-len(g['items']),g['drive']))
    block.update(groups=good,score=round(sum(g['span'] for g in good[:2])/block['lengthM']*100+min(len(good[0]['items']),30),4));rank.append(block)
rank.sort(key=lambda b:(-b['score'],b['nodes']))
(out/'selection-geometric-ranking.json').write_text(json.dumps(rank,indent=2))
print(json.dumps({'candidateBlocks':len(blocks),'eligible':len(rank),'top':[{k:b[k] for k in ('name','lengthM','polyline','score')} for b in rank[:5]]}),flush=True)
features={}; sift=cv2.SIFT_create(nfeatures=3000)
def feat(i,sector):
    key=(i,sector)
    if key not in features:
        im=cv2.imread(cams[i]['path'],0); h,w=im.shape
        xx,yy=np.meshgrid((np.arange(800)+.5-400)/400,(np.arange(600)+.5-300)/400)
        angle=sector*np.pi/2;f=np.array([np.cos(angle),np.sin(angle),0]);right=np.cross(f,[0,0,1]);down=np.array([0,0,-1.])
        rays=f+xx[:,:,None]*right+yy[:,:,None]*down
        rot=Rotation.from_rotvec(cams[i]['rec']['camera']['rotationAngleAxis']).as_matrix();dc=rays@rot.T
        u=(.5+np.arctan2(dc[:,:,0],dc[:,:,2])/(2*np.pi))%1;v=.5-np.arctan2(-dc[:,:,1],np.hypot(dc[:,:,0],dc[:,:,2]))/np.pi
        crop=cv2.remap(im,(u*w).astype(np.float32),(v*h).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        mask=np.zeros(crop.shape,np.uint8);mask[100:520]=255
        kp,desc=sift.detectAndCompute(crop,mask);features[key]=(np.float32([k.pt for k in kp]),desc,crop)
    return features[key]
def sector_match(i,j,sector):
    p,d,_=feat(i,sector);q,e,_=feat(j,sector)
    if d is None or e is None:return dict(inliers=0,matches=0)
    pairs=cv2.BFMatcher().knnMatch(d,e,k=2);good=[m for m,n in pairs if m.distance<.82*n.distance]
    if len(good)<8:return dict(inliers=0,matches=len(good))
    pp=np.float32([p[m.queryIdx] for m in good]);qq=np.float32([q[m.trainIdx] for m in good]);F,mask=cv2.findFundamentalMat(pp,qq,cv2.USAC_MAGSAC,2,0.999)
    inl=mask.ravel().astype(bool) if mask is not None else np.zeros(len(good),bool)
    return dict(inliers=int(inl.sum()),matches=len(good),imageA=cams[i]['id'],imageB=cams[j]['id'],sector=sector,method='World-aligned pinhole views 800x600 HFOV90; SIFT ratio .82; MAGSAC F 2px; sky/vehicle mask',inlierSpanPixels=np.ptp(pp[inl],axis=0).tolist() if inl.any() else [0,0],pointsA=pp[inl].tolist(),pointsB=qq[inl].tolist())
def match(i,j):return max((sector_match(i,j,s) for s in range(4)),key=lambda m:m['inliers'])
winner=None
for block in rank[:12]:
    train,held=block['groups'][:2];items=train['items'];step=max(1,len(items)//12);trainitems=items[::step]
    adj=[match(l[1],r[1]) for l,r in zip(trainitems,trainitems[1:])]
    cross=[]
    for t,i in held['items'][::max(1,len(held['items'])//5)]:
        j=min(items,key=lambda x:abs(x[0]-t))[1];cross.append(match(j,i))
    block['actualOverlap']={'adjacentTrain':adj,'independentDrive':cross}
    ok=sum(m['inliers']>=30 for m in adj)>=max(3,len(adj)*.5) and sum(m['inliers']>=20 for m in cross)>=2
    print(json.dumps({'checked':block['name'],'trainInliers':[m['inliers'] for m in adj],'heldoutInliers':[m['inliers'] for m in cross],'accepted':ok}),flush=True)
    if ok:winner=block;break
if winner is None:raise SystemExit('No block passed actual image overlap thresholds')
lookup={c['id']:i for i,c in enumerate(cams)}
for n,m in enumerate(sorted(winner['actualOverlap']['independentDrive'],key=lambda x:-x['inliers'])[:2]):
    _,_,im1=feat(lookup[m['imageA']],m['sector']);_,_,im2=feat(lookup[m['imageB']],m['sector']);proof=cv2.cvtColor(np.hstack([im1,im2]),cv2.COLOR_GRAY2BGR)
    for p,q in zip(m['pointsA'],m['pointsB']):
        p=tuple(np.int32(p));q=tuple(np.int32(q)+[800,0]);cv2.line(proof,p,q,(40,220,40),1);cv2.circle(proof,p,3,(0,0,255),1);cv2.circle(proof,q,3,(0,0,255),1)
    cv2.imwrite(str(out/f'independent-drive-overlap-{n+1}.jpg'),proof)
def record(item):
    t,i=item;c=cams[i];r=c['rec'];rr=c['reg'];rot=Rotation.from_rotvec(r['camera']['rotationAngleAxis']).as_matrix()
    ta,tb=np.radians(rr.get('tiltDeg',[0,0])); tilt=Rotation.from_rotvec([tb,-ta,0.]).as_matrix();yaw=Rotation.from_euler('z',-rr.get('dyawDeg',0),degrees=True).as_matrix()
    # Exact register_cameras.apply_registration convention; preserve the fit inputs too.
    rot=rot@yaw@tilt
    im=cv2.imread(c['path']);h,w=im.shape[:2]
    return dict(id=c['id'],driveId=c['driveId'],captureDate=c['date'],imagePath=c['path'],imageSha256=r.get('imageSha256'),width=w,height=h,alongM=t,eyeEnu=[*rr['xy'],rr['z']],rotationWorldToCamera=rot.tolist(),originalProviderRotationAngleAxis=r['camera']['rotationAngleAxis'],providerLocationWgs84=r['camera']['locationWgs84'],providerAltitudeM=r['camera'].get('altitudeM'),providerMetadataPath=str(area/'imagery/entities'/f"{c['id']}.json"),registration=rr,sourceUrl=r['sourceUrl'])
train,held=winner['groups'][:2]
contract=dict(schema='block-splat-cameras-v1',coordinateSystem={'world':'ENU meters relative to world.center','projectedCrs':str(epsg),'viewer':'[east,up,-north]','verticalOriginM':world['verticalOrigin'],'center':world['center'],'projection':'camera = R @ (point_enu-eyeEnu); u=.5+atan2(camera.x,camera.z)/(2*pi); v=.5-atan2(-camera.y,hypot(camera.x,camera.z))/pi'},route={'enu':[[*p,0.] for p in winner['polyline']],'lengthM':winner['lengthM'],'osmWay':winner['osmWay'],'label':winner['name']},train=[record(it) for it in train['items']],heldout=[record(it) for it in held['items']])
(out/'cameras.json').write_text(json.dumps(contract,indent=2))
evidence=dict(schema='sequence-block-selection-v1',rules={'topology':'motor-road intersection nodes have graph degree != 2; degree-two chains cross OSM way boundaries','contextRadiusM':context_radius,'studyRadiusM':study_radius,'projectedCrs':str(epsg),'joinedMultiWayBlocks':joined,'candidateBlocks':len(blocks),'lengthM':[65,210],'cameraDistanceM':12,'minFramesPerDrive':7,'minSpanFraction':.65,'maxGapM':20,'rank':'sum of best two drive spans / block length *100 + min(trainFrameCount,30)','actualImageGate':'at least half adjacent training pairs and at least 3 pairs have >=30 SIFT/MAGSAC inliers; >=2 cross-drive pairs have >=20 inliers','identityRules':'No road name, road ID, building name or building ID is used in rules or ranking'},sources={'fullSequenceIndex':str(area/'imagery/coverage.json'),'cachedSourceIndex':str(area/'imagery/source-index-registered.json'),'providerMetadata':str(area/'imagery/entities'),'cachedFrames':str(area/'imagery/images'),'coverageCount':len(cov['images']),'sourceCount':len(src),'eligibleBlocks':len(rank),'inputSha256':{str(p.relative_to(area)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [area/'world.json',area/'config.json',area/'data/osm.xml',area/'imagery/source-index-registered.json',area/'imagery/coverage.json',area/'imagery/registered-cameras.json']}},winner=winner,trainDrive=train['drive'],heldoutDrive=held['drive'],trainCount=len(contract['train']),heldoutCount=len(contract['heldout']),limitation='Image matches measure common visible surfaces, not surveyed pose accuracy. Provider sequence IDs and distinct capture dates establish separate drives; no claimed independent survey. Heldout images enter selection overlap checks but must never enter representation training.')
(out/'selection-evidence.json').write_text(json.dumps(evidence,indent=2));print(json.dumps({'winner':contract['route'],'train':len(contract['train']),'heldout':len(contract['heldout'])}),flush=True)

# Explicit source links, free of signed CDN query strings or credentials.
creator_by_id={str(v['id']):v.get('creator_id') for v in cov['images']}
attributions={'imageryProvider':'Mapillary','imageSources':[dict(imageId=c['id'],provider='Mapillary',contributorId=creator_by_id.get(c['id']),captureDate=c['captureDate'],publicSourceUrl=c['sourceUrl']) for split in ('train','heldout') for c in contract[split]],'mapGeometry':{'attribution':'© OpenStreetMap contributors','url':'https://www.openstreetmap.org/copyright'},'note':'Contributor IDs are from cached public coverage metadata; contributor display names and provider license terms were not cached.'}
(out/'source-attributions.json').write_text(json.dumps(attributions,indent=2))
for split in ('train','heldout'):
    for c in contract[split]:
        metadata=json.loads(pathlib.Path(c['providerMetadataPath']).read_text())
        assert metadata['sequence']==c['driveId']
        assert hashlib.sha256(pathlib.Path(c['imagePath']).read_bytes()).hexdigest()==c['imageSha256']
assert train['drive']!=held['drive']
(out/'verification.json').write_text(json.dumps({'verifiedOriginalImageHashes':sum(len(contract[k]) for k in ('train','heldout')),'verifiedProviderSequenceMetadata':sum(len(contract[k]) for k in ('train','heldout')),'distinctTrainHeldoutDriveIds':True,'routeEnuZ':'zero placeholder: derive from world terrain, camera eyeEnu z is real registered height','credentialMaterialWritten':False},indent=2))

import importlib.metadata, sys
packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pyproj','opencv-python-headless','pillow')}
(out/'reproduction.json').write_text(json.dumps({'pythonVersion':sys.version,'packages':packages,'command':'uv run '+ ' '.join('--with '+p+'=='+v for p,v in packages.items())+' python work/regen/select_sequence_block.py --area '+str(area),'offlineCapable':True,'framesDownloaded':0,'sourceAttributions':'source-attributions.json'},indent=2))
