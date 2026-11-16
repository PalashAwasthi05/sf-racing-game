"""Expand only stereo gaps using a metric facade-prior plane sweep validated by three cameras."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from block_splats import atomic_json, calibrated_inputs, write_model
from block_splats_pose import pano, unit, load_cameras
from block_splats_raster import native, images, run_views


def frozen_model(path):
    records=np.fromfile(path,dtype=[('x','<f4',(3,)),('u','<f4',(3,)),('v','<f4',(3,)),('c','u1',(4,))])
    convert=lambda arr: np.ascontiguousarray(arr[:,[0,2,1]]*np.array([1,-1,1]),dtype='f')
    return (convert(records['x']),convert(records['u']),convert(records['v']),
            records['c'][:,:3].astype('f')/255,records['c'][:,3].astype('f')/255)


def photograph(im,cam,pts):
    ray=(pts-cam['eye'])@cam['R'].T
    h,w=im.shape[:2]
    x=(np.arctan2(ray[:,:,0],ray[:,:,2])/(2*np.pi)+.5)*w-.5
    y=(np.arctan2(ray[:,:,1],np.hypot(ray[:,:,0],ray[:,:,2]))/np.pi+.5)*h-.5
    return cv2.remap(im,x.astype('f'),y.clip(0,h-1).astype('f'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)[:,:,::-1]


def sweep(path,cameras,along,base_xyz):
    with np.load(path) as d:
        xyz=d['xyz'][1::3,1::3].copy()
        side=int(d['side']); spacing=3/float(d['ppm'])
    valid=np.isfinite(xyz).all(axis=-1)
    xyz=np.nan_to_num(xyz)
    height,width=valid.shape
    outward=np.array([along[1]*side,-along[0]*side,0.],dtype='f')
    # Restrict augmentation to a missing chart region across all plausible offsets.
    # Otherwise a second, slightly displaced version of an existing edge might be learned.
    base_side=((base_xyz[:,:2]-cameras[len(cameras)//2]['eye'][:2])@outward[:2]) < 0
    base_front=base_xyz[base_side]
    if len(base_front):
        axes=np.array([[*along,0],[0.,0.,1.]])
        tree=cKDTree(base_front@axes.T)
        dist,_=tree.query((xyz@axes.T)[valid],workers=4)
        gap=np.zeros_like(valid);gap[valid]=dist>.40
    else:
        gap=valid.copy()
    geom=[]
    for c in cameras:
        delta=xyz-c['eye']
        distance=np.linalg.norm(delta,axis=-1)
        facing=-(delta@outward)/np.maximum(distance,.1)
        # Head-on sources outrank glancing and far sources. No per-facade identities enter.
        geom.append(np.where(valid&(distance<46)&(facing>.27),facing**2/np.maximum(distance,1),0))
    geom=np.array(geom)
    order=np.argsort(-geom,axis=0,kind='stable')[:4]
    sufficient=np.take_along_axis(geom,order,axis=0)[2]>.003
    offsets=np.arange(-1.8,1.81,.3).astype('f')
    cub=np.zeros((len(offsets),4,height,width,3),np.uint8)
    for i,c in enumerate(cameras):
        im=pano(c['imagePath'])
        needed=(order==i)
        if not needed.any():
            continue
        cols=np.flatnonzero(needed.any(axis=(0,1)))
        l,r=cols.min(),cols.max()+1
        for k,offset in enumerate(offsets):
            warped=photograph(im,c,xyz[:,l:r]+outward*offset)
            for j in range(4):
                mask=needed[j,:,l:r]
                cub[k,j,:,l:r][mask]=warped[mask]
    costs=[]; groups=[]
    triples=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    pair_keys=[(i,j) for i in range(4) for j in range(i+1,4)]
    # A triple needs independent metric baselines and all three source observations.
    allowed=[]
    eye=np.stack([c['eye'] for c in cameras])
    weights=np.take_along_axis(geom,order,axis=0)
    for t in triples:
        amin=np.minimum.reduce([np.linalg.norm(eye[order[i]]-eye[order[j]],axis=-1) for i,j in pair_keys if i in t and j in t])
        allowed.append((weights[list(t)].min(0)>.003)&(amin>1.6))
    for k in range(len(offsets)):
        rgba=cub[k].astype('f')
        gray=rgba.mean(axis=3)
        # High-pass luminance retains window/balcony edges in variable exposure.
        detail=np.stack([x-cv2.blur(x,(7,7)) for x in gray])
        paircost={}
        for i,j in pair_keys:
            diff=np.abs(rgba[i]-rgba[j]).mean(axis=-1)
            high=np.abs(detail[i]-detail[j])
            paircost[i,j]=cv2.blur(np.minimum(diff,75)*.58+np.minimum(high,55)*.42,(5,5))
        tcost=[]
        for tid,t in enumerate(triples):
            distances=[paircost[i,j] for i,j in pair_keys if i in t and j in t]
            tcost.append(np.where(allowed[tid],.62*np.maximum.reduce(distances)+.38*np.mean(distances,axis=0),200.))
        costs.append(np.min(tcost,axis=0));groups.append(np.argmin(tcost,axis=0))
    costs=np.array(costs)
    k=np.argmin(costs,axis=0)
    y,x=np.indices(k.shape)
    minimum=costs[k,y,x]
    alt=np.min(np.where(abs(np.arange(len(offsets))[:,None,None]-k)>2,costs,250),axis=0)
    depth=offsets[k]
    supported=valid&gap&sufficient&(minimum<18)&(alt-minimum>1.45)&(k>0)&(k<len(offsets)-1)
    median=cv2.medianBlur(depth.astype('f'),5)
    supported &= (abs(depth-median)<.4) & (median>=-.12)
    # Subsample cost parabola at the accepted grid minimum and keep metric relief smooth.
    prev=costs[np.maximum(0,k-1),y,x];nxt=costs[np.minimum(len(offsets)-1,k+1),y,x]
    curve=(prev+nxt-2*minimum)
    depth=depth+.15*np.clip((prev-nxt)/np.maximum(curve,1.e-4),-1,1)
    depth=cv2.medianBlur(depth.astype('f'),3)
    depth=cv2.GaussianBlur(depth,(3,3),.6)
    centers=xyz+outward*depth[:,:,None]
    dx=(np.roll(centers,-1,axis=1)-np.roll(centers,1,axis=1))*.5
    dy=(np.roll(centers,-1,axis=0)-np.roll(centers,1,axis=0))*.5
    supported &= (np.linalg.norm(dx,axis=-1)<spacing*2.)&(np.linalg.norm(dy,axis=-1)<spacing*2.)
    selected_triple=np.array(triples)[np.stack(groups)[k,y,x]]
    obs=np.stack([cub[k,selected_triple[:,:,j],y,x] for j in range(3)])
    disagreement=np.max(abs(obs.astype('f')-np.median(obs,axis=0)).mean(axis=-1),axis=0)
    supported &= disagreement<29.
    tangents=np.stack((dx[supported],dy[supported]),axis=-1)
    vectors,scales,_=np.linalg.svd(tangents,full_matrices=False)
    scales=np.clip(scales,spacing*.7,spacing*2.)
    u=vectors[:,:,0]*scales[:,0,None]*.80;v=vectors[:,:,1]*scales[:,1,None]*.80
    rgb=np.median(obs,axis=0)[supported]/255
    opacity=np.clip(.95-minimum[supported]*.006,.80,.95)
    alongs=centers[supported,:2]@along
    stats={'side':side,'chartCells':int(valid.sum()),'stereoGapCells':int((valid&gap).sum()),
           'threeViewGaussians':int(supported.sum()),'medianDepthBeyondFacadeM':float(np.median(depth[supported])),
           'extentAlongStreetM':[float(alongs.min()),float(alongs.max())] if len(alongs) else [],
           'medianThreeViewPatchCost':float(np.median(minimum[supported]))}
    return tuple(np.ascontiguousarray(a,dtype='f') for a in (centers[supported],u,v,rgb,opacity)),stats


def validation(fn,base,mixed,records,out):
    pixels=total=0;sse=mae=lost=0;panels=[];rows=[]
    for identity,side,eye,view,f,target in records:
        h,w=target.shape[:2];outputs=[]
        for surf,rgb in (base,mixed):
            img=np.zeros((h,w,4),'f')
            fn(len(rgb),w,h,f,eye,view,surf,rgb,target,img,np.zeros((0,3),'f'),np.zeros(0,'f'),0)
            outputs.append(img)
        reference,pred=outputs
        new=(reference[:,:,3]<.72)&(pred[:,:,3]>=.72)
        norm=pred[:,:,:3]/np.maximum(pred[:,:,3:],1.e-5)
        err=(target[new]-norm[new]).astype(float)
        pixels+=int(new.sum());total+=h*w;mae+=float(abs(err).sum());sse+=float((err**2).sum())
        lost+=int(((reference[:,:,3]>=.72)&(pred[:,:,3]<.72)).sum())
        rows.append({'camera':identity,'side':side,'newCoverage':float(new.mean()),'newPixels':int(new.sum()),
                     'newMAE':float(abs(err).mean()*255) if len(err) else None})
        original=np.where(new[:,:,None],norm,.08)
        panels.append(np.uint8(np.clip(np.hstack([target,original]),0,1)*255))
    cv2.imwrite(str(out/'training-validation-new-pixels.jpg'),np.vstack(panels)[...,::-1])
    return {'newCoverage':pixels/total,'newPixels':pixels,'totalPixels':total,'lostBaselineCoveredPixels':lost,
            'newMAESrgb255':mae/max(pixels*3,1)*255,'newPSNR':-10*np.log10(max(sse/max(pixels*3,1),1.e-12)),'views':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cameras',type=Path);ap.add_argument('--evaluate-locked',action='store_true')
    args=ap.parse_args();start=time.monotonic()
    cv2.setNumThreads(4)
    out=args.cameras.parent/'mesh-conditioned-candidate';out.mkdir(exist_ok=True)
    data,cameras,route,along,_=calibrated_inputs(args.cameras,args.cameras.parent)
    fit=[c for i,c in enumerate(cameras) if i%5!=2];reserved=[c for i,c in enumerate(cameras) if i%5==2]
    base=frozen_model(args.cameras.parent/'block-splats.bin')
    pack=lambda m:(np.ascontiguousarray(np.column_stack([m[0],m[1],m[2],m[4]]),'f'),np.ascontiguousarray(m[3],'f'))
    fn=native()
    if args.evaluate_locked:
        report=json.loads((out/'block-splats.json').read_text())
        review_path=out/'visual-review.json'
        review=json.loads(review_path.read_text()) if review_path.exists() else {}
        if (not report['numericValidationPassed'] or not review.get('accepted')
                or review.get('lockedSha256')!=report['sha256']):
            raise ValueError('requires locked and numerically/visually accepted candidate')
        path=out/'block-splats.bin'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report['sha256']:
            raise ValueError('locked asset changed')
        rec=images(load_cameras(data['heldout'],args.cameras),along)
        metrics,_,panels=run_views(fn,*pack(frozen_model(path)),rec,preview=True)
        atomic_json(out/'independent-drive.json',metrics)
        for side in (-1,1):
            p=max((p for p in panels if p[2]==side),key=lambda p:p[0])
            cv2.imwrite(str(out/f'heldout-side-{side}.jpg'),p[-1][:,:,::-1])
        return
    parts=[];stats=[]
    for side in (-1,1):
        m,s=sweep(args.cameras.parent/f'surface-side-{side}.npz',fit,along,base[0])
        print('Conditioned component',s,flush=True);parts.append(m);stats.append(s)
    mixed=[np.ascontiguousarray(np.concatenate([base[k],parts[0][k],parts[1][k]]),'f') for k in range(5)]
    surf,rgb=pack(mixed);records=images(fit,along);history=[]
    for epoch in range(5):
        met,grad,_=run_views(fn,surf,rgb,records,fit=epoch<4)
        history.append({k:v for k,v in met.items() if k!='perView'})
        print('epoch',epoch,history[-1],flush=True)
        if epoch<4:
            rgb[len(base[0]):]=np.clip(rgb[len(base[0]):]+.86*grad[len(base[0]):],0,1)
    mixed[3]=rgb
    # Persist the candidate BEFORE scoring the reserved images. A separate command scores the final drive.
    sha=write_model(out/'block-splats.bin',mixed)
    check=validation(fn,pack(base),(surf,rgb),images(reserved,along),out)
    report={'schema':'block-surface-gaussians-plus-v1','strideBytes':40,'count':len(rgb),'bytes':len(rgb)*40,
        'sha256':sha,'frozenBaseCount':len(base[0]),'frozenBaseSha256':hashlib.sha256((args.cameras.parent/'block-splats.bin').read_bytes()).hexdigest(),
        'method':'Gap-only three-camera photometric plane sweep over thirteen metric depths, depth-discrimination and continuity gates, projected covariance and four robust alpha-composited RGB updates.',
        'conditionalCameraIds':[c['id'] for c in fit],'validationCameraIds':[c['id'] for c in reserved],
        'splitLimit':'Reserved four frame images were excluded from fitting the expansion. Prior camera calibration and the original frozen free-stereo layer used the entire training drive.',
        'components':stats,'rgbFitting':history,'trainingDriveValidation':check,'numericValidationPassed':bool(check['newCoverage']>=.015 and check['newMAESrgb255']<=38),
        'elapsedSeconds':time.monotonic()-start}
    atomic_json(out/'block-splats.json',report)
    print('LOCKED',sha,'validation',str({k:v for k,v in check.items() if k!='views'}),'passed',report['numericValidationPassed'],flush=True)


if __name__=='__main__':
    main()
