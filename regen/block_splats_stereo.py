"""Dense left/right-checked stereo with multi-pair fusion and per-surface covariance."""
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from block_splats_pose import perspective, unit, stereo_basis


def stereo_match(gray_a, gray_b, maximum, negative=False):
    sgbm = cv2.StereoSGBM_create(minDisparity=-maximum if negative else 0,numDisparities=maximum,
                blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=2,uniquenessRatio=9,
                speckleWindowSize=90,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sgbm.compute(gray_a,gray_b).astype('f')/16.


def triangulate_pair(first,second,route,along,stride):
    width,height,focal=864,586,363.
    delta = second['eye']-first['eye']
    view = stereo_basis(delta)
    a = perspective(first,view,width,height,focal)
    b = perspective(second,view,width,height,focal)
    baseline = np.linalg.norm(delta)
    maximum = int(np.clip(np.ceil(focal*baseline/6.5/16)*16,80,400))
    ga,gb = [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in (a,b)]
    # Restrict sky and vehicle mount directions by geometry/corridor bounds below.
    clahe = cv2.createCLAHE(clipLimit=2.,tileGridSize=(12,8))
    l,r = [clahe.apply(im) for im in (ga,gb)]
    left = stereo_match(l,r,maximum)
    right = stereo_match(r,l,maximum,negative=True)
    y,x = np.indices(left.shape)
    rx = np.rint(x-left).astype(int).clip(0,width-1)
    agreement = abs(left+right[y,rx])
    good = (left>focal*baseline/58.) & (left<maximum-.5) & (agreement<=1.25) & (x-left>=.5)
    # Homogeneous patches and gross exposure or moving-object disagreements make poor surfels.
    error = np.mean(abs(a.astype('f')-b[y,rx].astype('f')),axis=2)
    error=cv2.blur(error,(3,3))
    localmean=cv2.blur(ga.astype('f'),(9,9))
    contrast=np.sqrt(np.maximum(cv2.blur(ga.astype('f')**2,(9,9))-localmean**2,0))
    good &= (error<44) & (contrast>2.1)
    z=focal*baseline/np.maximum(left,1.)
    sx=(x-(width-1)/2)/focal
    sy=(y-(height-1)/2)/focal
    xyz=(np.stack([sx*z,sy*z,z],axis=-1)@view + first['eye']).astype('f')
    route_normal=np.array([-along[1],along[0]])
    longitudinal=(xyz[:,:,:2]-route[0])@along
    lateral=(xyz[:,:,:2]-route[0])@route_normal
    elev=xyz[:,:,2]-first['eye'][2]
    good &= (longitudinal>0)&(longitudinal<np.linalg.norm(route[-1]-route[0])) & (abs(lateral)>5.) & (abs(lateral)<47.) & (elev>-3.7) & (elev<27.)
    # Central metric differences estimate the two principal one-sigma directions of a
    # projected pixel footprint. Depth discontinuities cannot make giant Gaussian flakes.
    offset=2
    dx=(np.roll(xyz,-offset,axis=1)-np.roll(xyz,offset,axis=1))/(2*offset)
    dy=(np.roll(xyz,-offset,axis=0)-np.roll(xyz,offset,axis=0))/(2*offset)
    scale=z/focal
    for shift,axis in [(offset,1),(-offset,1),(offset,0),(-offset,0)]:
        other=np.roll(left,shift,axis=axis)
        good &= (other>0)&(abs(other-left)<np.maximum(.045*left,2.8))
    good[:offset]=False;good[-offset:]=False;good[:,:offset]=False;good[:,-offset:]=False
    # Reject surface directions dominated by triangulation noise.
    good &= (np.linalg.norm(dx,axis=-1)<4.5*scale)&(np.linalg.norm(dy,axis=-1)<4.5*scale)
    good &= (x%stride==0)&(y%stride==0)
    xyz,dx,dy,rgb,err,lr,z=xyz[good],dx[good],dy[good],a[good][:,::-1],error[good],agreement[good],z[good]
    # SVD of the 3x2 differential yields true orthogonal tangent covariance axes.
    tangent=np.stack((dx,dy),axis=-1)
    u,s,_=np.linalg.svd(tangent,full_matrices=False)
    s=np.clip(s, (z/focal)[:,None]*.6, (z/focal)[:,None]*3.)
    major=(u[:,:,0]*s[:,0,None]*stride*.76).astype('f')
    minor=(u[:,:,1]*s[:,1,None]*stride*.76).astype('f')
    score=(1/(1+err/20+lr*.3+z/60)).astype('f')
    return xyz,major,minor,rgb,score,{'baselineM':round(float(baseline),2),'validStrideSamples':len(xyz),
                                      'leftRightValidPixels':int(((left>0)&(agreement<=1.25)).sum())}


def wall_distance_filter(xyz, route, along, out):
    # Use released static facade support only as a distance gate; stereo sets all centers.
    meshes = [p for p in out.glob('surface-side-*.npz')]
    points = []
    for p in meshes:
        with np.load(p) as d:
            grid=d['xyz']
            mask=d['validMask'] if 'validMask' in d else np.isfinite(grid).all(axis=2)
            valid=grid[mask.astype(bool)]
            points.append(valid[::4])
    if not points:
        return np.ones(len(xyz),bool),None
    tree=cKDTree(np.vstack(points))
    distances,_=tree.query(xyz,k=1,workers=4,distance_upper_bound=3.1)
    return np.isfinite(distances),3.1


def fuse(segments, route, along, out):
    xyz,u,v,rgb,score=[np.concatenate([s[k] for s in segments]) for k in range(5)]
    pair_id=np.concatenate([np.full(len(s[0]),i,dtype='int16') for i,s in enumerate(segments)])
    wall,threshold=wall_distance_filter(xyz,route,along,out)
    xyz,u,v,rgb,score,pair_id=[a[wall] for a in (xyz,u,v,rgb,score,pair_id)]
    if not len(xyz):
        raise ValueError('no stereo samples passed the geometric support filter')
    # Confirm each retained Gaussian against independently triangulated neighboring pairs.
    # Each neighbor votes only when location and appearance are compatible.
    tree=cKDTree(xyz)
    accepted=np.zeros(len(xyz),bool)
    validated=np.zeros(len(xyz),np.uint8)
    for s in range(0,len(xyz),24000):
        rows=np.arange(s,min(len(xyz),s+24000))
        _,idx=tree.query(xyz[rows],k=min(22,len(xyz)),distance_upper_bound=.36,workers=4)
        present=idx<len(xyz)
        idx=np.minimum(idx,len(xyz)-1)
        compatible=(np.mean(abs(rgb[rows,None].astype('f')-rgb[idx].astype('f')),axis=-1)<38.) & present
        other=(pair_id[rows,None]!=pair_id[idx]) & compatible
        accepted[rows]=np.any(other,axis=1)
        validated[rows]=np.count_nonzero(other,axis=1)
    validated_count=int(accepted.sum())
    # Store one highest-quality sample per metric voxel, preserving stereo-measured outlines.
    order=np.argsort(-(score+np.minimum(validated,4)*.025),kind='stable')
    order=order[accepted[order]]
    _,first=np.unique(np.floor(xyz[order]/.066).astype('i4'),axis=0,return_index=True)
    chosen=order[np.sort(first)]
    opacity=np.clip(.76+score[chosen]*.16+np.minimum(validated[chosen],4)*.018,.76,.98)
    return (xyz[chosen],u[chosen],v[chosen],rgb[chosen].astype('f')/255,opacity.astype('f')),{
        'triangulatedSamples':sum(len(s[0]) for s in segments),'withinStaticSupport':len(xyz),
        'staticSupportToleranceM':threshold,'twoPairConfirmedSamples':validated_count,
        'fusionVoxelM':.066,'sameLocationToleranceM':.36,'primitives':len(chosen)}
