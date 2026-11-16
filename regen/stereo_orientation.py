"""Bounded relative orientation refinement using only cross-photo tracks.

The primary camera and both positions stay fixed. A withheld set checks the
new epipolar lines. A better image fit alone never verifies world-space depth.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def refit(primary,secondary,matches,*,max_degrees=2.5):
    if len(matches)<32:return secondary,{'applied':False,'reason':'insufficient-static-candidates'}
    (ca,aa,fa,pa),(cb,ab,fb,pb)=primary,secondary
    xy=np.asarray(matches,float); da=np.c_[(xy[:,0]-pa)/fa,np.ones(len(xy))]@aa
    baseline=cb-ca;baseline=baseline/np.linalg.norm(baseline); lines=np.cross(da,baseline);lines/=np.maximum(1e-9,np.linalg.norm(lines,axis=1,keepdims=True))
    b=np.c_[(xy[:,1]-pb)/fb,np.ones(len(xy))]; base=b@ab
    def residual(w,index):
        rot=Rotation.from_rotvec(w).as_matrix(); q=base[index]@rot.T; return (q*lines[index]).sum(axis=1)*fb/np.linalg.norm(q,axis=1)
    rng=np.random.default_rng(18);perm=rng.permutation(len(base));hold=perm[:max(10,len(perm)//4)];fit=perm[len(hold):]
    result=least_squares(lambda w:np.r_[residual(w,fit),np.degrees(w)*.8],np.zeros(3),bounds=(-np.radians(max_degrees),np.radians(max_degrees)),loss='soft_l1',f_scale=1.8)
    w=result.x;before=np.abs(residual(np.zeros(3),hold));after=np.abs(residual(w,hold));angle=np.degrees(np.linalg.norm(w));inliers=np.abs(residual(w,fit))<3.
    coverage=np.ptp(xy[fit][inliers,1],axis=0)/(2*np.maximum(1.,pb)) if sum(inliers)>2 else [0.,0.]
    stats={'applied':False,'rotationDeg':round(float(angle),3),'trainingInliers':int(sum(inliers)),'holdout':len(hold),'holdoutMedianBeforePx':round(float(np.median(before)),2),'holdoutMedianAfterPx':round(float(np.median(after)),2),'within3Px':round(float(np.mean(after<=3.)),3),'trainFrameSpan':np.round(coverage,3).tolist()}
    if angle>max_degrees or min(coverage)<.3 or inliers.mean()<.52 or np.mean(after<=3.)<.65 or np.median(after)>.7*np.median(before) or np.median(after)>2.5:return secondary,{**stats,'reason':'bounded-correction-or-heldout-support-failed'}
    rot=Rotation.from_rotvec(w).as_matrix();return (cb,ab@rot.T,fb,pb),{**stats,'applied':True,'rotationMatrix':rot.tolist()}
