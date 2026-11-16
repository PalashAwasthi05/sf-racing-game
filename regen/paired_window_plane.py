"""Estimate a local inter-photo warp for a physical storefront from surrounding
static features. Exclude all lettering being tested from registration. Reject
low texture, ambiguity across repeating frames, and inconsistent held-out ties.
"""
import cv2
import numpy as np
def cast(points,H):
    pts=np.array(points).reshape(-1,2);return pts@H[:,:2].T+H[:,2]
def _find(a,b,x,y,rad,search):
    h,w=a.shape;r=rad;s=search
    if not(r<=x<w-r and r<=y<h-r):return
    l,t=max(0,x-r-s),max(0,y-r-s);R,B=min(w,x+r+s+1),min(h,y+r+s+1)
    tile=a[y-r:y+r+1,x-r:x+r+1]
    if np.std(tile)<6:return
    q=cv2.matchTemplate(b[t:B,l:R],tile,cv2.TM_CCOEFF_NORMED)
    _,best,_,(xx,yy)=cv2.minMaxLoc(q)
    if best<.74:return
    q[max(0,yy-4):yy+5,max(0,xx-4):xx+5]=-1
    if best-q.max()<.07:return
    return np.array([l+xx+r,t+yy+r]),float(best)

def register(a,b,bounds,excluded,ppm):
    x,y,X,Y=list(map(int,bounds));mask=np.zeros(a.shape,np.uint8);mask[y:Y,x:X]=255
    for l,t,r,d in excluded:cv2.rectangle(mask,(int(l)-9,int(t)-9),(int(r)+10,int(d)+10),0,-1)
    pts=cv2.goodFeaturesToTrack(a,180,.012,max(9.,.23*ppm),mask=mask)
    stats={'accepted':False};src=[];dst=[];rad=max(6,round(.22*ppm));search=round(.99*ppm)
    for px,py in ([] if pts is None else pts[:,0,:]):
        xx,yy=round(float(px)),round(float(py))
        q=_find(a,b,xx,yy,rad,search)
        if q is None:continue
        pt=q[0];rev=_find(b,a,*pt,rad,search)
        if rev is None or np.linalg.norm(rev[0]-[xx,yy])>2:continue
        src.append([xx,yy]);dst.append(pt)
    stats['independentMutualTies']=len(src)
    if len(src)<12:return stats
    src=np.array(src,float);dst=np.array(dst,float);order=np.random.default_rng(883).permutation(len(src));hold=order[::3];tr=np.setdiff1d(order,hold);cv2.setRNGSeed(381)
    A,ins=cv2.estimateAffinePartial2D(src[tr],dst[tr],method=cv2.RANSAC,ransacReprojThreshold=max(2.,.055*ppm),maxIters=2000,confidence=.995)
    if A is None:return stats
    err=np.linalg.norm(cast(src[hold],A)-dst[hold],axis=1);used=src[tr][ins.ravel()>0];spread=np.ptp(used,axis=0)/[X-x,Y-y]
    stats.update(trainingInliers=len(used),heldoutMedianPx=round(float(np.median(err)),2),heldoutInlierShare=round(float((err<.08*ppm).mean()),3),spread=np.round(spread,3).tolist())
    if min(spread)<.40 or len(used)<7 or np.mean(err<.08*ppm)<.72 or np.median(err)>.06*ppm or not .90<np.linalg.det(A[:,:2])<1.10:return stats
    return {**stats,'accepted':True,'affine':A.tolist()}
