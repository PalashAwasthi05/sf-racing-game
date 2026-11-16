"""Register two photographs of one approximately planar architectural surface.

This is an image-space measurement. It cannot validate absolute camera positions,
recover unseen texture, or prove that matches lie on one physical wall. Matches
are mutual and ratio-tested, the homography is fitted on a training subset, and
independent matches must validate it before any pixel transfer is permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def project(points, H):
    p = np.c_[np.asarray(points, float), np.ones(len(points))] @ H.T
    return p[:, :2] / p[:, 2, None]


def align(a, b, mask_a=None, mask_b=None):
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.025)
    def mask(x):
        return None if x is None else (np.asarray(x, bool).astype('uint8')*255)
    ka, da = sift.detectAndCompute(gray_a, mask(mask_a))
    kb, db = sift.detectAndCompute(gray_b, mask(mask_b))
    audit = {'accepted': False, 'sourceKeypoints': len(ka), 'targetKeypoints': len(kb),
             'absolutePositionValidated': False, 'physicalPlaneValidated': False}
    if da is None or db is None or min(len(da), len(db)) < 2:
        return {**audit, 'reason': 'insufficient-descriptors'}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def distinct(x, y):
        return {m[0].queryIdx: m[0].trainIdx for m in matcher.knnMatch(x, y, k=2)
                if len(m) == 2 and m[0].distance < .72*m[1].distance}
    ab, ba = distinct(da, db), distinct(db, da)
    pairs = sorted((i,j) for i,j in ab.items() if ba.get(j) == i)
    # SIFT emits multiple orientations at one location. They are not independent
    # correspondences and must not inflate the holdout or spatial-support count.
    unique = {}
    for i,j in pairs:
        key = tuple(round(float(v), 1) for v in (*ka[i].pt, *kb[j].pt))
        unique.setdefault(key, (i,j))
    pairs = list(unique.values())
    audit['mutualDistinctMatches'] = len(pairs)
    if len(pairs) < 24:
        return {**audit, 'reason': 'insufficient-independent-matches'}
    pa = np.array([ka[i].pt for i,j in pairs], float)
    pb = np.array([kb[j].pt for i,j in pairs], float)
    # Reserve a reproducible subset BEFORE fitting. These points do not influence
    # RANSAC selection, refinement, or the acceptance thresholds.
    order = np.random.default_rng(0).permutation(len(pairs))
    holdout = order[:max(8, len(order)//4)]
    train = order[len(holdout):]
    cv2.setRNGSeed(0)
    H, inliers = cv2.findHomography(pa[train], pb[train], cv2.RANSAC, 2.5,
                                  maxIters=4000, confidence=.999)
    if H is None or not np.isfinite(H).all() or abs(H[2,2]) < 1e-10:
        return {**audit, 'reason': 'no-finite-homography'}
    H /= H[2,2]
    training_inliers = train[inliers.ravel().astype(bool)]
    err = np.linalg.norm(project(pa, H)-pb, axis=1)
    held = err[holdout]
    good = training_inliers
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    corners = np.array([[0,0],[wa-1,0],[wa-1,ha-1],[0,ha-1]], float)
    warped = project(corners, H)
    denominators = np.c_[corners, np.ones(4)] @ H[2]
    area = cv2.contourArea(warped.astype('float32'), oriented=True)
    hull_fraction = float(cv2.contourArea(cv2.convexHull(pa[good].astype('float32')))/(wa*ha)) if len(good)>2 else 0.
    quadrants = len(set((int(x>=wa/2), int(y>=ha/2)) for x,y in pa[good]))
    reasons = []
    if len(good)<16: reasons.append('insufficient-training-inliers')
    if len(good)/len(train)<.6: reasons.append('ambiguous-or-nonplanar-matches')
    if np.median(held)>2.5 or np.mean(held<=3.)<.7: reasons.append('heldout-registration-failed')
    if hull_fraction<.15 or quadrants<3: reasons.append('insufficient-spatial-support')
    if not (np.all(denominators>.05) and cv2.isContourConvex(warped.astype('float32')) and .35*wa*ha<area<2.8*wa*ha):
        reasons.append('folded-or-excessive-warp')
    audit.update(accepted=not reasons, reason='validated-image-correspondence' if not reasons else ';'.join(reasons),
                 homographySourceToTarget=H.tolist(), trainingInliers=len(good), trainingMatches=len(train),
                 holdoutMatches=len(holdout), holdoutMedianErrorPx=float(np.median(held)),
                 holdoutP90ErrorPx=float(np.percentile(held,90)), holdoutWithin3Px=float(np.mean(held<=3.)),
                 supportHullFraction=hull_fraction, supportedQuadrants=quadrants,
                 matches=[{'source':x.tolist(),'target':y.tolist(),'errorPx':float(e),
                           'holdout': bool(i in set(holdout.tolist()))} for i,(x,y,e) in enumerate(zip(pa,pb,err))])
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('heldout',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from PIL import Image, ImageDraw
    a=np.asarray(Image.open(args.source).convert('RGB'));b=np.asarray(Image.open(args.heldout).convert('RGB'))
    report=align(a,b);args.output.mkdir(parents=True,exist_ok=True)
    report['sources']=[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in (args.source,args.heldout)]
    (args.output/'alignment.json').write_text(json.dumps(report,indent=2)+'\n')
    panel=Image.new('RGB',(a.shape[1]+b.shape[1],max(a.shape[0],b.shape[0])),'#777')
    panel.paste(Image.fromarray(a));panel.paste(Image.fromarray(b),(a.shape[1],0));draw=ImageDraw.Draw(panel)
    for m in report.get('matches',[]):
        x,y=m['source'];u,v=m['target'];color='lime' if m['errorPx']<=3 else 'red'
        if m['holdout']:
            draw.line([x,y,u+a.shape[1],v],fill=color,width=1)
    panel.save(args.output/'heldout-matches.jpg')
    if report['accepted']:
        H=np.asarray(report['homographySourceToTarget']);warped=cv2.warpPerspective(a,H,(b.shape[1],b.shape[0]))
        Image.fromarray(warped).save(args.output/'source-aligned.png')
        Image.fromarray(np.uint8(.5*warped+.5*b)).save(args.output/'aligned-overlay.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ('matches','sources','homographySourceToTarget')}))


if __name__=='__main__':main()
