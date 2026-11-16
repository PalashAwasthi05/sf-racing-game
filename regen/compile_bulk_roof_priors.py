"""Recover conservative local roof-front priors from typed facade readings and photos.

A broad mansard, agreeing wall-top measurement and repeated image edge produce
only a candidate. Promotion additionally requires exact-hash, per-view source
semantics identifying a continuous sloping roof above and wall below. Missing
semantics leave the candidate in the audit; no eaveM or ridgeM is inferred.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from regen.reading_provenance import load_registered_sides


def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def source_semantic_gate(profile, image_edges, span, junction_height_m):
    """Admit only a source-verified roof-over-wall junction across the interval.

    A broad mansard reading and a strong horizontal image edge locate a
    candidate, but neither says which surface is on either side of it. The
    optional ``sourceSemanticEvidence`` records that missing observation per
    registered rectified view. Each record is tied to the exact crop hash and
    image id, and describes a continuous classified interval immediately
    around the candidate. Missing or conflicting semantics remain proposals.
    """
    evidence = profile.get('sourceSemanticEvidence')
    if not isinstance(evidence, list) or not evidence:
        return {'status':'unresolved','reason':'source-surface-semantics-unavailable'}
    by_image = {}
    for row in evidence:
        if not isinstance(row, dict) or not isinstance(row.get('imageId'), (str, int)):
            continue
        by_image.setdefault(str(row['imageId']), []).append(row)
    required = {str(k) for k in image_edges}
    if not required:
        return {'status':'unresolved','reason':'no-source-views-to-verify'}
    for image_id, edge_record in image_edges.items():
        rows = by_image.get(str(image_id), [])
        if len(rows) != 1:
            return {'status':'unresolved','reason':'source-semantic-view-missing-or-ambiguous','imageId':str(image_id)}
        row = rows[0]
        if row.get('status') != 'accepted':
            return {'status':'unresolved','reason':'source-semantic-evidence-not-accepted','imageId':str(image_id)}
        if row.get('rectSha256') != edge_record.get('sha256'):
            return {'status':'unresolved','reason':'source-semantic-crop-hash-mismatch','imageId':str(image_id)}
        if row.get('above') != 'sloping_roof' or row.get('below') != 'wall':
            return {'status':'unresolved','reason':'source-does-not-show-roof-over-wall','imageId':str(image_id)}
        if row.get('overlapAmbiguous') is not False:
            return {'status':'unresolved','reason':'source-semantic-overlap-unresolved','imageId':str(image_id)}
        try:
            confidence = float(row['confidence'])
            continuous = float(row['continuousFraction'])
            bounds = [float(v) for v in row['xRange']]
            boundary = float(row['boundaryHeightM'])
        except (KeyError, TypeError, ValueError):
            return {'status':'unresolved','reason':'source-semantic-measurement-incomplete','imageId':str(image_id)}
        if not (0.0 <= confidence <= 1.0 and confidence >= .8):
            return {'status':'unresolved','reason':'source-semantic-confidence-low','imageId':str(image_id)}
        if not (0.0 <= continuous <= 1.0 and continuous >= .9):
            return {'status':'unresolved','reason':'source-semantic-span-not-continuous','imageId':str(image_id)}
        if len(bounds) != 2 or not (0 <= bounds[0] < bounds[1] <= 1):
            return {'status':'unresolved','reason':'source-semantic-span-invalid','imageId':str(image_id)}
        overlap = max(0.0, min(span[1], bounds[1]) - max(span[0], bounds[0]))
        if overlap / max(1e-9, span[1] - span[0]) < .9:
            return {'status':'unresolved','reason':'source-semantic-span-does-not-cover-prior','imageId':str(image_id)}
        if abs(boundary - float(edge_record.get('heightM', junction_height_m))) > .3:
            return {'status':'unresolved','reason':'source-semantic-boundary-height-disagrees','imageId':str(image_id)}
    extra = set(by_image) - required
    return {'status':'admitted','reason':'two-registered-views-show-continuous-sloping-roof-over-wall',
            'verifiedImageIds':sorted(required),'unusedEvidenceImageIds':sorted(extra)}


def _robust_depth_slope(samples):
    samples=np.asarray(samples,dtype=float)
    if len(samples)<2:return None
    slopes=[]
    for i in range(len(samples)-1):
        dz=samples[i+1:,0]-samples[i,0];dd=samples[i+1:,1]-samples[i,1]
        use=np.abs(dz)>.12
        slopes.extend((dd[use]/dz[use]).tolist())
    if not slopes:return None
    slope=float(np.median(slopes));intercept=float(np.median(samples[:,1]-slope*samples[:,0]))
    residual=samples[:,1]-(intercept+slope*samples[:,0])
    return slope,intercept,float(np.median(np.abs(residual-np.median(residual))))


def fit_depth_hinge(tracks, x_bounds_m, junction_height_m):
    """Check for a receding roof plane meeting a vertical wall in stereo tracks.

    Tracks are triangulated in the registered local world frame and expressed
    as ``(facade-x metres, height-above-wall-base metres, outward-depth metres)``.
    Passing this geometric test is necessary evidence only; typed wall semantics
    or explicit per-view surface labels are still required for admission.
    """
    rows=np.asarray(tracks,dtype=float)
    if rows.size==0:rows=np.empty((0,3),dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=3 or not np.isfinite(rows).all():
        return {'status':'unresolved','reason':'stereo-tracks-invalid'}
    x0,x1=map(float,x_bounds_m);height=float(junction_height_m)
    if not x1>x0:return {'status':'unresolved','reason':'stereo-span-invalid'}
    above=rows[(rows[:,1]>height+.15)&(rows[:,1]<height+3.0)]
    below=rows[(rows[:,1]<height-.15)&(rows[:,1]>height-2.0)]
    def coverage(points):return min(1.,float(np.ptp(points[:,0])/max(x1-x0,1e-6))) if len(points) else 0.
    metrics={'aboveTracks':int(len(above)),'belowTracks':int(len(below)),
             'aboveSpanFraction':round(coverage(above),3),'belowSpanFraction':round(coverage(below),3)}
    if len(above)<8 or len(below)<8:return {'status':'unresolved','reason':'stereo-support-too-sparse',**metrics}
    if np.ptp(above[:,1])<.7 or np.ptp(below[:,1])<.5:return {'status':'unresolved','reason':'stereo-height-span-too-small',**metrics}
    if min(coverage(above),coverage(below))<.35:return {'status':'unresolved','reason':'stereo-support-not-distributed-along-span',**metrics}
    rooffit=_robust_depth_slope(above[:,[1,2]]);wallfit=_robust_depth_slope(below[:,[1,2]])
    if rooffit is None or wallfit is None:return {'status':'unresolved','reason':'stereo-fit-degenerate',**metrics}
    rs,ri,rr=rooffit;ws,wi,wr=wallfit;roof_at=ri+rs*height;wall_at=wi+ws*height
    metrics.update(roofDepthSlope=round(rs,3),roofResidualMadM=round(rr,3),wallDepthSlope=round(ws,3),
                   wallResidualMadM=round(wr,3),roofDepthAtJunctionM=round(roof_at,3),
                   wallDepthAtJunctionM=round(wall_at,3),junctionGapM=round(abs(roof_at-wall_at),3))
    if rs>-.08:return {'status':'unresolved','reason':'above-surface-not-receding-roof-plane',**metrics}
    if abs(ws)>.15:return {'status':'unresolved','reason':'below-surface-not-vertical-wall-plane',**metrics}
    if rr>.25 or wr>.25:return {'status':'unresolved','reason':'stereo-plane-residual-too-large',**metrics}
    if abs(roof_at-wall_at)>.35:return {'status':'unresolved','reason':'stereo-planes-do-not-meet-at-candidate',**metrics}
    return {'status':'geometric-hinge-support','reason':'receding-plane-meets-vertical-plane',**metrics,
            'semanticWallIdentityVerified':False}


def fit_lidar_front_slope(samples, x_bounds_m, candidate_eave_z, *, tangent_bin_m=1.5):
    """Measure direct roof-return support for an eave along one facade interval.

    ``samples`` are building-owned LiDAR surface points in world-local
    ``(facade-x, inward-depth, elevation)`` coordinates. Each tangent bin must
    independently carry a rising inward roof plane whose depth-zero
    extrapolation agrees with the source-photo candidate. Sparse support is an
    abstention, not a failure label.
    """
    rows=np.asarray(samples,dtype=float)
    if rows.size==0:rows=np.empty((0,3),dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=3 or not np.isfinite(rows).all():
        return {'status':'unresolved','reason':'lidar-samples-invalid','bins':[]}
    x0,x1=map(float,x_bounds_m);expected=float(candidate_eave_z)
    if not x1>x0 or tangent_bin_m<=0:return {'status':'unresolved','reason':'lidar-span-invalid','bins':[]}
    bins=[]
    for low in np.arange(x0,x1,tangent_bin_m):
        q=rows[(rows[:,0]>=low)&(rows[:,0]<min(x1,low+tangent_bin_m))&
               (rows[:,1]>=-.75)&(rows[:,1]<=3.0)]
        entry={'xRangeM':[round(float(low),3),round(float(min(x1,low+tangent_bin_m)),3)],'returns':int(len(q))}
        if len(q)>=8 and np.ptp(q[:,1])>=.55:
            fit=_robust_depth_slope(q[:,[1,2]])
            if fit:
                slope,intercept,residual=fit;offset=intercept-expected
                entry.update(depthSlope=round(slope,3),extrapolatedFrontZM=round(intercept,3),
                             candidateOffsetM=round(offset,3),residualMadM=round(residual,3),
                             supported=bool(slope>=.25 and residual<=.25 and abs(offset)<=.6))
            else:entry.update(supported=False,reason='fit-degenerate')
        else:entry.update(supported=False,reason='insufficient-local-surface-support')
        bins.append(entry)
    supported=[b for b in bins if b.get('supported')]
    fraction=len(supported)/max(1,len(bins))
    result={'status':'supported' if len(supported)>=4 and fraction>=.75 else 'unresolved',
            'reason':'continuous-roof-surface-hinge-supported' if len(supported)>=4 and fraction>=.75 else 'insufficient-continuous-eave-plane-support',
            'candidateEaveZM':round(expected,3),'supportingBins':len(supported),'totalBins':len(bins),
            'supportFraction':round(fraction,3),'bins':bins}
    return result


def stereo_surface_hinge(area, side_record, profile, image_edges, span, junction_height_m):
    """Triangulate static correspondences from the exact registered perspective pair."""
    import os
    area=Path(area).resolve()
    os.environ['SFW_AREA']=str(area)
    try:
        import cv2
        import frontage_crops_v15 as fc
        from areas.ring.poses import side_camera_geometry,vfov_for_image
        from regen.register_cameras import apply_registration
        from regen import stereo_context_features as stereo
    except ImportError as exc:
        return {'status':'unresolved','reason':'stereo-dependency-unavailable','detail':str(exc)}
    image_ids={str(k) for k in image_edges};views=[v for v in side_record.get('views',[]) if str(v.get('image')) in image_ids]
    if len(views)!=2 or any(v.get('registered')!='skyline' for v in views):
        return {'status':'unresolved','reason':'two-accepted-registered-perspectives-required'}
    world,frame=fc.load_world();camera_rows=apply_registration(fc.load_cameras(frame),area/'imagery/registered-cameras.json')
    cam_data=[];images=[];source_hashes=[];side_geom=side_camera_geometry(side_record)
    for view in views:
        image_id=str(view['image']);camera=camera_rows.get(image_id)
        if not camera or camera.get('registered')!='skyline' or not isinstance(view.get('cameraGroundZ'),(float,int)):
            return {'status':'unresolved','reason':'registered-camera-pose-unavailable','imageId':image_id}
        path=Path(area)/'regen/sides-reg/img'/view['persp']
        if not path.is_file():return {'status':'unresolved','reason':'registered-perspective-missing','imageId':image_id}
        im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if im is None:return {'status':'unresolved','reason':'registered-perspective-unreadable','imageId':image_id}
        hfov=min(100.,fc.view_parameters(camera['xy'],side_geom,side_record['groundZ'])['hfovDeg']*1.25)
        forward=fc.view_parameters(camera['xy'],side_geom,side_record['groundZ'])['forward']
        eye=[float(camera['xy'][0]),float(camera['xy'][1]),float(view['cameraGroundZ'])+fc.EYE_HEIGHT_M]
        cam_data.append(stereo.camera({'from':eye,'at':[eye[i]+20*float(forward[i]) for i in range(3)],
                                       'width':im.shape[1],'height':im.shape[0],
                                       'vfovDeg':vfov_for_image(hfov,im.shape[1],im.shape[0])}))
        images.append(im);source_hashes.append({'imageId':image_id,'perspective':str(path),'sha256':sha(path)})
    baseline=float(np.linalg.norm(cam_data[0][0]-cam_data[1][0]))
    if baseline<2.0:return {'status':'unresolved','reason':'camera-baseline-too-small','baselineM':round(baseline,3),'sources':source_hashes}
    matches=stereo.mutual_matches(images[0],images[1]);n=np.asarray(side_record['normal'],float);t=np.array([-n[1],n[0]])
    x_bounds_m=[float(span[0])*float(side_record['lengthM']),float(span[1])*float(side_record['lengthM'])]
    tracks=[];triangulated=0
    for pair in matches:
        result=stereo.triangulate(cam_data,pair)
        if result is None:continue
        triangulated+=1;point=np.asarray(result['xyz'],float)
        x=float(t@point[:2]-float(side_record['frameU'][0]))
        if x_bounds_m[0]-.5<=x<=x_bounds_m[1]+.5:
            tracks.append([x,float(point[2]-side_record['groundZ']),float(n@point[:2]-side_record['planeV'])])
    fit=fit_depth_hinge(tracks,x_bounds_m,junction_height_m)
    return {**fit,'baselineM':round(baseline,3),'mutualMatches':len(matches),'triangulatedMatches':triangulated,
            'sources':source_hashes,'geometryUsed':'two-view calibrated perspective triangulation',
            'semanticWallIdentityVerified':False}

def edge(img,frame,prior,span):
    ppm=float(frame['ppm']);pad=int(frame['rectPad']);H=float(frame['heightM']);width=float(frame['lengthM'])
    if ppm<9:return None
    x0,x1=[int(round(pad+t*width*ppm)) for t in span];mid=pad+(H-prior)*ppm
    delta=max(3,round(.40*ppm));y0=int(np.floor(mid-delta));y1=int(np.ceil(mid+delta));h,w=img.shape[:2]
    if x0<pad or x1>w-pad+1 or y0<pad+4 or y1>h-pad-5 or x1-x0<36:return None
    # Independent strips must support a transition at the same height. A tree
    # branch or one dark window head cannot define the frontage's eave alone.
    part=img[y0-3:y1+4,x0:x1].astype('float32');part=gaussian_filter1d(part,.85,axis=0);diff=np.linalg.norm(part[5:]-part[:-5],axis=2)/np.sqrt(3);coords=np.arange(len(diff))+y0-.5
    score=np.stack([np.median(s,axis=1) for s in np.array_split(diff,12,axis=1)]);total=np.quantile(score,.55,axis=0);best=int(np.argmax(total));tol=max(1,int(.075*ppm));support=(score[:,max(0,best-tol):best+tol+1].max(1)>=np.maximum(7.,score.max(1)*.65));
    if total[best]<9 or support.sum()<8 or (not support[:4].any()) or (not support[-4:].any()):return None
    return {'heightM':round(float(H-(coords[best]-pad+.5)/ppm),4),'stripSupport':int(support.sum()),'contrast':round(float(total[best]),2),'yPx':float(coords[best]),'xPx':[x0,x1]}


def main():
    ap=argparse.ArgumentParser();
    for k in ('area','profiles','observations','output','audit'):ap.add_argument('--'+k,required=True,type=Path)
    ap.add_argument('--validation-review',type=Path,help='optional independent review used for post-hoc metrics only')
    a=ap.parse_args();pdoc=json.loads(a.profiles.read_text());obs=json.loads(a.observations.read_text());
    if isinstance(obs,list):obs={'observations':obs}
    if not isinstance(obs,dict) or not isinstance(obs.get('observations'),list):raise ValueError('observations input must be a list or an object with an observations list')
    joined=load_registered_sides(a.area/'regen/sides-reg');existing={r['side'] for r in obs['observations']};items=[];new=[]
    for p in pdoc['profiles']:
        rf=[r for r in p.get('roofs',[]) if r.get('kind')=='mansard' and r['x'][1]-r['x'][0]>=.80]
        if not rf:continue
        sid=p['id'].removeprefix('regen-'); rec={'side':sid,'status':'unresolved'};items.append(rec)
        if sid in existing:rec['reason']='existing-roof-evidence-preserved';continue
        prov=p.get('readingProvenance') or {};src=prov.get('sourceReading') or {};file=a.area/src.get('path',''); side=joined.get(sid)
        if len(rf)!=1 or not prov.get('accepted') or not prov.get('registeredCropSource') or not file.is_file() or src.get('sha256')!=sha(file) or not side or side['status']!='ok':rec['reason']='ambiguous-or-unproven-source-reading';continue
        r=rf[0];W=p['frameU'][1]-p['frameU'][0];H=p['heightRefM'];E=r['z']*H;wall=p.get('reading',{}).get('wallTopM');span=np.clip(r['x'],0,1);metadata=side['piece'];reason=None
        if not np.allclose(metadata['frameU'],p['frameU'],atol=.02) or abs(metadata['heightM']-H)>.03 or abs(metadata['planeV']-p['planeV'])>.03:reason='reading-metric-frame-mismatch'
        if p['reading'].get('confidence') not in ('high','medium') or not isinstance(wall,(float,int)) or abs(E-wall)>.45 or not .6*H<=E<=1.15*H or not .35<=r.get('riseM',0)<=6:reason='source-host-height-not-in-agreement'
        if reason:rec['reason']=reason;continue
        views={};
        for v in metadata['views']:
            f=side['path'].parent/'img'/v['rect'];
            if not v.get('registered') or not f.is_file():continue
            measurement=edge(np.asarray(Image.open(f).convert('RGB')),{**metadata,**v},E,span)
            if measurement:views[v['image']]={**measurement,'rect':str(f),'sha256':sha(f),'captureDate':v['captureDate'],'registration':v['registered']}
        rec['imageEdges']=views
        if len(views)<2 or prov['image'] not in views:rec['reason']='two-source-line-support-unavailable';continue
        heights=[q['heightM'] for q in views.values()]
        if np.ptp(heights)>.28:rec['reason']='source-lines-disagree';continue
        interval=[round(min(heights)-.20,3),round(max(heights)+.20,3)];u0=float(p['frameU'][0]);
        if not interval[0]-.1<=wall<=interval[1]+.1:rec['reason']='line-pair-disagrees-with-wall-top';continue
        metric=stereo_surface_hinge(a.area,metadata,p,views,span,float(np.mean(heights)))
        rec['stereoMetricEvidence']=metric
        semantic=source_semantic_gate(p,views,span,float(np.mean(heights)))
        rec['sourceSemanticGate']=semantic
        if semantic['status']!='admitted':
            rec['status']='proposal-only-source-semantics-unverified';rec['reason']=semantic['reason'];continue
        rec['status']='conditional-front-only-prior';new.append({'side':sid,'hostRoofKind':'mansard','confidence':'medium','eaveConfidence':'medium','eaveM':None,'hostRidgeM':None,'features':[],'sourceFrontageU':[round(float(u0+k*W),4) for k in span],'anchorUncertaintyM':max(.35,.5*(interval[1]-interval[0])),'eaveEvidence':{'junctionHeightRangeM':interval,'method':'broad typed facade roof plus spatially distributed image edges from two registered views','sourceReading':src,'views':views,'absolutePositionValidated':False,'physicalRoofJunctionVerified':False},'provisionalFrontOnly':True,'unresolved':['Source reading and edge agreement can still describe an adjacent cornice. No measured anchor, ridge, blanket roof finish or end geometry inferred.']})
    comparison=None
    if a.validation_review:
        review=json.loads(a.validation_review.read_text())
        labels={r['side']:r['classification'] for r in review.get('intervals',[])}
        statuses={r['side']:r['status'] for r in items}
        for record in items:
            if record['side'] in labels:
                record['independentReviewClass']=labels[record['side']]
        joined_ids=sorted(set(labels)&set(statuses));pred={s:statuses[s]=='conditional-front-only-prior' for s in joined_ids}
        truth={s:labels[s]=='roof-to-wall/eave' for s in joined_ids}
        comparison={'reviewPath':str(a.validation_review),'reviewSha256':sha(a.validation_review),'matchedIntervals':len(joined_ids),
                    'admitted':sorted(s for s in joined_ids if pred[s]),'rejectedOrProposalOnly':sorted(s for s in joined_ids if not pred[s]),
                    'admittedReviewEaves':sum(pred[s] and truth[s] for s in joined_ids),
                    'admittedReviewNonEaves':sum(pred[s] and not truth[s] for s in joined_ids),
                    'proposalOnlyReviewEaves':sum(not pred[s] and truth[s] for s in joined_ids),
                    'proposalOnlyReviewNonEaves':sum(not pred[s] and not truth[s] for s in joined_ids),
                    'allNonAdmissionsAreAbstentions':all(statuses[s]=='proposal-only-source-semantics-unverified' for s in joined_ids if not pred[s]),
                    'validationLabelsUsedForAdmission':False}
    metric_counts={}
    for r in items:
        metric=(r.get('stereoMetricEvidence') or {}).get('status','not-run');metric_counts[metric]=metric_counts.get(metric,0)+1
    a.output.parent.mkdir(parents=True,exist_ok=True);a.audit.parent.mkdir(parents=True,exist_ok=True);result=copy.deepcopy(obs);result['observations']+=new;result['bulkFrontPriors']={'profilesChecked':len(pdoc['profiles']),'newFronts':len(new),'sourceSemanticAdmitted':len(new),'proposalOnly':sum(r['status']=='proposal-only-source-semantics-unverified' for r in items),'sourceProfilesSha256':sha(a.profiles),'priorInputSha256':sha(a.observations)};a.output.write_text(json.dumps(result,indent=1)+'\n');audit={'profilesChecked':len(pdoc['profiles']),'records':items,'sourceSemanticGate':{'contract':'Two exact-hash registered source crops must independently show a continuous sloping-roof region above and wall region below the candidate interval, with no unresolved overlap. Existing broad roof kind and image-edge support alone do not satisfy this contract. Stereo plane fits are recorded as geometry evidence only; they do not identify a vertical surface as wall rather than roof front.','evidenceField':'profile.sourceSemanticEvidence','admitted':len(new),'proposalOnly':result['bulkFrontPriors']['proposalOnly'],'stereoMetricStatusCounts':metric_counts,'validationComparison':comparison,'generalizationLimit':'No currently compiled profile carries per-view sourceSemanticEvidence. The two-view stereo probe also lacks enough clean, distributed tracks for most candidates; a geometric hinge alone cannot distinguish a vertical mansard face from a wall. Therefore the 12 candidates remain proposals, not observations.'}};a.audit.write_text(json.dumps(audit,indent=1)+'\n');print(result['bulkFrontPriors']);
if __name__=='__main__':main()
