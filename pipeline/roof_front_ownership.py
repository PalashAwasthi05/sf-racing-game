"""STAGED pure roof front ownership. No file, camera, ID or generator introspection.

Source priors and planar closures remain provisional. Measured masks are immutable.
"""
from __future__ import annotations
import numpy as np
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import nearest_points


def _numbers(value, size):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        return None
    if any(isinstance(v, (bool, np.bool_)) for v in value):
        return None
    try: result = np.asarray(value, dtype=float)
    except (TypeError, ValueError): return None
    return result if result.shape == (size,) and np.isfinite(result).all() else None


def prepare_front_prior(observation, profile, source_ground_z, shell_base_z):
    """Validate one already joined source/profile pair; no provenance promotion.

    sourceFrontageU is absolute in this profile's tangent and is intersected
    with frameU. Missing uses legacy full span; invalid explicit input abstains.
    The caller owns exact source/hash/frame joins and supplies the source datum.
    """
    def reject(reason): return None, {'status':'unresolved','reason':reason}
    if not isinstance(observation,dict) or not isinstance(profile,dict):return reject('invalid-source-record')
    if observation.get('confidence') not in ('medium','high') or observation.get('hostRoofKind') not in ('hip','mansard'):
        return reject('no-confident-pitched-front-prior')
    if observation.get('frontageKind') == 'gable-end': return reject('gable-wall-is-not-a-hip-front')
    frame = _numbers(profile.get('frameU'),2)
    normal = _numbers(profile.get('normal'),2)
    try: plane, ground, base = float(profile['planeV']), float(source_ground_z), float(shell_base_z)
    except (TypeError, ValueError, KeyError): return reject('invalid-source-frame')
    if frame is None or frame[1] <= frame[0] or normal is None or np.linalg.norm(normal) < 1e-9 or not np.isfinite([plane,ground,base]).all():
        return reject('invalid-source-frame')
    span = frame.copy()
    if 'sourceFrontageU' in observation:
        observed = _numbers(observation['sourceFrontageU'],2)
        if observed is None or observed[1] <= observed[0]: return reject('invalid-explicit-source-frontage-span')
        span = np.array([max(frame[0], observed[0]), min(frame[1], observed[1])])
        if span[1] <= span[0]: return reject('empty-source-frontage-intersection')
    evidence = observation.get('eaveEvidence')
    if evidence is None:evidence={}
    if not isinstance(evidence,dict):return reject('invalid-source-eave-evidence')
    interval = _numbers(evidence.get('junctionHeightRangeM'),2)
    if 'junctionHeightRangeM' in evidence and interval is None:return reject('invalid-explicit-source-eave-interval')
    try: uncertainty = float(observation.get('anchorUncertaintyM',0.))
    except (TypeError, ValueError): return reject('invalid-source-uncertainty')
    if interval is None:
        try: center = float(observation['eaveM'])
        except (TypeError, ValueError, KeyError): return reject('no-bounded-source-eave')
        if not np.isfinite([center,uncertainty]).all() or uncertainty <= 0:
            return reject('unbounded-or-exact-scalar-source-eave')
        interval = np.array([center-uncertainty, center+uncertainty])
    if interval[1] <= interval[0] or not np.isfinite(uncertainty) or uncertainty < 0:
        return reject('invalid-source-eave-interval')
    uncertainty = max(uncertainty, float(np.ptp(interval)/2))
    reading = profile.get('reading') or {}
    if not isinstance(reading,dict):return reject('invalid-source-reading')
    try: wall = float(reading['wallTopM'])
    except (TypeError, ValueError, KeyError): return reject('missing-source-wall-top')
    if reading.get('confidence') not in ('medium','high') or not np.isfinite(wall):
        return reject('no-confident-source-wall-top')
    if wall < interval[0]-uncertainty or wall > interval[1]+uncertainty:
        return reject('source-wall-top-and-eave-prior-disagree')
    level = ground + interval[1]
    if level < base-1e-8: return reject('source-eave-below-shell-floor-needs-separate-base-resolution')
    normal = normal/np.linalg.norm(normal); tangent = np.array([-normal[1],normal[0]])
    prior = {'normal':normal,'tangent':tangent,'planeV':plane,'frameU':span.tolist(),
        'profileFrameU':frame.tolist(),'sourceFrontageU':span.tolist(),'levelZ':level,'zBase':ground,
        'heightIntervalM':interval.tolist(),'eaveZRange':(ground+interval).tolist(),
        'wallTopZ':ground+wall,'wallTopM':wall,'roofKind':observation['hostRoofKind'],
        'side':observation.get('side'),'profileId':profile.get('id'),
        'confidence':observation['confidence'],'eaveConfidence':observation.get('eaveConfidence'),
        'uncertaintyM':uncertainty,'sourceReading':profile.get('readingProvenance'),
        'sourceEaveEvidence':evidence,'provenance':observation.get('provenance'),
        'evidenceStatus':'provisional-source-interval','cameraPositionValidated':False}
    return prior, {'status':'provisional-source-prior','sourceFrontageU':span.tolist(),
        'confidenceUpgraded':False,'measuredAnchor':False}


def _solve_planar_cap(xy,faces,z,known,unknown,priors,footprint,validated_creases,cell_m=.5,protected_domains=()):
    """Bound a local hip front by a source eave and measured opposing sides.

    The minimum cap slope satisfying all native anchor inequalities is used;
    its intersection with those two sides determines the bounded front depth.
    This is a provisional local envelope, not a whole-footprint roof template.
    """
    xy=np.asarray(xy);z=np.asarray(z);known=np.asarray(known,bool);unknown=np.asarray(unknown,bool)
    boundary=np.array([footprint.boundary.distance(Point(p))<1e-7 for p in xy]);protected=np.zeros(len(xy),bool)
    for domain in protected_domains:protected|=np.array([domain.covers(Point(p)) for p in xy])
    out=z.copy();photo=np.zeros(len(xy),bool);records=[]
    for prior in priors:
        n,t=prior['normal'],prior['tangent'];u=xy@t;depth=prior['planeV']-xy@n;lo,hi=prior['frameU'];on=boundary&(abs(depth)<.1)&(u>=lo-.12)&(u<=hi+.12)&unknown&~protected
        if on.sum()<2:continue
        offset=float(np.median(depth[on]));d=depth-offset;E=prior['levelZ'];anchors=known&(u>=lo-.12)&(u<=hi+.12)&(d>cell_m/4)
        if not anchors.any():continue
        k=max(0.,float(np.max((z[anchors]-E)/d[anchors])))+1e-7
        if not .0<k<=3.7:
            records.append({'side':prior['side'],'status':'source-eave-incompatible-with-native-anchor-envelope','requiredSlope':k});continue
        candidates=[]
        for crease in validated_creases:
            planes=np.asarray(crease['model']['planes'],float)
            if planes.shape!=(2,3):continue
            tangent_slopes=planes[:,:2]@t
            if tangent_slopes[0]*tangent_slopes[1]>=0:continue
            # The crease itself has already passed measured spatial support
            # and opposing-plane tests in the frozen native reconstruction.
            A=planes[:,:2]@t;B=-planes[:,:2]@n;C=planes[:,:2]@(n*(prior['planeV']-offset))+planes[:,2]-E
            matrix=np.c_[A,B-k]
            if abs(np.linalg.det(matrix))<1e-8:continue
            contact_u,contact_depth=np.linalg.solve(matrix,-C)
            if not lo-.12<=contact_u<=hi+.12 or contact_depth<=0:continue
            world_contact=n*(prior['planeV']-offset-contact_depth)+t*contact_u
            distance=float(crease['zone'].distance(Point(world_contact)))
            candidates.append((distance,contact_depth,planes,crease,contact_u))
        if not candidates:
            records.append({'side':prior['side'],'status':'no-spatially-validated-opposing-roof-sides'});continue
        distance,contact_depth,planes,crease,contact_u=min(candidates,key=lambda q:(q[0],q[1]))
        # The new cap may connect only up to the first supported ridge zone,
        # with the native crease continuation budget retained as uncertainty.
        budget=float(crease.get('audit',{}).get('maximumDirectSupportContinuationM',np.sqrt(2)*cell_m))
        if distance>budget+cell_m:
            records.append({'side':prior['side'],'status':'cap-contact-outside-bounded-supported-ridge','contactDistanceM':distance,'continuationBudgetM':budget});continue
        envelope=(np.c_[xy,np.ones(len(xy))]@planes.T).min(1);cap=E+k*d
        local=unknown&~protected&(u>=lo-.12)&(u<=hi+.12)&(d>=-.1)&(d<=contact_depth+cell_m)
        predicted=np.minimum(envelope,cap)
        # Actual measured anchors keep their exact values even where finite
        # cell ownership extends slightly across a fitted crease.
        out[local]=predicted[local];photo|=on
        records.append({'side':prior['side'],'profileId':prior['profileId'],'status':'provisional-measured-side-bounded-front-cap',
            'sourceIntervalM':prior['heightIntervalM'],'provisionalHeightM':E-prior.get('zBase',0.),'confidence':prior['confidence'],'eaveConfidence':prior['eaveConfidence'],'uncertaintyM':prior['uncertaintyM'],
            'sourceReading':prior['sourceReading'],'capSlopeFromAnchorInequalities':k,'capPitchDegConditional':float(np.degrees(np.arctan(k))),
            'anchorConstraints':int(anchors.sum()),'changedUnknownVertices':int((local&(abs(out-z)>1e-7)).sum()),
            'capRidgeContactInwardDepthM':float(contact_depth),'capRidgeContactSourceU':float(contact_u-lo),'capContactDistanceToValidatedRidgeZoneM':distance,
            'nativeOpposingSidePlanes':planes.tolist(),'nativeCreaseEvidence':crease.get('audit',{}),'nativeBoundaryRegistrationOffsetM':offset,
            'appliedBoundaryNodes':np.flatnonzero(on).tolist(),'semantic':'Provisional local front plane joins uncertain source eave to independently supported side planes; it does not lower measured anchors or establish camera position.'})
    if not photo.any():return None,{'status':'no-admissible-planar-front-cap','records':records}
    out[known|protected]=z[known|protected]
    return (out,known|photo,unknown&~photo),{'status':'conditional-local-planar-front-cap','records':records,'appliedBoundaryVertices':int(photo.sum()),
        'changedUnknownVertices':int((abs(out-z)>1e-7).sum()),'maximumLoweringM':float(np.maximum(z-out,0).max()),
        'maximumRaisingOfUnmeasuredFillM':float(np.maximum(out-z,0).max()),'measuredAnchorMaximumChangeM':float(abs(out[known]-z[known]).max()) if known.any() else 0.,
        'protectedSourceMaximumChangeM':float(abs(out[protected]-z[protected]).max()) if protected.any() else 0.,'provisionalPhotoNodesAreNotVerifiedMeasuredAnchors':True}


def solve_front_ownership(xy, faces, z, measured_anchor_mask, unknown_mask,
                          regions, validated_creases, footprint, front_priors,
                          protected_domains=(), cell_m=.5):
    """Return a provisional local face and separate masks; inputs are unchanged.

    The caller supplies final native baseline heights after native grade/host
    resolution. Existing source parts/components must be resolved against that
    baseline and held fixed while this result is composed.
    """
    xy = np.asarray(xy,float); faces = np.asarray(faces,int); before = np.asarray(z,float)
    measured = np.asarray(measured_anchor_mask,bool); unknown = np.asarray(unknown_mask,bool)
    if xy.shape != (len(before),2) or measured.shape != before.shape or unknown.shape != before.shape or not np.isfinite(np.r_[xy.ravel(),before]).all():
        raise ValueError('malformed native continuation state')
    if np.any(measured & unknown): raise ValueError('measured and unknown masks overlap')
    solved,audit = _solve_planar_cap(xy,faces,before,measured,unknown,front_priors,
        footprint,validated_creases,cell_m=cell_m,protected_domains=protected_domains)
    if solved is None:
        return {'z':before.copy(),'measured_anchor_mask':measured.copy(),
            'provisional_boundary_mask':np.zeros(len(before),bool),
            'changed_unknown_mask':np.zeros(len(before),bool),'audit':audit}
    after,solver_fixed,solver_unknown = solved
    provisional = solver_fixed & ~measured
    changed = np.abs(after-before) > 1e-7
    if np.any(changed & measured): raise ValueError('front solver modified measured roof anchors')
    audit = {**audit,'confidenceUpgraded':False,'measuredAnchorMaskUnchanged':True,
        'provisionalConstraintsAreNotMeasuredAnchors':True}
    return {'z':after,'measured_anchor_mask':measured.copy(),
        'provisional_boundary_mask':provisional,'changed_unknown_mask':changed & unknown,
        'audit':audit}


def continue_supported_plane_to_observed_perimeter(xy, faces, z, measured_anchor_mask,
        unknown_mask, regions, footprint, front_priors, protected_domains=(),
        cell_m=.5, maximum_grade=2.5):
    """Opt-in bounded continuation of a validated plane to its OWN seen edge.

    A front observation cannot authorize a changed unobserved flank. Each
    candidate needs an explicit source-frame span, a broad two-dimensional
    inlier hull bracketing that span, and at most one grid diagonal of outward
    continuation. Courtyards and source-owned domains remain untouched.
    """
    xy=np.asarray(xy,float); faces=np.asarray(faces,int); before=np.asarray(z,float)
    measured=np.asarray(measured_anchor_mask,bool); unknown=np.asarray(unknown_mask,bool)
    out=before.copy(); records=[]; budget=float(np.sqrt(2)*cell_m)
    boundary=np.array([footprint.boundary.distance(Point(p))<1e-7 for p in xy])
    protected=np.zeros(len(xy),bool)
    for domain in protected_domains: protected|=np.array([domain.covers(Point(p)) for p in xy])
    def face_grades(values):
        tri=xy[faces]; delta=tri[:,1:]-tri[:,:1]
        determinant=delta[:,0,0]*delta[:,1,1]-delta[:,0,1]*delta[:,1,0]
        dz=values[faces[:,1:]]-values[faces[:,:1]]; valid=abs(determinant)>1e-10
        grades=np.zeros(len(faces))
        gx=(dz[valid,0]*delta[valid,1,1]-delta[valid,0,1]*dz[valid,1])/determinant[valid]
        gy=(delta[valid,0,0]*dz[valid,1]-dz[valid,0]*delta[valid,1,0])/determinant[valid]
        grades[valid]=np.hypot(gx,gy);return grades
    original_grades=face_grades(before)
    for prior in front_priors:
        n,t=np.asarray(prior['normal']),np.asarray(prior['tangent']);lo,hi=prior['frameU']
        along=xy@t; depth=prior['planeV']-xy@n
        edge=boundary&unknown&~protected&(abs(depth)<=.1)&(along>=lo)&(along<=hi)
        if not edge.any():continue
        for index,region in enumerate(regions):
            report={'profileId':prior.get('profileId'),'region':index}
            validation=region.get('validation') or {}
            points=np.asarray(region.get('inlierXY',[]),float)
            plane=_numbers(region.get('plane'),3)
            if validation.get('status')!='heldout-validated' or plane is None:
                records.append({**report,'status':'unresolved-no-native-heldout-plane-validation'});continue
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
                records.append({**report,'status':'unresolved-no-broad-inlier-support'});continue
            point_u=points@t; point_d=prior['planeV']-points@n
            if point_u.min()>lo+budget or point_u.max()<hi-budget or np.ptp(point_d)<cell_m:
                records.append({**report,'status':'unresolved-inliers-do-not-bracket-observed-edge-in-two-dimensions'});continue
            hull=MultiPoint(points).convex_hull
            if hull.geom_type!='Polygon' or hull.area<=0:
                records.append({**report,'status':'unresolved-collinear-inlier-strip'});continue
            predicted=np.c_[xy,np.ones(len(xy))]@plane
            interval=np.asarray(prior['eaveZRange'],float)
            seeds=np.flatnonzero(edge&(predicted>=interval[0]-1e-7)&(predicted<=interval[1]+1e-7))
            seeds=np.array([i for i in seeds if hull.distance(Point(xy[i]))<=budget+1e-8],int)
            if not len(seeds):
                records.append({**report,'status':'unresolved-remote-or-eave-incompatible-plane'});continue
            candidates=np.flatnonzero(unknown&~protected&(along>=lo)&(along<=hi)&(depth>=-.1)&(depth<=point_d.max()))
            measured_domain=region['geometry'].intersection(footprint)
            changed=[]
            for vertex in candidates:
                if hull.distance(Point(xy[vertex]))>budget+1e-8:continue
                # Intersect the SAME inward normal, rather than borrowing
                # lateral points around the corner of a courtyard/notch.
                ray=LineString([xy[vertex],xy[vertex]-n*max(cell_m,float(point_d.max()-depth[vertex]))])
                support=measured_domain.intersection(ray)
                if support.is_empty:continue
                anchor=nearest_points(Point(xy[vertex]),support)[1]
                segment=LineString([xy[vertex],anchor.coords[0]])
                if not footprint.buffer(1e-8).covers(segment):continue
                if any(domain.intersection(segment).length>1e-8 for domain in protected_domains):continue
                changed.append(int(vertex))
            trial=out.copy();trial[changed]=predicted[changed];trial[measured|protected]=before[measured|protected]
            touched=np.any(np.isin(faces,changed),axis=1);new_grade=face_grades(trial)
            allowed=np.maximum(original_grades,max(maximum_grade,float(np.linalg.norm(plane[:2]))))
            if np.any(touched&(new_grade>allowed+1e-6)):
                records.append({**report,'status':'unresolved-would-create-new-steep-transition'});continue
            out=trial
            records.append({**report,'status':'provisional-bounded-own-perimeter-continuation','boundaryNodes':seeds.tolist(),
                'changedUnknownVertices':len(changed),'maximumContinuationM':max(hull.distance(Point(xy[i])) for i in changed) if changed else 0.,
                'continuationBudgetM':budget,'confidence':prior['confidence'],'eaveConfidence':prior.get('eaveConfidence'),
                'provisionalNotMeasured':True})
    return {'z':out,'measured_anchor_mask':measured.copy(),'changed_unknown_mask':(abs(out-before)>1e-7)&unknown,
        'audit':{'status':'opt-in-supported-own-perimeter-continuation','records':records,
            'measuredAnchorMaximumChangeM':float(abs(out[measured]-before[measured]).max()) if measured.any() else 0.,
            'confidenceUpgraded':False,'footprintUnchanged':True}}


def merge_preserved_components(baseline_specs, baseline_audit, candidate_specs, candidate_audit):
    """Keep existing source component coordinates; admit only newly supported priors.

    Source IDs below are joins, never selectors. Existing inset/height/aperture
    parameters are returned verbatim; a changed host cannot rescale them.
    """
    if len(baseline_specs)!=len(baseline_audit.get('accepted',[])) or len(candidate_specs)!=len(candidate_audit.get('accepted',[])):
        raise ValueError('component spec/evidence alignment mismatch')
    def key(spec,evidence):
        facing=_numbers(spec.get('facing'),2);center=_numbers(spec.get('xy'),2)
        if facing is None or center is None:raise ValueError('invalid component source frame')
        norm=np.linalg.norm(facing)
        if norm<1e-9:raise ValueError('invalid component facing')
        facing=facing/norm;tangent=np.array([-facing[1],facing[0]])
        return (evidence.get('side'),evidence.get('profileId'),spec.get('kind'),
            round(float(center@tangent),5),round(float(spec.get('widthM',0)),5),
            round(float(spec.get('bottomM',spec.get('eaveM',0))),5),round(float(spec.get('ridgeM',0)),5))
    keys={key(s,e) for s,e in zip(baseline_specs,baseline_audit.get('accepted',[]))}
    specs=list(baseline_specs);accepted=list(baseline_audit.get('accepted',[]));added=[]
    for spec,evidence in zip(candidate_specs,candidate_audit.get('accepted',[])):
        identity=key(spec,evidence)
        if identity in keys:continue
        specs.append(spec);record={**evidence,'provisionalHostSupport':True,'confidenceUpgraded':False}
        accepted.append(record);added.append(record);keys.add(identity)
    rejected=[]
    for record in candidate_audit.get('rejected',[]):
        retained=any(e.get('side')==record.get('side') and e.get('featureKind')==record.get('kind') for e in baseline_audit.get('accepted',[]))
        rejected.append({**record,**({'baselineSourceGeometryRetained':True} if retained else {})})
    return specs,{'status':'partial-provisional' if rejected or added else baseline_audit.get('status','resolved'),
        'accepted':accepted,'rejected':rejected,'baselineSupportAudit':baseline_audit,
        'newlySupportedSourcePriors':added,'existingSourceParametersUnchanged':True,
        'confidenceUpgraded':False}

