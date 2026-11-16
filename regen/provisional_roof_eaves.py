"""External provisional wall/eave ownership on otherwise unknown roof fill.

This is a conditional best-fit hypothesis, never verified roof evidence. The
source interval bounds the eave; measured native planes bound the local apron.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from shapely.geometry import Point,LineString
from pipeline.roof_assemblies import _profiles_for_side


def source_priors(observations,profiles,identity,z0,base_z):
    priors=[];rejected=[]
    for obs in observations:
        joined=_profiles_for_side(profiles,obs.get('side'),identity,identity)
        if not joined:continue
        reason=None
        if obs.get('confidence') not in ('medium','high') or obs.get('hostRoofKind') not in ('hip','mansard') or obs.get('frontageKind')=='gable-end':reason='no-confident-non-gable-pitched-host-observation'
        evidence=obs.get('eaveEvidence') or {};interval=evidence.get('junctionHeightRangeM')
        if not isinstance(interval,list) or len(interval)!=2:reason=reason or 'no-bounded-source-eave-interval'
        if reason:rejected.append({'side':obs.get('side'),'reason':reason});continue
        interval=np.asarray(interval,float)
        if not np.isfinite(interval).all() or interval[1]<interval[0]:rejected.append({'side':obs.get('side'),'reason':'invalid-eave-interval'});continue
        uncertainty=max(float(obs.get('anchorUncertaintyM',0)),float((interval[1]-interval[0])/2))
        for profile in joined:
            reading=profile.get('reading') or {};wall=reading.get('wallTopM')
            if not isinstance(wall,(int,float)) or not np.isfinite(wall) or reading.get('confidence') not in ('medium','high'):
                rejected.append({'side':obs['side'],'profile':profile['id'],'reason':'no-confident-source-wall-top'});continue
            if wall<interval[0]-uncertainty or wall>interval[1]+uncertainty:
                rejected.append({'side':obs['side'],'profile':profile['id'],'reason':'source-wall-top-and-eave-interval-disagree','wallTopM':wall,'eaveIntervalM':interval.tolist()});continue
            # Least change to an over-high native boundary uses the highest
            # admissible source level, not a fabricated exact midpoint.
            level=max(base_z,z0+float(interval[1]))
            if level>z0+interval[1]+1e-7:rejected.append({'side':obs['side'],'reason':'source-eave-interval-below-current-shell-floor'});continue
            n=np.asarray(profile['normal'],float);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
            priors.append({'side':obs['side'],'profileId':profile['id'],'normal':n,'tangent':t,'planeV':float(profile['planeV']),
                'frameU':list(profile['frameU']),'levelZ':level,'zBase':z0,'heightIntervalM':interval.tolist(),'confidence':obs['confidence'],
                'eaveConfidence':obs.get('eaveConfidence'),'uncertaintyM':uncertainty,'wallTopM':wall,
                'sourceReading':profile.get('readingProvenance'),'sourceEaveEvidence':evidence})
    return priors,rejected


def solve_unknown_apron(xy,faces,z,known,unknown,priors,footprint,cell_m=.5,protected_domains=(),support_mode='grid-strip'):
    xy=np.asarray(xy);faces=np.asarray(faces,int);z=np.asarray(z,float);known=np.asarray(known,bool);unknown=np.asarray(unknown,bool)
    boundary=np.array([footprint.boundary.distance(Point(p))<1e-7 for p in xy]);protected=np.zeros(len(xy),bool)
    for domain in protected_domains:protected|=np.array([domain.covers(Point(p)) for p in xy])
    prior_nodes=np.zeros(len(xy),bool);levels=z.copy();zone=np.zeros(len(xy),bool);records=[]
    # Width tolerance follows the native grid diagonal. A source eave cannot
    # borrow a roof on the far side of a courtyard or a different frontage.
    lateral_tolerance=np.sqrt(2)*cell_m
    for prior in priors:
        n,t=prior['normal'],prior['tangent'];u=xy@t;depth=prior['planeV']-xy@n
        lo,hi=prior['frameU'];on=boundary&(abs(depth)<=.10)&(u>=lo-.12)&(u<=hi+.12)
        seeds=np.flatnonzero(on&unknown&~protected&(z>prior['levelZ']+.01));support=[];rejected=[]
        for index in seeds:
            lateral=((u>=lo)&(u<=hi)) if support_mode=='observed-frontage' else abs(u-u[index])<=lateral_tolerance
            eligible=np.flatnonzero(known&~protected&lateral&(depth>=depth[index]+cell_m/2))
            eligible=sorted(eligible,key=lambda j:float(np.linalg.norm(xy[j]-xy[index])))
            selected=next((j for j in eligible if footprint.buffer(1e-7).covers(LineString([xy[index],xy[j]]))),None)
            if selected is None:rejected.append(int(index));continue
            support.append((int(index),int(selected),float(u[index]),float(depth[selected])))
        if len(support)<2:
            records.append({'side':prior['side'],'profileId':prior['profileId'],'status':'no-bounded-corresponding-interior-support','unknownBoundaryCandidates':len(seeds),'supportedCandidates':len(support)});continue
        support.sort(key=lambda q:q[2]);accepted=[]
        # Use only contiguous source-side segments. No fit bridges an absent
        # boundary run; each piece ends at actual measured interior support.
        for first,second in zip(support,support[1:]):
            if second[2]-first[2]>2*lateral_tolerance:continue
            between=(u>=first[2]-1e-7)&(u<=second[2]+1e-7)&(depth>=-.1)
            end_depth=np.interp(u,[first[2],second[2]],[first[3],second[3]])
            local=between&(depth<=end_depth+cell_m/2)&unknown&~protected
            zone|=local;accepted.extend([first,second])
        accepted={q[0]:q for q in accepted}
        for index in accepted:prior_nodes[index]=True;levels[index]=min(levels[index],prior['levelZ'])
        records.append({'side':prior['side'],'profileId':prior['profileId'],'status':'provisional-source-eave-interval' if accepted else 'no-contiguous-supported-source-run',
            'sourceIntervalM':prior['heightIntervalM'],'provisionalLevelZ':prior['levelZ'],'provisionalHeightM':prior['levelZ']-prior.get('zBase',0.),'confidence':prior['confidence'],'eaveConfidence':prior['eaveConfidence'],
            'uncertaintyM':prior['uncertaintyM'],'sourceWallTopM':prior['wallTopM'],'sourceReading':prior['sourceReading'],
            'unknownBoundaryCandidates':len(seeds),'appliedBoundaryNodes':sorted(accepted),'rejectedBoundaryNodesWithoutInteriorSupport':rejected,
            'interiorSupports':[{'boundaryVertex':q[0],'measuredVertex':q[1],'measuredXYZ':[float(xy[q[1],0]),float(xy[q[1],1]),float(z[q[1]])],
                'sourceToSupportDistanceM':float(np.linalg.norm(xy[q[0]]-xy[q[1]])),'inwardSupportDepthM':q[3],
                'supportLateralOffsetM':float(abs(u[q[1]]-u[q[0]]))} for q in accepted.values()],
            'supportMode':support_mode,
            'entireMatchedBoundaryUnknown':bool(on.any() and unknown[on].all()),'allOverheightBoundaryCandidatesApplied':len(accepted)==len(seeds),
            'semantic':'wall ends at uncertain photographed eave; the interior roof plane is independently measured; this connecting apron remains provisional'})
    if not prior_nodes.any():return z.copy(),known.copy(),unknown.copy(),{'status':'no-applicable-provisional-eave','records':records,'appliedBoundaryVertices':0}
    zone&=unknown&~known&~protected;zone|=prior_nodes
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0);lengths=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1);weights=1/np.maximum(lengths,1e-6)
    adjacency=coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(xy),len(xy))).tocsr();laplacian=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    free=zone&~prior_nodes;fixed=~free;out=z.copy();out[prior_nodes]=levels[prior_nodes]
    if free.any():out[free]=spsolve(laplacian[free][:,free],-laplacian[free][:,fixed]@out[fixed])
    # A lower eave cannot authorize raising any provisional roof cell.
    out[zone]=np.minimum(out[zone],z[zone]);out[known|protected]=z[known|protected]
    return out,known|prior_nodes,unknown&~prior_nodes,{'status':'conditional-local-piecewise-apron','records':records,
        'appliedBoundaryVertices':int(prior_nodes.sum()),'changedUnknownVertices':int((abs(out-z)>1e-7).sum()),
        'maximumLoweringM':float((z-out).max()),'measuredAnchorMaximumChangeM':float(abs(out[known]-z[known]).max()) if known.any() else 0.,
        'protectedSourceMaximumChangeM':float(abs(out[protected]-z[protected]).max()) if protected.any() else 0.,
        'unchangedOutsideLocalApron':bool(np.array_equal(out[~zone],z[~zone])),
        'provisionalPhotoNodesAreNotVerifiedMeasuredAnchors':True}


def solve_planar_front_cap(xy,faces,z,known,unknown,priors,footprint,validated_creases,cell_m=.5,protected_domains=()):
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
