"""Staged pure source-span wall outline ownership on unmeasured roof closure.

An annotated front silhouette is a provisional wall boundary, not a measured
roof surface or a whole-footprint roof style. Never moves native/source anchors.
"""
import re
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from shapely.geometry import Point,LineString


def _polygonal_area(geometry):
    """Retain all polygon components/holes and discard line/point remnants."""
    from shapely.ops import unary_union
    parts=[]
    def collect(item):
        if item is None or item.is_empty:return
        if item.geom_type=='Polygon':parts.append(item)
        elif item.geom_type in ('MultiPolygon','GeometryCollection'):
            for part in item.geoms:collect(part)
    collect(geometry)
    if not parts:return None
    area=unary_union(parts)
    return area if not area.is_empty else None


def prepare_gable_prior(profile,z_base):
    """Prepare an already joined lint-preserved source outline; no file I/O.

    Source heights must already be in the model-relative facade datum. The
    caller verifies the profile's reading/registration hashes before admission.
    The legacy outline lacks pixel error, so it remains an uncertain prior.
    """
    if not isinstance(profile,dict):return None,{'status':'unresolved','reason':'invalid-profile-record','confidenceUpgraded':False}
    audit={'profileId':profile.get('id'),'status':'unresolved','confidenceUpgraded':False}
    reading=profile.get('reading') or {};provenance=profile.get('readingProvenance') or {}
    try:
        if not profile.get('gableOutline'):raise ValueError('no-lint-preserved-gable-outline')
        if reading.get('confidence') not in ('medium','high'):raise ValueError('source-confidence-insufficient')
        source=provenance.get('sourceReading') or {}
        if provenance.get('status')!='ok' or provenance.get('accepted') is not True:raise ValueError('source-provenance-not-accepted')
        if provenance.get('registeredCropSource') is not True:raise ValueError('source-was-not-registered-crop')
        if not re.fullmatch('[0-9a-f]{64}',str(source.get('sha256',''))) or not source.get('path'):raise ValueError('missing-source-reading-identity')
        if 'mapillary:'+str(provenance.get('image')) not in profile.get('sourceIds',[]):raise ValueError('source-image-not-joined-to-profile')
        a=np.asarray(profile['gableOutline'],float);frame=np.asarray(profile['frameU'],float);h=float(profile['heightRefM']);wall=float(reading['wallTopM']);n=np.asarray(profile['normal'],float)
        if frame.shape!=(2,) or n.shape!=(2,) or a.ndim!=2 or a.shape[1]!=2 or len(a)<3:raise ValueError('malformed-source-frame-or-outline')
        if not np.isfinite([*frame,h,wall,*n,profile['planeV'],z_base]).all() or h<=0 or frame[1]<=frame[0] or np.linalg.norm(n)<1e-8:raise ValueError('invalid-source-frame')
        if not np.isfinite(a).all():raise ValueError('nonfinite-source-outline')
        if np.any(np.diff(a[:,0])<0):raise ValueError('nonmonotone-source-outline-requires-polygonal-front')
        if np.any(np.diff(a[:,0])==0):raise ValueError('vertical-step-source-outline-requires-riser-topology')
        n=n/np.linalg.norm(n);t=np.array([-n[1],n[0]]);trace=np.c_[frame[0]+a[:,0]*(frame[1]-frame[0]),z_base+a[:,1]*h]
        if trace[0,0]<frame[0]-.02 or trace[-1,0]>frame[1]+.02:raise ValueError('outline-outside-source-frame')
        if np.max(trace[1:-1,1])<=max(trace[0,1],trace[-1,1])+.5:raise ValueError('no-distinct-source-front-gable')
        if max(abs(trace[[0,-1],1]-z_base-wall))>.75:raise ValueError('source-outline-eaves-disagree-with-wall-top')
        prior={'profileId':profile['id'],'sourceOutlineIndex':None,'normal':n,'tangent':t,'planeV':float(profile['planeV']),'traceUZ':trace,'confidence':reading['confidence'],'sourceIds':profile.get('sourceIds',[]),'sourceReading':provenance,'uncertaintyM':None,'uncertaintyReason':'legacy annotation has no pixel uncertainty; provisional numerical boundary only, never a measured anchor','semanticGate':'lint-preserved-gable-outline-with-source-wall-top-agreement'}
        return prior,{**audit,'status':'prepared-provisional-source-outline','observedAbsoluteFrontageU':trace[[0,-1],0].tolist(),'legacyPixelUncertaintyUnquantified':True}
    except (ValueError,TypeError,KeyError,IndexError,AttributeError) as error:return None,{**audit,'reason':str(error)}


def outline_mesh_points(priors,footprint,plane_tolerance=.1):
    """Insert actual source span breaks only on the matching exterior edge.

    Topology points are not eave anchors. Courtyard rings are excluded: a front
    outline cannot acquire an unrelated interior perimeter by proximity.
    """
    from shapely.ops import nearest_points,unary_union
    footprint=_polygonal_area(footprint)
    if footprint is None:return [],[{'status':'unresolved-no-polygonal-footprint'}]
    exteriors=unary_union([p.exterior for p in ([footprint] if footprint.geom_type=='Polygon' else footprint.geoms) if p.geom_type=='Polygon'])
    points=[];audit=[]
    for prior in priors:
        for u,z in prior['traceUZ']:
            q=prior['normal']*prior['planeV']+prior['tangent']*u;p=Point(q);distance=float(exteriors.distance(p))
            if distance<=plane_tolerance:
                snapped=np.asarray(nearest_points(exteriors,p)[0].coords[0]);points.append(snapped);audit.append({'profileId':prior['profileId'],'sourceU':float(u),'registrationDistanceM':distance,'topologyOnly':True})
    return points,audit


def solve(xy,faces,z,known,unknown,priors,footprint,protected_domains=(),cell_m=.5):
    xy=np.asarray(xy,float);faces=np.asarray(faces,int);z=np.asarray(z,float);known=np.asarray(known,bool);unknown=np.asarray(unknown,bool)
    footprint=_polygonal_area(footprint)
    if footprint is None:return z.copy(),np.zeros(len(z),bool),np.zeros(len(z),bool),{'status':'unresolved-no-polygonal-footprint','appliedBoundaryVertices':0,'records':[]}
    used=np.zeros(len(xy),bool);used[np.unique(faces)]=True
    from shapely.ops import unary_union
    exterior=unary_union([p.exterior for p in ([footprint] if footprint.geom_type=='Polygon' else footprint.geoms) if p.geom_type=='Polygon'])
    boundary=np.array([exterior.distance(Point(p))<1e-7 for p in xy])&used;protected=np.zeros(len(xy),bool)
    for domain in protected_domains:protected|=np.array([domain.covers(Point(p)) for p in xy])
    levels=z.copy();photo=np.zeros(len(xy),bool);zone=np.zeros(len(xy),bool);records=[]
    for prior in priors:
        n,t=prior['normal'],prior['tangent'];u=xy@t;d=prior['planeV']-xy@n;trace=prior['traceUZ'];lo,hi=trace[[0,-1],0]
        on=boundary&(abs(d)<.1)&(u>=lo-1e-6)&(u<=hi+1e-6)
        candidates=np.flatnonzero(on&unknown&~known&~protected);supports=[]
        for j in candidates:
            eligible=np.flatnonzero(known&~protected&(abs(u-u[j])<=np.sqrt(2)*cell_m)&(d>d[j]+cell_m/2))
            eligible=sorted(eligible,key=lambda k:float(np.linalg.norm(xy[k]-xy[j])))
            k=next((k for k in eligible if footprint.buffer(1e-7).covers(LineString([xy[j],xy[k]]))),None)
            if k is not None:supports.append((int(j),int(k),float(u[j]),float(d[k])))
        supports.sort(key=lambda q:q[2]);accepted={}
        for a,b in zip(supports,supports[1:]):
            if b[2]-a[2]>2*np.sqrt(2)*cell_m:continue
            local=(u>=a[2]-1e-7)&(u<=b[2]+1e-7)&(d>=-.1)&(d<=np.interp(u,[a[2],b[2]],[a[3],b[3]])+1e-7)&unknown&~known&~protected
            zone|=local;accepted[a[0]]=a;accepted[b[0]]=b
        for j in accepted:
            target=float(np.interp(u[j],trace[:,0],trace[:,1]))
            if photo[j] and abs(levels[j]-target)>1e-6:return z.copy(),np.zeros(len(z),bool),np.zeros(len(z),bool),{'status':'unresolved-conflicting-source-outlines','records':records,'appliedBoundaryVertices':0}
            photo[j]=True;levels[j]=target
        records.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in prior.items()}|{'status':'provisional-source-span-outline' if accepted else 'unresolved-no-contiguous-corresponding-measured-support','unknownBoundaryCandidates':len(candidates),'measuredBoundaryPreserved':int((on&known).sum()),'appliedBoundaryNodes':sorted(accepted),'interiorSupportNodes':sorted({q[1] for q in accepted.values()}),'observedAbsoluteFrontageU':[float(lo),float(hi)]})
    if not photo.any():return z.copy(),np.zeros(len(z),bool),np.zeros(len(z),bool),{'status':'no-admissible-source-outline','records':records,'appliedBoundaryVertices':0}
    zone&=unknown&~known&~protected&used;zone|=photo
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1);weights=1/np.maximum(length,1e-6)
    adj=coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(z),len(z))).tocsr();lap=diags(np.asarray(adj.sum(1)).ravel())-adj
    free=zone&~photo;fixed=~free;out=z.copy();out[photo]=levels[photo]
    if free.any():out[free]=spsolve(lap[free][:,free],-lap[free][:,fixed]@out[fixed])
    if not np.isfinite(out).all():return z.copy(),np.zeros(len(z),bool),np.zeros(len(z),bool),{'status':'unresolved-disconnected-solve','records':records,'appliedBoundaryVertices':0}
    out[known|protected]=z[known|protected];changed=abs(out-z)>1e-8
    return out,photo,changed,{'status':'external-provisional-source-outline-ownership','records':records,'appliedBoundaryVertices':int(photo.sum()),'changedUnknownVertices':int(changed.sum()),'maximumLoweringM':float(np.maximum(z-out,0).max()),'maximumRaisingM':float(np.maximum(out-z,0).max()),'measuredAnchorMaximumChangeM':float(abs(out[known]-z[known]).max()) if known.any() else 0.,'protectedMaximumChangeM':float(abs(out[protected]-z[protected]).max()) if protected.any() else 0.,'outsideLocalSupportDomainMaximumChangeM':float(abs(out[~zone]-z[~zone]).max()) if (~zone).any() else 0.,'provisionalNotMeasured':True}
