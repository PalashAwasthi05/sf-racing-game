"""External source-span wall outline ownership on unmeasured roof closure.

An annotated front silhouette is a provisional wall boundary, not a measured
roof surface or a whole-footprint roof style. Never moves native/source anchors.
"""
import re
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
from shapely.geometry import Point,LineString


def source_priors(profiles,identity,source_id,z0):
    priors=[];rejected=[]
    for p in profiles:
        match=p.get('match') or {};part=match.get('partId')
        if not (str(part)==str(identity) or (part is None and str(match.get('featureId'))==str(source_id))):continue
        reading=p.get('reading') or {}
        if reading.get('confidence') not in ('medium','high'):continue
        # Legacy parapets can also be ornamental. This external eligibility
        # gate requires the reading to explicitly identify a gable; runtime
        # adoption should use a typed source semantic field instead of notes.
        if not re.search(r'\bgable(?:d|s)?\b',str(reading.get('notes','')),re.I):continue
        for i,parapet in enumerate(p.get('parapets',[])):
            if parapet.get('bandM'):continue
            reason=None
            try:
                a=np.asarray(parapet['top'],float);frame=np.asarray(p['frameU'],float);h=float(p['heightRefM']);wall=float(reading['wallTopM']);n=np.asarray(p['normal'],float);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
                if a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):reason='not-monotone-explicit-source-outline'
                elif not np.isfinite([*frame,h,wall,*n,p['planeV'],z0]).all() or h<=0 or frame[1]<=frame[0]:reason='invalid-source-frame'
                else:
                    trace=np.c_[frame[0]+a[:,0]*(frame[1]-frame[0]),z0+a[:,1]*h]
                    if trace[0,0]<frame[0]-.02 or trace[-1,0]>frame[1]+.02:reason='outline-outside-source-frame'
                    elif np.max(trace[1:-1,1])<=max(trace[0,1],trace[-1,1])+.5:reason='no-distinct-source-front-gable'
                    elif max(abs(trace[[0,-1],1]-z0-wall))>.75:reason='source-outline-eaves-disagree-with-wall-top'
                    else:priors.append({'profileId':p['id'],'sourceOutlineIndex':i,'normal':n,'tangent':t,'planeV':float(p['planeV']),'traceUZ':trace,'confidence':reading['confidence'],'sourceIds':p.get('sourceIds',[]),'sourceReading':p.get('readingProvenance'), 'uncertaintyM':None,'uncertaintyReason':'legacy source annotation has no pixel uncertainty; hypothesis remains provisional, never a hard measured anchor','semanticGate':'explicit-gable-reading-and-multivertex-parapet-outline'})
            except (ValueError,TypeError,KeyError,IndexError):reason='malformed-source-outline'
            if reason:rejected.append({'profileId':p['id'],'outlineIndex':i,'reason':reason})
    return priors,rejected


def solve(xy,faces,z,known,unknown,priors,footprint,protected_domains=(),cell_m=.5):
    xy=np.asarray(xy,float);faces=np.asarray(faces,int);z=np.asarray(z,float);known=np.asarray(known,bool);unknown=np.asarray(unknown,bool)
    used=np.zeros(len(xy),bool);used[np.unique(faces)]=True
    boundary=np.array([footprint.boundary.distance(Point(p))<1e-7 for p in xy])&used;protected=np.zeros(len(xy),bool)
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
        for j in accepted:photo[j]=True;levels[j]=np.interp(u[j],trace[:,0],trace[:,1])
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
