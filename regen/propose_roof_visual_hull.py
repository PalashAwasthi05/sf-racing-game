"""External, conditional visual-hull ceiling for unsupported native roof fill.

Camera error bounds are explicit scenario assumptions, not validated confidence
intervals. No production data or generation rule is changed.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import math
import cv2
import numpy as np
import trimesh
from shapely.geometry import Point,Polygon
from scipy.ndimage import map_coordinates
from regen.check_roof_sky_projection import sha,raster,prism


def pixel(point, projection, camera_offset=None):
    camera=np.asarray(projection['cameraXYZ'])+(0 if camera_offset is None else camera_offset)
    q=(np.asarray(point)-camera)@np.asarray(projection['basisRightDownForward']).T
    if q[2]<=.05:return None,None
    return q[:2]/q[2]*projection['focalPx']+np.asarray(projection['sizePx'])/2-.5,float(q[2])


def uncertainty_px(point, projection, xy_margin, z_margin, angle_deg):
    p,z=pixel(point,projection)
    if p is None:return math.inf
    shifts=[]
    for offset in itertools.product((-xy_margin,xy_margin),(-xy_margin,xy_margin),(-z_margin,z_margin)):
        q,_=pixel(point,projection,np.asarray(offset))
        if q is None:return math.inf
        shifts.append(np.linalg.norm(q-p))
    return float(max(shifts)+projection['focalPx']*math.tan(math.radians(angle_deg)))


def sample(field, xy, default=0.):
    if xy is None or xy[0]<0 or xy[1]<0 or xy[0]>field.shape[1]-1 or xy[1]>field.shape[0]-1:return default
    return float(map_coordinates(field,np.array([[xy[1]],[xy[0]]]),order=1,mode='constant',cval=default)[0])


def constraint_field(groups, projection):
    h,w=groups['sky'].shape
    sky=(groups['sky']>=.95)&(groups['occluder']<=.05)&(groups['structure']<=.05)
    _,labels=cv2.connectedComponents(sky.astype(np.uint8));border=set(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])-{0}
    sky&=np.isin(labels,list(border));clearance=cv2.distanceTransform(sky.astype(np.uint8),cv2.DIST_L2,5)
    corners=np.asarray(projection['facadeCornersPx']);xs=np.arange(w)
    def edge(indices):
        p=corners[indices];p=p[np.argsort(p[:,0])];return np.interp(xs,p[:,0],p[:,1])
    top,bottom=edge([1,3]),edge([0,2]);solid=(groups['structure']>=.7)&(groups['occluder']<=.1)&(np.arange(h)[:,None] <= (top+.35*np.maximum(0,bottom-top))[None,:])
    eligible=np.maximum.accumulate(solid[::-1],axis=0)[::-1]
    x0,x1=sorted(corners[[1,3],0]);eligible&=((xs>=x0)&(xs<=x1))[None,:]
    return clearance,eligible.astype(np.float32)


def conflict(point, view, xy_margin, z_margin, base_margin=5.):
    p,depth=pixel(point,view['projection'])
    if p is None or sample(view['eligible'],p)<.999:return False,None
    amount=sample(view['clearance'],p)
    allowance=base_margin+uncertainty_px(point,view['projection'],xy_margin,z_margin,view['angleMarginDeg'])
    if amount<=allowance:return False,{'pixelPx':p.tolist(),'skyClearancePx':amount,'allowancePx':allowance}
    return True,{'pixelPx':p.tolist(),'skyClearancePx':amount,'allowancePx':allowance,'depthM':depth}


def ceiling(point, views, xy_margin, z_margin, max_lower=1.):
    votes=[];checks=[]
    for view in views:
        bad,proof=conflict(point,view,xy_margin,z_margin)
        checks.append({'image':view['image'],'viewIndex':view['index'],'robustContradiction':bad,'proof':proof})
        if bad:votes.append((view,proof))
    images={v['image'] for v,_ in votes}
    if len(images)<2:return None,{'status':'fewer-than-two-independent-margin-robust-contradictions','independentImages':len(images),'sourceChecks':checks}
    required=[]
    for view,proof in votes:
        lower=point.copy();lower[2]-=max_lower
        bad,_=conflict(lower,view,xy_margin,z_margin)
        if bad:return None,{'status':'required-lowering-exceeds-bounded-scenario','sourceImage':view['image'],'maximumChangeM':max_lower}
        lo,hi=0.,max_lower
        for _ in range(24):
            mid=(lo+hi)/2;q=point.copy();q[2]-=mid
            if conflict(q,view,xy_margin,z_margin)[0]:lo=mid
            else:hi=mid
        required.append(hi)
    delta=max(required)+.002
    if delta>max_lower:return None,{'status':'required-lowering-exceeds-bounded-scenario'}
    proposed=point.copy();proposed[2]-=delta
    # Every accepted view must leave confident sky; an ROI exit is not proof.
    proofs=[]
    for view,before in votes:
        q,_=pixel(proposed,view['projection'])
        if sample(view['eligible'],q)<.999:return None,{'status':'lowering-exits-supported-source-column'}
        bad,after=conflict(proposed,view,xy_margin,z_margin)
        if bad:return None,{'status':'source-ceiling-not-satisfied'}
        proofs.append({'image':view['image'],'sourceSha256':view['sourceSha256'],'viewIndex':view['index'],'before':before,'after':after})
    for view in views:
        if conflict(proposed,view,xy_margin,z_margin)[0]:
            return None,{'status':'proposal-contradicts-another-accepted-source','sourceImage':view['image']}
    return proposed,{'status':'conditional-ceiling-proposal','loweringM':delta,'independentImages':len(images),'sources':proofs}


def protected_vertices(mesh,audit,observations,profiles,mass,support_domain=None):
    """Conservative superset of every planar anchor and source-owned domain."""
    shell=audit['roofShell'];struct=shell.get('structuralPlaneSurface',{});planes=list(struct.get('planes',[]))
    host=shell.get('observedHostForm',{})
    if host.get('status','').startswith('applied-'):planes+=host.get('model',{}).get('planes',[])
    if not planes or struct.get('method')!='harmonic-graph-interpolation-with-Dirichlet-plane-and-eave-anchors':return np.ones(len(mesh.vertices),bool),{'reason':'no-verifiable-harmonic-fill-domain'}
    xyz=np.asarray(mesh.vertices);design=np.c_[xyz[:,:2],np.ones(len(xyz))]
    protected=xyz[:,2]<=float(mass['baseRoofZ'])+.05
    planar=np.zeros(len(xyz),bool)
    for plane in planes:planar|=abs(design@plane-xyz[:,2])<=.05
    if support_domain is not None:planar&=np.array([support_domain.buffer(.002).covers(Point(p)) for p in xyz[:,:2]])
    protected|=planar
    source_domains=[]
    for obs in observations:
        joined=[p for p in profiles if p.get('side')==obs.get('side') and p.get('building',p.get('buildingId'))==mass['id']]
        # Existing profiles often carry sourceId instead of building.
        if not joined:joined=[p for p in profiles if p.get('sourceSide')==obs.get('side') or p.get('id')=='regen-'+str(obs.get('side'))]
        for profile in joined:
            n=np.asarray(profile['normal']);n=n/np.linalg.norm(n);t=np.array([-n[1],n[0]])
            u0,u1=profile['frameU'];front=n*profile['planeV']
            for feature in obs.get('features',[]):
                if feature.get('confidence') not in ('medium','high'):continue
                if feature.get('x0M') is None or feature.get('x1M') is None:continue
                a=front+t*(u0+feature['x0M']);b=front+t*(u0+feature['x1M']);source_domains.append(Polygon([a,b,b-n*6,a-n*6]).buffer(.10))
            if host.get('status','').startswith('applied-') and obs.get('frontageRoofline'):
                a=front+t*u0;b=front+t*u1;depth=float(host.get('zoneDepthM',6));source_domains.append(Polygon([a,b,b-n*depth,a-n*depth]).buffer(.10))
            eave=obs.get('eaveM')
            if eave is not None:
                from shapely.geometry import LineString
                line=LineString([front+t*u0,front+t*u1]);protected|=np.array([line.distance(Point(p))<.30 for p in xyz[:,:2]])&(abs(xyz[:,2]-(mass['z0']+eave))<.10)
    for part in shell.get('boundedSourceRoofParts',[]):
        vertices=part.get('verticesXYZ')
        if vertices and len(vertices)>=3:source_domains.append(Polygon(np.asarray(vertices)[:,:2]).buffer(.10))
    for geometry in source_domains:protected|=np.array([geometry.covers(Point(p)) for p in xyz[:,:2]])
    return protected,{'planarAnchorToleranceM':.05,'sourceDomainCount':len(source_domains),'protectedCount':int(protected.sum()),'planarAnchorCount':int(planar.sum()),
        'qualification':'Plane proximity within independently reproduced native Dirichlet domains; source features/eaves additionally protected. Unmatched/non-harmonic domains abstain.' if support_domain is not None else 'Conservative plane proximity without a reproduced support domain.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('projection-audit','scene','massing','roof-audit','observations','profiles','registration','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--xy-margin',type=float,default=.5);ap.add_argument('--z-margin',type=float,default=.25);ap.add_argument('--angular-margin',type=float,default=.5);ap.add_argument('--max-lower',type=float,default=1.)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=json.loads(a.projection_audit.read_text());scene=trimesh.load(a.scene,force='scene');masses=json.loads(a.massing.read_text());audits={r['buildingId']:r for r in json.loads(a.roof_audit.read_text())};obs=json.loads(a.observations.read_text());obs=obs.get('observations',obs);profiles=json.loads(a.profiles.read_text())['profiles'];registration=json.loads(a.registration.read_text())
    roofs={m['id']:trimesh.util.concatenate([g for key,g in scene.geometry.items() if key.startswith('roofs__'+m['id']+'___')]) for m in masses}
    proxies={m['id']:prism(m,float(roofs[m['id']].bounds[1,2])) for m in masses}
    from regen.rebuild_roof_support_review import contexts
    from regen.reproduce_roof_support_domains import reproduce
    area=a.massing.parent.parent
    clouds=contexts(area,masses,np.load(a.massing.parent/'rooftop-structure-returns.npz'))
    results=[];combined_before=[];combined_after=[];support_records={}
    for mass in masses:
        building=mass['id'];mesh=roofs[building];records=[r for r in report['records'] if r['building']==building and r.get('metrics',{}).get('conflictPixels',0)]
        all_votes={}
        for record in records:
            for face in record['metrics']['implicatedFacets']:
                for vertex in mesh.faces[face['faceIndex']]:all_votes.setdefault(int(vertex),set()).add(record['image'])
        # Exporters split coincident vertices at material/normal seams. Keep
        # every coincident copy together so an external proposal cannot crack
        # the native roof simply by moving one triangle's copy.
        _,inverse=np.unique(np.round(mesh.vertices,6),axis=0,return_inverse=True)
        coordinate_groups={int(g):np.flatnonzero(inverse==g) for g in np.unique(inverse)}
        group_votes={}
        for vertex,images in all_votes.items():group_votes.setdefault(int(inverse[vertex]),set()).update(images)
        candidates={g for g,images in group_votes.items() if len(images)>=2}
        domain=None
        if candidates:
            domain,support_record=reproduce(mass,clouds[building],obs,profiles,audits[building]);support_records[building]=support_record
            (a.output/'support-domain-audit.json').write_text(json.dumps(support_records,indent=2)+'\n')
        protected,protection=protected_vertices(mesh,audits[building],obs,profiles,mass,domain)
        if candidates and domain is None:protected[:]=True;protection['status']='unresolved-exported-plane-domain-abstention'
        row=next(r['reviewRow'] for r in report['records'] if r['building']==building);views=[]
        for record in records:
            projection={**record['projection'],'sizePx':list(Image_size(record['sourcePath']))}
            with np.load(record['semanticCachePath']) as archive:groups={k:archive[k].astype(np.float32) for k in ('sky','structure','occluder')}
            clearance,eligible=constraint_field(groups,projection);reg=registration.get(record['image'],{})
            angle=max(a.angular_margin,float(reg.get('costDeg',0)))
            # Confirm source-pixel visibility against exact roof depth and
            # conservative known-neighbor prisms for initial vertex votes.
            depth,_=raster(mesh,projection,projection['sizePx']);other=np.full_like(depth,np.inf)
            for identity,proxy in proxies.items():
                if identity!=building:
                    dd,_=raster(proxy,projection,projection['sizePx']);np.minimum(other,dd,out=other)
            eligible*=(np.isfinite(depth)&(depth<=other+.02)).astype(np.float32)
            views.append({**record,'projection':projection,'clearance':clearance,'eligible':eligible,'angleMarginDeg':angle,
                'registrationResidualDeg':reg.get('costDeg'),'registrationAccepted':reg.get('accepted',False)})
        new=np.asarray(mesh.vertices).copy();proposals=[];rejected=[]
        for group in sorted(candidates):
            members=coordinate_groups[group];index=int(members[0])
            proposed,evidence=ceiling(new[index],views,a.xy_margin,a.z_margin,a.max_lower)
            if protected[members].any():
                rejected.append({'vertexIndex':index,'beforeXYZ':mesh.vertices[index].tolist(),'status':'protected-measured-plane-or-source-domain','counterfactualMarginAudit':evidence});continue
            if proposed is not None:
                if proposed[2]<=mass['baseRoofZ']+.05:rejected.append({'vertexIndex':index,'status':'would-cross-existing-roof-base'});continue
                new[members]=proposed;proposals.append({'vertexIndex':index,'coincidentVertexIndices':members.tolist(),'beforeXYZ':mesh.vertices[index].tolist(),'afterXYZ':proposed.tolist(),**evidence})
            else:rejected.append({'vertexIndex':index,**evidence})
        # No new abrupt local jump: retain the unchanged topology and reject
        # edits that increase any incident vertical difference by >5cm.
        edges=mesh.edges_unique;changed={i for p in proposals for i in p['coincidentVertexIndices']};bad=set()
        for _ in range(4):
            old_delta=abs(mesh.vertices[edges[:,0],2]-mesh.vertices[edges[:,1],2]);new_delta=abs(new[edges[:,0],2]-new[edges[:,1],2]);invalid=edges[new_delta>old_delta+.05]
            invalid_changes={int(i) for edge in invalid for i in edge if int(i) in changed-bad}
            if not invalid_changes:break
            invalid_changes={int(i) for vertex in invalid_changes for i in coordinate_groups[int(inverse[vertex])]}
            bad|=invalid_changes
            for i in invalid_changes:new[i]=mesh.vertices[i]
        for proposal in proposals:
            if proposal['vertexIndex'] in bad:rejected.append({'vertexIndex':proposal['vertexIndex'],'status':'would-introduce-new-local-height-jump'})
        proposals=[p for p in proposals if p['vertexIndex'] not in bad]
        candidate=trimesh.Trimesh(vertices=new,faces=mesh.faces.copy(),process=False)
        mesh.export(a.output/f'{row:02d}-before-roof.ply');candidate.export(a.output/f'{row:02d}-after-roof.ply');trimesh.Scene(mesh).export(a.output/f'{row:02d}-before-roof.glb');trimesh.Scene(candidate).export(a.output/f'{row:02d}-after-roof.glb');combined_before.append(mesh);combined_after.append(candidate)
        results.append({'building':building,'reviewRow':row,'implicatedVertexGroups':len(candidates),'protection':protection,'proposals':proposals,'rejected':rejected,'changedVertexGroups':len(proposals),'changedVertices':sum(len(p['coincidentVertexIndices']) for p in proposals),
            'sourceMarginRecords':[{'image':v['image'],'viewIndex':v['index'],'angleMarginDeg':v['angleMarginDeg'],'registrationResidualDeg':v['registrationResidualDeg'],'registrationAccepted':v['registrationAccepted']} for v in views],
            'maximumLoweringM':max((p['loweringM'] for p in proposals),default=0.),'topologyUnchanged':True,'watertightBefore':bool(mesh.is_watertight),'watertightAfter':bool(candidate.is_watertight)})
        print(json.dumps({'row':row,'implicated':len(candidates),'changed':len(proposals),'protected':int(sum(protected[coordinate_groups[g]].any() for g in candidates))}),flush=True)
        (a.output/'proposals.json').write_text(json.dumps({'status':'conditional-external-scenario-not-production','algorithmSha256':sha(__file__),'inputs':{name:{'path':str(getattr(a,name)),'sha256':sha(getattr(a,name))} for name in ('projection_audit','scene','massing','roof_audit','observations','profiles','registration')},
            'cameraScenario':{'xyPerAxisMarginM':a.xy_margin,'zMarginM':a.z_margin,'angularMarginDeg':a.angular_margin,'additionalAnglePolicy':'max(scenario angular margin, skyline registration costDeg)','validatedCameraConfidenceBound':False},
            'maximumLoweringM':a.max_lower,'minimumIndependentImages':2,'safeSkyMarginPx':5,'rows':results,'activeGeometryChanges':0},indent=2)+'\n')
    trimesh.util.concatenate(combined_before).export(a.output/'all22-before-roofs.ply');trimesh.util.concatenate(combined_after).export(a.output/'all22-after-roofs.ply')


def Image_size(path):
    from PIL import Image
    with Image.open(path) as image:return image.size


if __name__=='__main__':main()
