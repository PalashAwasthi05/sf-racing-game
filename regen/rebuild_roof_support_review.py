"""Rebuild a review cohort's roof batches before/after measured support fixes.

Read-only inputs; writes isolated PLY/GLB roof batches and audit records, with
the same existing host/feature adapters. Does not generate facades or a scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import trimesh
from scipy.signal import find_peaks
from shapely import contains_xy
from shapely.geometry import Polygon

from pipeline import roof_shell, roof_assemblies, roof_source_parts, roof_observations, spatial_rules


def contexts(area, masses, existing):
    result={m['id']:np.asarray(existing[m['id']+'__context']) for m in masses if m['id']+'__context' in existing}
    missing=[m for m in masses if m['id'] not in result]
    if not missing:return result
    from pyproj import Transformer
    world=json.loads((area/'world.json').read_text())
    ll=Transformer.from_crs(4326,32610,always_xy=True);ox,oy=ll.transform(world['center']['lon'],world['center']['lat'])
    to_merc=Transformer.from_crs(32610,3857,always_xy=True);to_utm=Transformer.from_crs(3857,32610,always_xy=True)
    original=np.load(area/'data/points.npz')['points']
    for m in missing:
        g=Polygon(m['outline'][0]);a,b,c,d=g.bounds
        xx,yy=to_merc.transform([a+ox-10,c+ox+10],[b+oy-10,d+oy+10])
        keep=(original[:,0]>=min(xx))&(original[:,0]<=max(xx))&(original[:,1]>=min(yy))&(original[:,1]<=max(yy))&(original[:,3]!=7)&(original[:,3]!=18)
        q=original[keep];x,y=to_utm.transform(q[:,0],q[:,1]);result[m['id']]=np.c_[x-ox,y-oy,q[:,2]-float(world['verticalOrigin'])]
    return result


def parameters(observations, profiles, identity, z0):
    constraints=[];hosts=[];exclusions=[];frontage_sides=[]
    for observation in observations:
        if observation.get('confidence') not in ('medium','high'):continue
        joined=roof_assemblies._profiles_for_side(profiles,observation.get('side'),identity,identity)
        if not joined:continue
        edge=(observation.get('eaveEvidence') or {}).get('visibleRoofEdgeHeightRangeM')
        eave=observation.get('eaveM');level=eave if eave is not None else (float(np.mean(edge)) if edge else None)
        if level is None:continue
        if observation.get('hostRoofKind') in ('hip','gable','mansard'):
            hosts.append({'kind':observation['hostRoofKind'],'eaveZ':z0+level,'ridgeZ':z0+observation['hostRidgeM'] if observation.get('hostRidgeM') is not None else None,
                'confidence':observation['confidence'],'side':observation['side'],'anchorRole':'eave' if eave is not None else 'visible-roof-edge',
                'frontageKind':observation.get('frontageKind'),'anchorUncertaintyM':float(observation.get('anchorUncertaintyM') or 0)})
        for profile in joined:
            n=np.asarray(profile['normal'],float);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
            a=n*profile['planeV']+t*profile['frameU'][0];b=n*profile['planeV']+t*profile['frameU'][1]
            points,_=roof_observations.measured_frontage(observation.get('frontageRoofline') or {})
            if isinstance(points,list) and len(points)>=2:
                p=np.asarray(points);world=n[None,:]*profile['planeV']+t[None,:]*(profile['frameU'][0]+p[:,0,None]);frontage_sides.append(observation['side'])
                for i in range(len(p)-1):constraints.append({'a':world[i].tolist(),'b':world[i+1].tolist(),'z':z0+p[i,1],'z1':z0+p[i+1,1],'role':'gable-frontage-roofline'})
            else:constraints.append({'a':a.tolist(),'b':b.tolist(),'z':z0+level,'role':'eave' if eave is not None else 'visible-roof-edge','uncertaintyM':float(observation.get('anchorUncertaintyM') or 0)})
            for feature in observation.get('features',[]):
                if feature.get('confidence') in ('medium','high'):
                    exclusions.append({'normal':n.tolist(),'planeV':profile['planeV'],'u0':profile['frameU'][0]+feature['x0M'],'u1':profile['frameU'][0]+feature['x1M'],'maxDepthM':6})
    hosts=[h for h in hosts if h['side'] not in frontage_sides]
    host=max(hosts,key=lambda h:(h['confidence']=='high',h['ridgeZ'] is not None)) if hosts and len({h['kind'] for h in hosts})==1 else None
    if host:host={**host,'excludeStrips':exclusions}
    return constraints,host


def metrics(mesh,footprint):
    tri=np.asarray(mesh.triangles);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1);pitch=np.degrees(np.arccos(np.clip(abs(cross[:,2])/np.maximum(length,1e-12),0,1)))
    peaks=0;prominence=0.;maximum_boundary_delta=0.
    for ring in [footprint.exterior,*footprint.interiors]:
        coords=np.asarray(ring.coords)
        for a,b in zip(coords[:-1],coords[1:]):
            direction=b-a;span=np.linalg.norm(direction)
            if span<1e-8:continue
            relative=mesh.vertices[:,:2]-a;along=relative@direction/span;distance=abs(direction[0]*relative[:,1]-direction[1]*relative[:,0])/span;mask=(distance<.002)&(along>=-.001)&(along<=span+.001);samples={}
            for t,z in zip(along[mask],mesh.vertices[mask,2]):samples[round(float(t),4)]=max(samples.get(round(float(t),4),-1e9),float(z))
            z=np.array([samples[k] for k in sorted(samples)])
            if len(z)>2:
                p,info=find_peaks(z,prominence=.1);peaks+=len(p);prominence=max(prominence,max(info['prominences'],default=0));maximum_boundary_delta=max(maximum_boundary_delta,float(abs(np.diff(z)).max()))
    return {'roofFaces':len(mesh.faces),'roofAreaM2':float(mesh.area),'steepFacesOver75Deg':int(np.sum(pitch>75)),
        'boundaryPeaksOver10cm':peaks,'maximumBoundaryPeakProminenceM':float(prominence),'maximumAdjacentBoundaryDeltaM':maximum_boundary_delta}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--area',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    masses=json.loads((args.build/'massing-dump.json').read_text());prior=json.loads((args.build/'roof-envelope-audit.json').read_text());observations=json.loads(args.observations.read_text());observations=observations.get('observations',observations) if isinstance(observations,dict) else observations;profiles=json.loads(args.profiles.read_text())['profiles'];jobs=json.loads(args.jobs.read_text());rows={j['building']:i for i,j in enumerate(jobs)}
    saved=np.load(args.build/'rooftop-structure-returns.npz');clouds=contexts(args.area,masses,saved);args.output.mkdir(parents=True,exist_ok=True)
    # Before mode reverses only the two support admission changes. All other
    # current host, source-part, topology, and feature behavior remains equal.
    before=types.ModuleType('pipeline._roof_shell_before_support_fix');before.__package__='pipeline';exec(compile(Path(roof_shell.__file__).read_text(),str(roof_shell.__file__),'exec'),before.__dict__)
    before._raw_points_for_kept_cells=lambda raw,*args:np.asarray(raw)
    before._retain_owned_components=lambda mask,*args,**kwargs:(np.asarray(mask,bool),[])
    report=[];combined={'before':[],'after':[]}
    footprints=[Polygon(m['outline'][0]) for m in masses]
    for m,footprint in zip(masses,footprints):
        identity=m['id'];base=next(a for a in prior if a['buildingId']==identity)['baseRoofZ'];raw=clouds[identity];raw=raw[contains_xy(footprint,raw[:,0],raw[:,1])&(raw[:,2]>m['z0']+3)];a,b,c,d=footprint.bounds;nx=max(2,int(np.ceil((c-a)/1.5)));ny=max(2,int(np.ceil((d-b)/1.5)));inner=raw[contains_xy(footprint.buffer(-.65),raw[:,0],raw[:,1])];_,keep=spatial_rules.surface_raster(inner,a,d,1.5,nx,ny);inner=inner[keep];_,keep=spatial_rules.surface_raster(raw,a,d,1.5,nx,ny);boundary=raw[keep];constraints,host=parameters(observations,profiles,identity,m['z0']);record={'building':identity,'reviewRow':rows.get(identity)}
        for mode,module in [('before',before),('after',roof_shell)]:
            shell=module.build(footprint,inner,base,boundary_points=boundary,eave_constraints=constraints,host_spec=host)
            parts,partaudit=roof_source_parts.resolve(observations,profiles,identity,identity,footprint);shell=roof_source_parts.compose(shell,parts,[g for mm,g in zip(masses,footprints) if mm['id']!=identity])
            features,featureaudit=roof_assemblies.resolve(observations,profiles,identity,identity,footprint,m['z0'],shell['surface']);cutters,_=module.component_cutters(features,m['z0']);surface=module.clip_component_intersections(shell['surface'],cutters);skirt=module.clip_component_intersections(shell['skirt'],cutters)
            surface.export(args.output/f'{rows.get(identity,0):02d}-{mode}-roof.ply')
            if skirt is not None:skirt.export(args.output/f'{rows.get(identity,0):02d}-{mode}-skirt.ply')
            scene=trimesh.Scene();scene.add_geometry(surface,node_name='roof');
            if skirt is not None:scene.add_geometry(skirt,node_name='skirt')
            scene.export(args.output/f'{rows.get(identity,0):02d}-{mode}-roof.glb')
            audit=shell['audit'];structural=audit.get('structuralPlaneSurface',{});accepted=[f for f in structural.get('fit',{}).get('patches',[]) if 'heldOutP95M' in f and 'rejected' not in f]
            record[mode]={'metrics':metrics(surface,footprint),'structuralRejectedRawPoints':audit['structuralInputPointsRejectedByMeasuredCellFilter'],
                'heldOutP95M':[r['heldOutP95M'] for r in accepted],'ownership':structural.get('supportOwnership',[]),
                'planes':structural.get('planes',[]),'features':featureaudit,'sourceParts':partaudit,
                'host':audit.get('observedHostForm',{}),'unknownFillUnchanged':True}
            combined[mode].append(surface)
        report.append(record)
        print(json.dumps({'row':record['reviewRow'],'beforePeaks':record['before']['metrics']['boundaryPeaksOver10cm'],'afterPeaks':record['after']['metrics']['boundaryPeaksOver10cm'],'rejectedRaw':record['after']['structuralRejectedRawPoints'],'sourceFeatures':[len(record[k]['features']['accepted']) for k in ['before','after']]}),flush=True)
        (args.output/'audit.json').write_text(json.dumps({'cases':report,'roofShellSha256':hashlib.sha256(Path(roof_shell.__file__).read_bytes()).hexdigest(),
            'qualification':'All cohort native roof/host/source-part/component paths replayed, no facade/full-scene generation. Exported massing footprint is rounded1mm. Source building identity/frame matched through existing adapters; all current fixed source features retained by those adapters. Before mode bypasses only the measured raw-cell filter and post-reassignment tiny-component gate.'},indent=2)+'\n')
    for mode,meshes in combined.items():trimesh.util.concatenate(meshes).export(args.output/f'all22-{mode}-roofs.ply')


if __name__=='__main__':main()
