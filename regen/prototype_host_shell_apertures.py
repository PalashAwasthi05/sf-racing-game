"""Build an isolated GLB prototype for measured openings covered by host shells.

This adapter never edits compiler inputs or production geometry code. It uses
the side-keyed source review jobs to find only supported opening/shell conflicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from regen.host_shell_apertures import (
    GLB_TO_SOURCE,
    SOURCE_TO_GLB,
    TriangleIndex,
    clip_shell_mesh,
    opening_center_source,
    opening_host_conflicts,
    opening_outline_uv,
    opening_slab,
    reveal_mesh,
    build_measured_receiver,
    _project,
)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def _source_vertices(mesh, transform):
    source_from_local=GLB_TO_SOURCE@np.asarray(transform,dtype=float)
    vertices=np.asarray(mesh.vertices,dtype=float)@source_from_local[:3,:3].T+source_from_local[:3,3]
    return vertices,source_from_local


def _source_mesh(mesh, transform):
    out=mesh.copy()
    vertices,source_from_local=_source_vertices(mesh,transform)
    out.vertices=vertices
    return out,source_from_local


def _local_mesh(mesh_source, source_from_local):
    out=mesh_source.copy()
    local_from_source=np.linalg.inv(source_from_local)
    out.vertices=np.asarray(mesh_source.vertices)@local_from_source[:3,:3].T+local_from_source[:3,3]
    return out


def _feature_zbase(scene, feature_id):
    zs=[]
    for node in scene.graph.nodes_geometry:
        if (node.startswith(f'buildings__{feature_id}___')
                and '__observed_' not in node):
            transform,geometry_name=scene.graph.get(node);mesh=scene.geometry[geometry_name]
            source,_=_source_vertices(mesh,transform)
            if len(source):zs.extend(source[:,2].tolist())
    return float(min(zs)) if zs else None


def _add_reveal(scene, source_mesh, node_name, geometry_name):
    mesh=source_mesh.copy()
    mesh.vertices=np.asarray(mesh.vertices)@SOURCE_TO_GLB[:3,:3].T
    scene.add_geometry(mesh,node_name=node_name,geom_name=geometry_name,
                       transform=np.eye(4))


def _detail_visibility(scene, profile, opening, job, zbase):
    """Count matched-camera rays to emitted detail triangles inside one opening."""
    from shapely.geometry import Point
    from regen.host_shell_apertures import profile_tangent
    outline=opening_outline_uv(profile,opening,zbase).buffer(.03)
    tangent=profile_tangent(profile);feature_id=profile['match']['featureId']
    index=TriangleIndex.from_scene(scene);counts={'tested':0,'observedDetailFirst':0,
        'baseShellFirst':0,'otherFirst':0,'noHit':0}
    for node in scene.graph.nodes_geometry:
        if not node.startswith(f'facades__{feature_id}__observed'):
            continue
        transform,geometry_name=scene.graph.get(node);mesh=scene.geometry[geometry_name]
        world=mesh.vertices@transform[:3,:3].T+transform[:3,3]
        source=world@GLB_TO_SOURCE[:3,:3].T
        for face in mesh.faces:
            center=source[face].mean(axis=0);uv=(float(center[:2]@tangent),float(center[2]))
            if not outline.covers(Point(uv)):
                continue
            pix=_project(center,job)
            if pix[2]<=0 or not (0<=pix[0]<job['width'] and 0<=pix[1]<job['height']):
                continue
            counts['tested']+=1;hits=index.ray_hits(job['from'],center)
            if not hits:counts['noHit']+=1
            elif hits[0]['node'].startswith(f'facades__{feature_id}__observed'):
                counts['observedDetailFirst']+=1
            elif hits[0]['node'].startswith(f'buildings__{feature_id}___') and '__observed_' not in hits[0]['node']:
                counts['baseShellFirst']+=1
            else:counts['otherFirst']+=1
    return counts


def _receiver_node_expected(node, profile, opening, opening_index):
    feature_id=profile['match']['featureId']
    detail_prefix=f'facades__{feature_id}__observed_{profile["id"]}'
    generated_prefix=f'facades__{feature_id}__aperture_receiver_{opening_index}_'
    staged_prefix=f'{detail_prefix}_aperture_{opening_index}_part_'
    if node.startswith((generated_prefix,staged_prefix)):
        kind=(opening.get('style') or {}).get('kind')
        role=node.partition('___')[0].replace('-','_')
        return (kind in ('window','shopfront','terrace_frame') and
                role.endswith(('_glazing','_meeting_rail','_muntin','_mullion','_transom'))
                or kind=='entry' and ('_door_leaf_' in role or role.endswith(('_door_glass','_door_panel'))))
    if node.startswith(f'{detail_prefix}_aperture_'):
        return False
    if not node.startswith(detail_prefix):return False
    style=opening.get('style') or {}
    if style.get('kind')=='window':
        return node.endswith('___glass')
    if style.get('kind')=='entry':
        material=(style.get('door') or {}).get('material')
        return bool(material and node.endswith('___'+str(material)))
    return False


def _receiver_ray_coverage(scene, profile, opening, job, zbase, opening_index,
                           triangle_index=None):
    """Cast full camera rays through interior points and count expected receivers."""
    from shapely.geometry import Point
    from regen.host_shell_apertures import opening_outline_uv,profile_tangent
    full=opening_outline_uv(profile,opening,zbase)
    outline=full.buffer(-.08)
    if outline.is_empty:outline=full.buffer(-.01)
    min_u,min_z,max_u,max_z=outline.bounds;tangent=profile_tangent(profile)
    normal=np.asarray(profile['normal'],dtype=float);camera=np.asarray(job['from'],dtype=float)
    feature_id=profile['match']['featureId'];index=triangle_index or TriangleIndex.from_scene(scene)
    counts={'sampled':0,'expectedReceiverFirst':0,'baseShellFirst':0,
            'otherGeometryFirst':0,'skyOrNoHit':0}
    expected=f'facades__{feature_id}__aperture_receiver_{opening_index}_'
    for fu in np.linspace(.08,.92,9):
        for fz in np.linspace(.08,.92,9):
            u=min_u+(max_u-min_u)*fu;z=min_z+(max_z-min_z)*fz
            if not outline.covers(Point(u,z)):continue
            xy=tangent*u+normal*float(profile['planeV']);point=np.array([xy[0],xy[1],z])
            delta=point-camera;length=np.linalg.norm(delta)
            if length<1e-8:continue
            hits=index.ray_hits(camera,camera+delta/length*60.)
            counts['sampled']+=1
            if not hits:counts['skyOrNoHit']+=1
            elif _receiver_node_expected(hits[0]['node'],profile,opening,opening_index):
                counts['expectedReceiverFirst']+=1
            elif hits[0]['node'].startswith(f'buildings__{feature_id}___') and '__observed_' not in hits[0]['node']:
                counts['baseShellFirst']+=1
            else:counts['otherGeometryFirst']+=1
    return counts


def _center_expected_receiver_hit(scene, profile, opening, job, zbase, opening_index, triangle_index=None):
    """Return the first expected glazing/leaf hit on the measured camera ray."""
    target=opening_center_source(profile,opening,zbase)
    camera=np.asarray(job['from'],dtype=float);direction=target-camera
    direction/=max(1e-9,np.linalg.norm(direction))
    index=triangle_index or TriangleIndex.from_scene(scene)
    hits=index.ray_hits(camera,camera+direction*60.)
    first=hits[0] if hits else None
    return first if first and _receiver_node_expected(
        first['node'],profile,opening,opening_index) else None


def build_candidate(input_glb, profiles_path, jobs_path, output_glb,
                    report_path):
    profiles=json.loads(Path(profiles_path).read_text())['profiles']
    jobs=json.loads(Path(jobs_path).read_text())
    scene=trimesh.load(input_glb,force='scene')
    original=trimesh.load(input_glb,force='scene')
    profile_by_id={p.get('id'):p for p in profiles}
    review_sides=[]
    for job in jobs:
        side=job.get('side')
        profile=profile_by_id.get('regen-'+str(side))
        if profile is None:
            continue
        feature_id=str(profile.get('match',{}).get('featureId') or '')
        zbase=_feature_zbase(original,feature_id)
        if zbase is None:
            review_sides.append({'sideId':side,'status':'no-base-shell','openingCount':len(profile.get('openings',[]))})
            continue
        conflicts=opening_host_conflicts(original,profile,job,zbase)
        review_sides.append({'sideId':side,'profileId':profile['id'],
                             'openingCount':len(profile.get('openings',[])),
                             'zBaseM':zbase,'supportedConflictCount':len(conflicts),
                             'conflicts':conflicts})

    # Build an audit record from all 22 matched sides before altering anything.
    modifications=[]
    modified_sources={}
    for side_record in review_sides:
        if side_record.get('supportedConflictCount',0)==0:
            continue
        side=side_record['sideId'];profile=profile_by_id[side_record['profileId']]
        zbase=side_record['zBaseM'];job=next(j for j in jobs if j.get('side')==side)
        feature_id=profile['match']['featureId']
        for conflict in side_record['conflicts']:
            node=conflict['firstHit']['node']
            transform,geometry_name=scene.graph.get(node)
            if geometry_name not in modified_sources:
                source_mesh,source_from_local=_source_mesh(scene.geometry[geometry_name],transform)
                modified_sources[geometry_name]=(source_mesh,source_from_local,transform,node)
            source_mesh,source_from_local,transform,node=modified_sources[geometry_name]
            slab=opening_slab(source_mesh,profile,conflict,zbase)
            if slab is None:
                modifications.append({'sideId':side,'openingIndex':conflict['openingIndex'],
                                      'node':node,'status':'rejected-no-local-slab'})
                continue
            clipped,stats=clip_shell_mesh(source_mesh,profile,conflict['outlineUV'],slab,zbase)
            if stats['affectedTriangles']==0 or stats['removedProjectedAreaM2']<=1e-7:
                modifications.append({'sideId':side,'openingIndex':conflict['openingIndex'],
                                      'node':node,'status':'rejected-no-supported-triangle',
                                      'slabV':list(slab),'stats':stats})
                continue
            modified_sources[geometry_name]=(clipped,source_from_local,transform,node)
            reveal=reveal_mesh(profile,conflict['outlineUV'],slab,zbase,
                               material=scene.geometry[geometry_name].visual.material)
            # Reveal nodes live directly in GLB coordinates and retain the
            # same host PBR material; the wall mesh itself remains in its layer.
            reveal_node=f"facades__{feature_id}__aperture_reveal_{conflict['openingIndex']}"
            reveal_geometry=reveal_node
            modifications.append({'sideId':side,'openingIndex':conflict['openingIndex'],
                                  'node':node,'revealNode':reveal_node,'status':'clipped',
                                  'slabV':[round(float(slab[0]),5),round(float(slab[1]),5)],
                                  'hostPlaneOffsetM':round(float(conflict['hitPlaneV']-profile['planeV']),5),
                                  'outlineAreaM2':round(float(conflict['outlineUV'].area),5),
                                  'removedProjectedAreaM2':round(float(stats['removedProjectedAreaM2']),5),
                                  'affectedTriangles':int(stats['affectedTriangles']),
                                  '_reveal':reveal})

    # Rebuild a receiver only when the existing measured receiver fails the
    # source-camera check. Some openings already have correctly placed source
    # geometry; cloning those would create a second leaf/pane.
    receiver_records=[]
    for record in modifications:
        if record.get('status')!='clipped':continue
        side=record['sideId'];profile=profile_by_id['regen-'+side]
        job=next(j for j in jobs if j.get('side')==side)
        zbase=next(r['zBaseM'] for r in review_sides if r.get('sideId')==side)
        opening=profile['openings'][record['openingIndex']]
        existing_center_hit=_center_expected_receiver_hit(
            original,profile,opening,job,zbase,record['openingIndex'])
        existing_coverage=_receiver_ray_coverage(
            original,profile,opening,job,zbase,record['openingIndex'])
        record['existingExpectedCenterHit']=None if existing_center_hit is None else {
            'node':existing_center_hit['node'],
            'distanceM':round(float(existing_center_hit['distanceM']),5)}
        record['existingReceiverCoverage']=existing_coverage
        sampled=max(1,existing_coverage['sampled'])
        existing_ratio=existing_coverage['expectedReceiverFirst']/sampled
        # Center-ray correspondence establishes the correct physical receiver;
        # interior support prevents a tiny trim/mullion intersection from
        # suppressing reconstruction of an otherwise missing pane/leaf.
        if existing_ratio>=.30:
            record['receiverAction']='preserved-existing-source-receiver'
            record['reassembledReceiverNodes']=[]
            record['reassembledReceiverPartCount']=0
            record['receiverRayCoverageBefore']=existing_coverage
            continue
        parts=build_measured_receiver(profile,opening,zbase,original,
                                      record['openingIndex'])
        fid=profile['match']['featureId'];names=[]
        for part_index,part in enumerate(parts):
            node=(f"facades__{fid}__aperture_receiver_{record['openingIndex']}_"
                  f"{part_index}_{part['role'].replace('-','_')}")
            names.append(node);receiver_records.append((node,part['mesh']))
        record['reassembledReceiverNodes']=names
        record['reassembledReceiverPartCount']=len(parts)
        record['receiverAction']='reassembled-from-measured-outline-and-style'
        record['receiverRayCoverageBefore']=existing_coverage

    # Replace each unique structural geometry once, then append reveals and receivers.
    reveal_records=[]
    for geom_name,(source_mesh,source_from_local,transform,node) in modified_sources.items():
        scene.geometry[geom_name]=_local_mesh(source_mesh,source_from_local)
    for record in modifications:
        reveal=record.pop('_reveal',None)
        if reveal is not None:
            reveal_records.append((record['revealNode'],reveal))
    for node,reveal in reveal_records:
        _add_reveal(scene,reveal,node,node)
    for node,receiver in receiver_records:
        _add_reveal(scene,receiver,node,node)

    # Recheck every changed aperture along its actual matched source camera.
    after_index=TriangleIndex.from_scene(scene)
    for record in modifications:
        if record.get('status')!='clipped':
            continue
        side=record['sideId'];profile=profile_by_id['regen-'+side]
        job=next(j for j in jobs if j.get('side')==side)
        zbase=next(r['zBaseM'] for r in review_sides if r.get('sideId')==side)
        opening=profile['openings'][record['openingIndex']]
        target=opening_center_source(profile,opening,zbase)
        camera=np.asarray(job['from'],dtype=float);direction=target-camera
        direction/=max(1e-9,np.linalg.norm(direction))
        # Continue through the measured opening plane so a recessed pane/leaf
        # must actually receive the source-camera ray before unrelated geometry.
        hits=after_index.ray_hits(camera,camera+direction*60.)
        fid=profile['match']['featureId']
        blocked=bool(hits and hits[0]['node'].startswith(f'buildings__{fid}___')
                     and '__observed_' not in hits[0]['node'])
        record['centerRayAfterFirstHit']=None if not hits else {'distanceM':round(hits[0]['distanceM'],5),'node':hits[0]['node']}
        record['centerRayStillBlockedByBaseShell']=blocked
        record['centerRayReachesExpectedReceiver']=bool(hits and
            _receiver_node_expected(hits[0]['node'],profile,opening,
                                    record['openingIndex']))
        opening=profile['openings'][record['openingIndex']]
        record['openingDetailRaysBefore']=_detail_visibility(
            original,profile,opening,job,zbase)
        record['openingDetailRaysAfter']=_detail_visibility(
            scene,profile,opening,job,zbase)
        record['receiverRayCoverageAfter']=_receiver_ray_coverage(
            scene,profile,opening,job,zbase,record['openingIndex'])

    output_glb=Path(output_glb);output_glb.parent.mkdir(parents=True,exist_ok=True)
    scene.export(output_glb)
    for side_record in review_sides:
        side_record.pop('conflicts',None)
    report={'prototype':'source-camera-gated host-shell aperture clipping','inputGlb':{'path':str(input_glb),'sha256':sha256(input_glb)},
            'candidateGlb':{'path':str(output_glb),'sha256':sha256(output_glb)},
            'profiles':{'path':str(profiles_path),'sha256':sha256(profiles_path)},
            'jobs':{'path':str(jobs_path),'sha256':sha256(jobs_path)},
            'criteria':{'matchedSideId':True,'matchedSourceCamera':True,
                        'openingOutlineOverlap':True,'sameFeatureBaseShellFirstHit':True,
                        'profileNormalAlignmentMin':.55,'localNormalSlabMaxWidthM':1.5,
                        'cutsLimitedToMeasuredContourAndLocalSlab':True},
            'sideCount':len(review_sides),'sides':review_sides,
            'modifications':modifications,
            'summary':{'conflicts':sum(s.get('supportedConflictCount',0) for s in review_sides),
                       'clipped':sum(m.get('status')=='clipped' for m in modifications),
                       'rejected':sum(m.get('status','').startswith('rejected') for m in modifications)}}
    Path(report_path).parent.mkdir(parents=True,exist_ok=True)
    Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-glb',required=True)
    parser.add_argument('--profiles',required=True)
    parser.add_argument('--jobs',required=True)
    parser.add_argument('--output-glb',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=build_candidate(args.input_glb,args.profiles,args.jobs,args.output_glb,args.report)
    print(json.dumps(report['summary']))


if __name__=='__main__':
    main()
